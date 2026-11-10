"""
db.py
Lớp lưu trữ SQLite cho Lá Xanh. Chỉ dùng thư viện chuẩn (sqlite3).

Bảng:
  plants    cây trong nhật ký (thuộc một client_id = một trình duyệt)
  entries   mỗi lần chẩn đoán được lưu cho một cây
  events    ngày tưới / bón phân do người dùng ghi
  feedback  ảnh và nhãn người dùng gửi (đúng/sai) để duyệt và train lại

Không có đăng nhập: mỗi trình duyệt tự sinh một client_id (UUID) và gửi qua header
X-Client-Id, nên mỗi người chỉ thấy cây của mình. Đây là cách ngăn nhầm dữ liệu giữa
các người dùng, KHÔNG phải bảo mật: ai biết client_id thì đọc được dữ liệu.

Đường dẫn DB: biến môi trường DB_PATH, mặc định data/laxanh.db cạnh file này.
"""

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "data" / "laxanh.db"))

CLIENT_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
MAX_PLANTS_PER_CLIENT = 50
MAX_ENTRIES_PER_PLANT = 500
MAX_EVENTS_PER_PLANT = 200
RETURN_ENTRIES = 60
RETURN_EVENTS = 30
MAX_THUMB_CHARS = 60_000
MAX_PLAN_CHARS = 8_000
EVENT_KINDS = ("watered", "fed")

SCHEMA = """
CREATE TABLE IF NOT EXISTS plants (
  id          TEXT PRIMARY KEY,
  client_id   TEXT NOT NULL,
  name        TEXT NOT NULL,
  species_key TEXT NOT NULL DEFAULT '',
  species_vi  TEXT NOT NULL DEFAULT '',
  plan_json   TEXT,
  created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plants_client ON plants(client_id, created_at);

CREATE TABLE IF NOT EXISTS entries (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  plant_id     TEXT NOT NULL REFERENCES plants(id) ON DELETE CASCADE,
  taken_at     INTEGER NOT NULL,
  condition    TEXT NOT NULL,
  condition_vi TEXT NOT NULL DEFAULT '',
  healthy_prob REAL,
  uncertain    INTEGER NOT NULL DEFAULT 0,
  thumb        TEXT
);
CREATE INDEX IF NOT EXISTS idx_entries_plant ON entries(plant_id, taken_at);

CREATE TABLE IF NOT EXISTS events (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  plant_id TEXT NOT NULL REFERENCES plants(id) ON DELETE CASCADE,
  kind     TEXT NOT NULL CHECK (kind IN ('watered', 'fed')),
  at       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_plant ON events(plant_id, kind, at);

CREATE TABLE IF NOT EXISTS feedback (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at          INTEGER NOT NULL,
  image_name          TEXT NOT NULL,
  image_sha1          TEXT NOT NULL UNIQUE,
  verdict             TEXT NOT NULL CHECK (verdict IN ('correct', 'wrong')),
  predicted_species   TEXT NOT NULL DEFAULT '',
  predicted_condition TEXT NOT NULL DEFAULT '',
  label_species       TEXT NOT NULL DEFAULT '',
  label_condition     TEXT NOT NULL DEFAULT '',
  reviewed            INTEGER NOT NULL DEFAULT 0
);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    with connect() as con:
        con.execute("PRAGMA journal_mode = WAL")
        con.executescript(SCHEMA)
        con.execute("PRAGMA user_version = 1")


def valid_client(client_id: str) -> bool:
    return bool(client_id and CLIENT_RE.match(client_id))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clean_ts(t, default=None) -> int:
    try:
        t = int(t)
    except (TypeError, ValueError):
        return default if default is not None else _now_ms()
    # 2000-01-01 .. 2100-01-01 (mili giây)
    return t if 946_684_800_000 <= t <= 4_102_444_800_000 else (default if default is not None else _now_ms())


def _clean_entry(e: dict) -> dict:
    thumb = e.get("thumb")
    if not (isinstance(thumb, str) and thumb.startswith("data:image/") and len(thumb) <= MAX_THUMB_CHARS):
        thumb = None
    hp = e.get("hp")
    try:
        hp = None if hp is None else min(1.0, max(0.0, float(hp)))
    except (TypeError, ValueError):
        hp = None
    return {
        "taken_at": _clean_ts(e.get("t")),
        "condition": str(e.get("cond") or "")[:40] or "unknown",
        "condition_vi": str(e.get("condVi") or "")[:80],
        "healthy_prob": hp,
        "uncertain": 1 if e.get("unc") else 0,
        "thumb": thumb,
    }


def _plan_json(plan) -> str | None:
    if not isinstance(plan, dict):
        return None
    s = json.dumps(plan, ensure_ascii=False)
    return s if len(s) <= MAX_PLAN_CHARS else None


def _own_plant(con, client_id: str, plant_id: str):
    return con.execute(
        "SELECT * FROM plants WHERE id = ? AND client_id = ?", (plant_id, client_id)
    ).fetchone()


def _insert_entry(con, plant_id: str, e: dict) -> None:
    c = _clean_entry(e)
    con.execute(
        "INSERT INTO entries (plant_id, taken_at, condition, condition_vi, healthy_prob, uncertain, thumb) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (plant_id, c["taken_at"], c["condition"], c["condition_vi"], c["healthy_prob"], c["uncertain"], c["thumb"]),
    )
    con.execute(
        "DELETE FROM entries WHERE plant_id = ? AND id NOT IN "
        "(SELECT id FROM entries WHERE plant_id = ? ORDER BY taken_at DESC, id DESC LIMIT ?)",
        (plant_id, plant_id, MAX_ENTRIES_PER_PLANT),
    )


def _insert_event(con, plant_id: str, kind: str, at: int) -> None:
    con.execute("INSERT INTO events (plant_id, kind, at) VALUES (?, ?, ?)", (plant_id, kind, at))
    con.execute(
        "DELETE FROM events WHERE plant_id = ? AND kind = ? AND id NOT IN "
        "(SELECT id FROM events WHERE plant_id = ? AND kind = ? ORDER BY at DESC, id DESC LIMIT ?)",
        (plant_id, kind, plant_id, kind, MAX_EVENTS_PER_PLANT),
    )


def _new_plant(con, client_id: str, name: str, species_key: str, species_vi: str, plan) -> str:
    n = con.execute("SELECT COUNT(*) FROM plants WHERE client_id = ?", (client_id,)).fetchone()[0]
    if n >= MAX_PLANTS_PER_CLIENT:
        raise ValueError(f"Mỗi trình duyệt chỉ lưu tối đa {MAX_PLANTS_PER_CLIENT} cây.")
    pid = "p" + uuid.uuid4().hex[:12]
    name = (name or "").strip()[:40] or (species_vi or "").strip()[:40] or "Cây của tôi"
    con.execute(
        "INSERT INTO plants (id, client_id, name, species_key, species_vi, plan_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (pid, client_id, name, (species_key or "")[:60], (species_vi or "")[:80], _plan_json(plan), _now_ms()),
    )
    return pid


# ---------------------------------------------------------------- nhật ký
def list_plants(client_id: str) -> list[dict]:
    out = []
    with connect() as con:
        plants = con.execute(
            "SELECT * FROM plants WHERE client_id = ? ORDER BY created_at, id", (client_id,)
        ).fetchall()
        for p in plants:
            entries = con.execute(
                "SELECT taken_at, condition, condition_vi, healthy_prob, uncertain, thumb FROM "
                "(SELECT * FROM entries WHERE plant_id = ? ORDER BY taken_at DESC, id DESC LIMIT ?) "
                "ORDER BY taken_at, id",
                (p["id"], RETURN_ENTRIES),
            ).fetchall()
            ev = {}
            for kind in EVENT_KINDS:
                rows = con.execute(
                    "SELECT at FROM (SELECT at, id FROM events WHERE plant_id = ? AND kind = ? "
                    "ORDER BY at DESC, id DESC LIMIT ?) ORDER BY at, id",
                    (p["id"], kind, RETURN_EVENTS),
                ).fetchall()
                ev[kind] = [r["at"] for r in rows]
            out.append({
                "id": p["id"],
                "name": p["name"],
                "species": p["species_key"],
                "speciesVi": p["species_vi"],
                "plan": json.loads(p["plan_json"]) if p["plan_json"] else None,
                "entries": [{
                    "t": e["taken_at"], "cond": e["condition"], "condVi": e["condition_vi"],
                    "hp": e["healthy_prob"], "unc": bool(e["uncertain"]), "thumb": e["thumb"],
                } for e in entries],
                "watered": ev["watered"],
                "fed": ev["fed"],
            })
    return out


def add_entry(client_id: str, plant_id, name: str, species_key: str, species_vi: str, plan, entry: dict) -> dict:
    """Thêm một lần chẩn đoán vào cây. Nếu plant_id rỗng thì tạo cây mới."""
    with connect() as con:
        if plant_id:
            p = _own_plant(con, client_id, plant_id)
            if not p:
                raise LookupError("Không tìm thấy cây.")
            pid, pname = p["id"], p["name"]
            if _plan_json(plan):
                con.execute("UPDATE plants SET plan_json = ? WHERE id = ?", (_plan_json(plan), pid))
        else:
            pid = _new_plant(con, client_id, name, species_key, species_vi, plan)
            pname = con.execute("SELECT name FROM plants WHERE id = ?", (pid,)).fetchone()[0]
        _insert_entry(con, pid, entry)
    return {"id": pid, "name": pname}


def add_event(client_id: str, plant_id: str, kind: str, at=None) -> None:
    if kind not in EVENT_KINDS:
        raise ValueError("kind phải là 'watered' hoặc 'fed'.")
    with connect() as con:
        if not _own_plant(con, client_id, plant_id):
            raise LookupError("Không tìm thấy cây.")
        _insert_event(con, plant_id, kind, _clean_ts(at))


def delete_plant(client_id: str, plant_id: str) -> bool:
    with connect() as con:
        cur = con.execute("DELETE FROM plants WHERE id = ? AND client_id = ?", (plant_id, client_id))
        return cur.rowcount > 0


def import_plants(client_id: str, plants: list[dict]) -> dict:
    """Chuyển nhật ký cũ từ localStorage. Bỏ qua cây đã có (trùng mốc thời gian lần chụp đầu)."""
    added = skipped = 0
    with connect() as con:
        for p in plants[:MAX_PLANTS_PER_CLIENT]:
            entries = [e for e in (p.get("entries") or []) if isinstance(e, dict)][-MAX_ENTRIES_PER_PLANT:]
            if entries:
                first_t = _clean_entry(entries[0])["taken_at"]
                dup = con.execute(
                    "SELECT 1 FROM entries e JOIN plants pl ON pl.id = e.plant_id "
                    "WHERE pl.client_id = ? AND e.taken_at = ? LIMIT 1", (client_id, first_t)
                ).fetchone()
                if dup:
                    skipped += 1
                    continue
            pid = _new_plant(con, client_id, p.get("name"), p.get("species"), p.get("speciesVi"), p.get("plan"))
            for e in entries:
                _insert_entry(con, pid, e)
            for kind, key in (("watered", "watered"), ("fed", "fed")):
                for at in (p.get(key) or [])[-MAX_EVENTS_PER_PLANT:]:
                    _insert_event(con, pid, kind, _clean_ts(at))
            added += 1
    return {"added": added, "skipped": skipped}


# ---------------------------------------------------------------- phản hồi
def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def feedback_exists(sha1: str) -> bool:
    with connect() as con:
        return con.execute("SELECT 1 FROM feedback WHERE image_sha1 = ?", (sha1,)).fetchone() is not None


def add_feedback(image_name: str, sha1: str, verdict: str, predicted_species: str, predicted_condition: str,
                 label_species: str, label_condition: str) -> bool:
    """Trả về False nếu ảnh này (cùng sha1) đã có."""
    with connect() as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO feedback (created_at, image_name, image_sha1, verdict, predicted_species, "
            "predicted_condition, label_species, label_condition) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_now_ms(), image_name, sha1, verdict, predicted_species, predicted_condition,
             label_species, label_condition),
        )
        return cur.rowcount > 0


def feedback_rows(include_correct: bool = False, only_unreviewed: bool = False) -> list[sqlite3.Row]:
    sql = "SELECT * FROM feedback WHERE 1=1"
    if not include_correct:
        sql += " AND verdict = 'wrong'"
    if only_unreviewed:
        sql += " AND reviewed = 0"
    with connect() as con:
        return con.execute(sql + " ORDER BY id").fetchall()


def import_legacy_feedback(feedback_dir: Path) -> int:
    """Nhập feedback.jsonl của bản cũ (nếu có) vào DB một lần, rồi đổi tên file."""
    log = feedback_dir / "feedback.jsonl"
    if not log.exists():
        return 0
    n = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            img = feedback_dir / "images" / r["image"]
            if add_feedback(r["image"], sha1_bytes(img.read_bytes()), r["verdict"],
                            r.get("predicted_species", ""), r.get("predicted_condition", ""),
                            r.get("label_species", ""), r.get("label_condition", "")):
                n += 1
        except (KeyError, OSError, ValueError):
            continue
    log.rename(log.with_suffix(".jsonl.migrated"))
    return n
