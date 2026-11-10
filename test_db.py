"""Kiểm thử db.py. Chạy: python test_db.py (hoặc pytest test_db.py). Dùng DB tạm, không đụng dữ liệu thật."""
import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
import db  # noqa: E402

A, B = "client-aaaaaaaa", "client-bbbbbbbb"
NOW = 1_800_000_000_000
ENTRY = {"t": NOW, "cond": "healthy", "condVi": "Khỏe mạnh", "hp": 0.91, "unc": False, "thumb": "data:image/jpeg;base64,AAAA"}


def test_all():
    db.init_db()
    db.init_db()  # chạy lại không lỗi

    assert db.valid_client(A) and not db.valid_client("ab") and not db.valid_client("a b c d e f g h")

    # tạo cây mới, rồi thêm lần chụp vào cây đó
    r = db.add_entry(A, None, "Monstera ban công", "monstera_deliciosa", "Trầu bà lá xẻ", {"interval_days": 7}, ENTRY)
    pid = r["id"]
    assert r["name"] == "Monstera ban công"
    db.add_entry(A, pid, "", "", "", None, {**ENTRY, "t": NOW + 1000, "cond": "overwatering_rot", "hp": 0.2})

    plants = db.list_plants(A)
    assert len(plants) == 1 and [e["cond"] for e in plants[0]["entries"]] == ["healthy", "overwatering_rot"]
    assert plants[0]["plan"] == {"interval_days": 7}  # plan=None ở lần 2 không xóa plan cũ
    assert plants[0]["entries"][0]["thumb"].startswith("data:image/")

    # cô lập theo client
    assert db.list_plants(B) == []
    try:
        db.add_entry(B, pid, "", "", "", None, ENTRY)
        assert False, "B không được thêm vào cây của A"
    except LookupError:
        pass
    assert db.delete_plant(B, pid) is False

    # sự kiện
    db.add_event(A, pid, "watered", NOW + 5000)
    db.add_event(A, pid, "fed")
    p = db.list_plants(A)[0]
    assert p["watered"] == [NOW + 5000] and len(p["fed"]) == 1
    try:
        db.add_event(A, pid, "bad")
        assert False
    except ValueError:
        pass

    # dữ liệu xấu bị làm sạch, không làm hỏng
    db.add_entry(A, pid, "", "", "", None, {"t": "abc", "cond": "", "hp": 5, "thumb": "javascript:x"})
    last = [e for e in db.list_plants(A)[0]["entries"] if e["cond"] == "unknown"][0]
    assert last["hp"] == 1.0 and last["thumb"] is None

    # import có tính idempotent
    old = [{"name": "Cây cũ", "species": "aloe_vera", "speciesVi": "Nha đam", "entries": [ENTRY], "watered": [NOW], "fed": []}]
    assert db.import_plants(B, old) == {"added": 1, "skipped": 0}
    assert db.import_plants(B, old) == {"added": 0, "skipped": 1}
    assert db.list_plants(B)[0]["watered"] == [NOW]

    # xóa cây kéo theo entries/events (ON DELETE CASCADE)
    assert db.delete_plant(A, pid) is True
    with db.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM entries WHERE plant_id = ?", (pid,)).fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM events WHERE plant_id = ?", (pid,)).fetchone()[0] == 0

    # giới hạn số cây
    for i in range(db.MAX_PLANTS_PER_CLIENT):
        db.add_entry("client-cccccccc", None, f"c{i}", "", "", None, ENTRY)
    try:
        db.add_entry("client-cccccccc", None, "thừa", "", "", None, ENTRY)
        assert False
    except ValueError:
        pass

    # feedback: chống trùng ảnh
    sha = db.sha1_bytes(b"abc")
    assert db.add_feedback("x.jpg", sha, "wrong", "aloe_vera", "healthy", "aloe_vera", "sunburn") is True
    assert db.add_feedback("y.jpg", sha, "wrong", "aloe_vera", "healthy", "aloe_vera", "sunburn") is False
    assert db.feedback_exists(sha)
    assert len(db.feedback_rows()) == 1 and db.feedback_rows(include_correct=True)[0]["label_condition"] == "sunburn"


if __name__ == "__main__":
    test_all()
    print("Tất cả kiểm thử đạt.")
