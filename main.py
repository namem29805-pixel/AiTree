"""
main.py
Backend FastAPI cho ứng dụng Lá Xanh — phiên bản KHÔNG GEMINI.

Chạy:
    uvicorn main:app --reload

Mở web:
    http://127.0.0.1:8000

API docs:
    http://127.0.0.1:8000/docs
"""

import inspect
import io
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

import db
from cnn_classifier import analyze_image, get_model_status
from rule_engine import CONDITION_ADVICE, SPECIES_DB, build_recommendation
from weather_client import get_weather

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_DIR = BASE_DIR / "feedback"
MAX_FEEDBACK_BYTES = 8 * 1024 * 1024
VALID_CONDITIONS = set(CONDITION_ADVICE) - {"uncertain"}

# Cho phép dùng được cả khi rule_engine là bản cũ (chưa có hồ sơ chăm sóc theo loài)
_CARE_AWARE = len(inspect.signature(build_recommendation).parameters) > 2

db.init_db()
db.import_legacy_feedback(FEEDBACK_DIR)

app = FastAPI(title="Lá Xanh API - 2 CNN Models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def web_home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/status")
def api_status():
    return {
        "status": "ok",
        "message": "Lá Xanh API đang chạy bằng 2 model CNN, không dùng Gemini.",
        "models": get_model_status(),
    }


@app.get("/species")
def list_species():
    return SPECIES_DB


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    placement: str = Form("indoor"),
    pot: str = Form("medium"),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Ảnh trống hoặc không đọc được.")

    try:
        analysis = await run_in_threadpool(
            analyze_image,
            image_bytes,
            mime_type=image.content_type or "image/jpeg",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi chạy model CNN: {e}")

    if not analysis.is_plant:
        return {
            "is_plant": False,
            "message": analysis.notes or "Ảnh không đủ tin cậy để nhận diện.",
            "species_confidence": analysis.species_confidence,
            "condition_confidence": analysis.condition_confidence,
        }

    weather = await run_in_threadpool(get_weather, lat, lon)
    result = (build_recommendation(analysis, weather, placement, pot)
              if _CARE_AWARE else build_recommendation(analysis, weather))
    result.update(
        {
            "is_plant": True,
            "species_confidence": analysis.species_confidence,
            "condition_confidence": analysis.condition_confidence,
            "notes": analysis.notes,
            "top_species": analysis.top_species,
            "top_condition": analysis.top_condition,
            "condition_probs": analysis.condition_probs,
            "gradcam": analysis.gradcam,
        }
    )
    return result


# ------------------------------------------------------------------ phản hồi
@app.post("/feedback")
async def feedback(
    image: UploadFile = File(...),
    verdict: str = Form(...),
    predicted_species: str = Form(""),
    predicted_condition: str = Form(""),
    correct_species: str = Form(""),
    correct_condition: str = Form(""),
):
    """Nhận phản hồi đúng/sai của người dùng, lưu ảnh + nhãn vào SQLite để duyệt và train lại sau."""
    if verdict not in {"correct", "wrong"}:
        raise HTTPException(status_code=400, detail="verdict phải là 'correct' hoặc 'wrong'.")
    for val, valid, name in (
        (correct_species, set(SPECIES_DB), "correct_species"),
        (correct_condition, VALID_CONDITIONS, "correct_condition"),
    ):
        if val and val not in valid:
            raise HTTPException(status_code=400, detail=f"{name} không hợp lệ.")

    data = await image.read()
    if not data or len(data) > MAX_FEEDBACK_BYTES:
        raise HTTPException(status_code=400, detail="Ảnh trống hoặc lớn hơn 8 MB.")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Tệp không phải ảnh hợp lệ.")

    if verdict == "correct":
        label_species, label_condition = predicted_species, predicted_condition
    else:
        label_species = correct_species or predicted_species
        label_condition = correct_condition or predicted_condition

    sha1 = db.sha1_bytes(data)
    if db.feedback_exists(sha1):
        return {"ok": True, "duplicate": True}
    name = f"{sha1}.jpg"
    (FEEDBACK_DIR / "images").mkdir(parents=True, exist_ok=True)
    (FEEDBACK_DIR / "images" / name).write_bytes(data)
    db.add_feedback(name, sha1, verdict, predicted_species, predicted_condition, label_species, label_condition)
    return {"ok": True}


# ------------------------------------------------------------------ nhật ký (SQLite)
def current_client(x_client_id: str = Header("")) -> str:
    if not db.valid_client(x_client_id):
        raise HTTPException(status_code=400, detail="Thiếu hoặc sai header X-Client-Id.")
    return x_client_id


def _dump(m) -> dict:
    return m.model_dump() if hasattr(m, "model_dump") else m.dict()


class EntryData(BaseModel):
    t: int
    cond: str = ""
    condVi: str = ""
    hp: Optional[float] = None
    unc: bool = False
    thumb: Optional[str] = None


class EntryIn(BaseModel):
    plant_id: Optional[str] = None
    name: str = Field("", max_length=40)
    species_key: str = Field("", max_length=60)
    species_vi: str = Field("", max_length=80)
    plan: Optional[dict] = None
    entry: EntryData


class EventIn(BaseModel):
    kind: str


class ImportPlant(BaseModel):
    name: str = ""
    species: str = ""
    speciesVi: str = ""
    plan: Optional[dict] = None
    entries: list[EntryData] = []
    watered: list[int] = []
    fed: list[int] = []


class ImportIn(BaseModel):
    plants: list[ImportPlant] = []


@app.get("/api/plants")
def api_list_plants(client: str = Depends(current_client)):
    return {"plants": db.list_plants(client)}


@app.post("/api/entries")
def api_add_entry(body: EntryIn, client: str = Depends(current_client)):
    try:
        return db.add_entry(client, body.plant_id, body.name, body.species_key, body.species_vi,
                            body.plan, _dump(body.entry))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/plants/{plant_id}/events")
def api_add_event(plant_id: str, body: EventIn, client: str = Depends(current_client)):
    try:
        db.add_event(client, plant_id, body.kind)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.delete("/api/plants/{plant_id}")
def api_delete_plant(plant_id: str, client: str = Depends(current_client)):
    if not db.delete_plant(client, plant_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy cây.")
    return {"ok": True}


@app.post("/api/import")
def api_import(body: ImportIn, client: str = Depends(current_client)):
    try:
        return db.import_plants(client, [_dump(p) for p in body.plants])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
