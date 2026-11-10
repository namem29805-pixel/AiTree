"""
cnn_classifier.py

Chạy 2 model CNN tự train:
- Model 1: nhận diện 9 loài cây (MobileNetV2)
- Model 2: nhận diện 6 tình trạng cây (EfficientNetB0)

Không dùng Gemini / API AI bên ngoài.
"""

import io
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL1_PATH = MODELS_DIR / "species_model.keras"
MODEL1_LABELS_PATH = MODELS_DIR / "labels_species.json"
MODEL2_PATH = MODELS_DIR / "model2_6classes.keras"
MODEL2_LABELS_PATH = MODELS_DIR / "model2_labels.json"

# Độ tin cậy thấp hơn ngưỡng này sẽ cảnh báo. Đây KHÔNG phải mô hình phát hiện
# ảnh không phải cây; 2 model hiện tại là closed-set classification.
LOW_CONFIDENCE_THRESHOLD = 0.50
NOT_PLANT_APPROX_THRESHOLD = 0.25
# Chênh lệch top-1 và top-2 nhỏ hơn mức này => coi là chưa chắc chắn
MARGIN_THRESHOLD = 0.15

SPECIES_VI = {
    "aeonium": "Sen đá Aeonium",
    "aloe_vera": "Nha đam",
    "calathea_lancifolia": "Đuôi công lá dài",
    "calathea_orbifolia": "Đuôi công lá tròn",
    "dracaena_marginata": "Phất dụ mảnh",
    "epiphyllum_anguliger": "Xương rồng xương cá",
    "monstera_deliciosa": "Trầu bà lá xẻ",
    "parlor_palm": "Cau tiểu trâm",
    "sansevieria": "Lưỡi hổ",
}

CONDITION_VI = {
    "healthy": "Khỏe mạnh",
    "dehydration_wilting": "Thiếu nước / Héo",
    "leaf_spot_fungal": "Đốm lá do nấm",
    "nutrient_deficiency": "Thiếu dinh dưỡng",
    "overwatering_rot": "Tưới quá nhiều / Úng",
    "sunburn": "Cháy nắng"
}


class PlantAnalysis(BaseModel):
    is_plant: bool = True
    species_key: Optional[str] = None
    species_common_vi: Optional[str] = None
    condition_key: Optional[str] = None
    condition_name_vi: Optional[str] = None
    health_status: Optional[str] = None
    species_confidence: Optional[float] = None
    condition_confidence: Optional[float] = None
    notes: Optional[str] = None
    top_species: list = []
    top_condition: list = []
    condition_probs: dict = {}
    condition_uncertain: bool = False
    gradcam: Optional[str] = None


_species_model = None
_condition_model = None
_species_labels = None
_condition_labels = None


def _normalize_label(value: str) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s


def _load_labels(path: Path) -> list[str]:
    """Hỗ trợ JSON dạng list, {index: label}, {label: index}, hoặc {classes:[...]}."""
    if not path.exists():
        raise FileNotFoundError(f"Chưa tìm thấy file nhãn: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [_normalize_label(x) for x in data]

    if isinstance(data, dict):
        if "classes" in data and isinstance(data["classes"], list):
            return [_normalize_label(x) for x in data["classes"]]

        # {"0": "aeonium", "1": "aloe_vera", ...}
        if all(str(k).isdigit() for k in data.keys()):
            items = sorted(data.items(), key=lambda kv: int(kv[0]))
            return [_normalize_label(v) for _, v in items]

        # {"aeonium": 0, "aloe_vera": 1, ...}
        if all(isinstance(v, int) for v in data.values()):
            items = sorted(data.items(), key=lambda kv: kv[1])
            return [_normalize_label(k) for k, _ in items]

        # fallback: giữ thứ tự insertion của values nếu là string
        if all(isinstance(v, str) for v in data.values()):
            return [_normalize_label(v) for v in data.values()]

    raise ValueError(f"Định dạng labels không hỗ trợ: {path}")


def _load_models():
    global _species_model, _condition_model, _species_labels, _condition_labels

    if _species_labels is None:
        _species_labels = _load_labels(MODEL1_LABELS_PATH)
    if _condition_labels is None:
        _condition_labels = _load_labels(MODEL2_LABELS_PATH)

    if _species_model is None or _condition_model is None:
        import tensorflow as tf

        if not MODEL1_PATH.exists():
            raise FileNotFoundError(
                f"Thiếu {MODEL1_PATH.name}. Hãy đặt Model 1 vào thư mục models/."
            )
        if not MODEL2_PATH.exists():
            raise FileNotFoundError(
                f"Thiếu {MODEL2_PATH.name}. Hãy đặt Model 2 vào thư mục models/."
            )

        if _species_model is None:
            _species_model = tf.keras.models.load_model(MODEL1_PATH, compile=False)
        if _condition_model is None:
            _condition_model = tf.keras.models.load_model(MODEL2_PATH, compile=False)

        if int(_species_model.output_shape[-1]) != len(_species_labels):
            raise ValueError(
                f"Model 1 có {_species_model.output_shape[-1]} outputs nhưng labels có "
                f"{len(_species_labels)} lớp."
            )
        if int(_condition_model.output_shape[-1]) != len(_condition_labels):
            raise ValueError(
                f"Model 2 có {_condition_model.output_shape[-1]} outputs nhưng labels có "
                f"{len(_condition_labels)} lớp."
            )


def _input_size(model) -> tuple[int, int]:
    shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    h = int(shape[1]) if shape[1] else 224
    w = int(shape[2]) if shape[2] else 224
    return h, w


def _open_rgb(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Không đọc được ảnh: {e}") from e


def _preprocess_species(img: Image.Image) -> np.ndarray:
    """Model 1 MobileNetV2: dùng preprocess_input như lúc test/training pipeline."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    h, w = _input_size(_species_model)
    resized = img.resize((w, h))
    arr = np.asarray(resized, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def _preprocess_condition(img: Image.Image) -> np.ndarray:
    """
    Model 2 EfficientNetB0: truyền RGB 0..255.
    EfficientNet Keras có preprocessing nội bộ trong pipeline model hiện tại.
    """
    h, w = _input_size(_condition_model)
    resized = img.resize((w, h))
    arr = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def _top_prediction(probs: np.ndarray, labels: list[str]) -> tuple[str, float]:
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])


def _topk(probs: np.ndarray, labels: list[str], names: dict, k: int = 3) -> list[dict]:
    order = np.argsort(probs)[::-1][:k]
    return [
        {"key": labels[i], "name_vi": names.get(labels[i], labels[i]), "prob": round(float(probs[i]), 4)}
        for i in order
    ]


def get_model_status() -> dict:
    return {
        "species_model": MODEL1_PATH.exists(),
        "species_labels": MODEL1_LABELS_PATH.exists(),
        "condition_model": MODEL2_PATH.exists(),
        "condition_labels": MODEL2_LABELS_PATH.exists(),
    }


def analyze_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> PlantAnalysis:
    _load_models()
    img = _open_rgb(image_bytes)

    x1 = _preprocess_species(img)
    species_probs = _species_model.predict(x1, verbose=0)[0]
    species_key, species_conf = _top_prediction(species_probs, _species_labels)

    x2 = _preprocess_condition(img)
    condition_probs = _condition_model.predict(x2, verbose=0)[0]
    condition_key, condition_conf = _top_prediction(condition_probs, _condition_labels)

    # Chỉ là bộ lọc gần đúng. Muốn phát hiện ảnh không phải cây tốt cần model thứ 3
    # hoặc thêm lớp "not_plant" vào tập train.
    if species_conf < NOT_PLANT_APPROX_THRESHOLD:
        return PlantAnalysis(
            is_plant=False,
            species_confidence=species_conf,
            condition_confidence=condition_conf,
            notes="Độ tin cậy nhận diện loài quá thấp. Ảnh có thể không rõ hoặc nằm ngoài 9 loài hệ thống hỗ trợ.",
        )

    top_species = _topk(species_probs, _species_labels, SPECIES_VI)
    top_condition = _topk(condition_probs, _condition_labels, CONDITION_VI)
    margin = top_condition[0]["prob"] - (top_condition[1]["prob"] if len(top_condition) > 1 else 0.0)
    condition_uncertain = condition_conf < LOW_CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD

    gradcam = None
    try:
        from gradcam import gradcam_overlay

        gradcam = gradcam_overlay(_condition_model, x2, _condition_labels.index(condition_key), img)
    except Exception as e:  # Grad-CAM chỉ là phần giải thích, lỗi thì bỏ qua
        print(f"[gradcam] bỏ qua: {e}")

    notes = []
    if species_conf < LOW_CONFIDENCE_THRESHOLD:
        notes.append("Độ tin cậy nhận diện loài thấp.")
    if condition_conf < LOW_CONFIDENCE_THRESHOLD:
        notes.append("Độ tin cậy nhận diện tình trạng thấp.")
    elif condition_uncertain:
        notes.append("Hai khả năng về tình trạng khá gần nhau.")

    health_status = "healthy" if condition_key == "healthy" else "diseased"

    return PlantAnalysis(
        is_plant=True,
        species_key=species_key,
        species_common_vi=SPECIES_VI.get(species_key, species_key),
        condition_key=condition_key,
        condition_name_vi=CONDITION_VI.get(condition_key, condition_key),
        health_status=health_status,
        species_confidence=species_conf,
        condition_confidence=condition_conf,
        notes=" ".join(notes) if notes else None,
        top_species=top_species,
        top_condition=top_condition,
        condition_probs={k: round(float(p), 4) for k, p in zip(_condition_labels, condition_probs)},
        condition_uncertain=condition_uncertain,
        gradcam=gradcam,
    )
