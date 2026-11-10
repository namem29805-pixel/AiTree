# Lá Xanh — tích hợp 2 model CNN, không Gemini

## 1. File model cần có

Đặt vào `models/`:

- `species_model.keras` — Model 1, nhận diện 9 loài
- `labels_species.json`
- `model2_6classes.keras` — Model 2, nhận diện 6 tình trạng
- `model2_labels.json`

## 2. Tạo môi trường Python

Khuyên dùng Python 3.11 hoặc 3.12 nếu máy gặp lỗi cài TensorFlow.

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Weather API (không bắt buộc)

Copy `.env.example` thành `.env`:

```text
WEATHER_API_KEY=...
```

Nếu không có key, hệ thống vẫn chạy nhận diện bằng 2 model CNN.

## 4. Chạy web

```powershell
uvicorn main:app --reload
```

Mở:

```text
http://127.0.0.1:8000
```

Kiểm tra model:

```text
http://127.0.0.1:8000/api/status
```

Cả 4 giá trị model/labels nên là `true`.

## 5. Luồng hoạt động

```text
Trình duyệt
  -> POST /analyze (ảnh + vị trí)
  -> Model 1: loài cây
  -> Model 2: tình trạng cây
  -> rule_engine: khuyến nghị
  -> weather_client: thời tiết (nếu có key)
  -> JSON trả về frontend
```

## 6. Gemini đã được bỏ

Không còn:
- `gemini_client.py`
- `test_gemini.py`
- `google-genai`
- `GEMINI_API_KEY`
- endpoint `/chat`
- khung chat Gemini trên frontend
