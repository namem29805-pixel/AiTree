"""
rule_engine.py

Sinh khuyến nghị từ:
- loài do Model 1 dự đoán
- tình trạng do Model 2 dự đoán
- thời tiết từ OpenWeatherMap (nếu có)

Không dùng Gemini và không cần growth-stage model.
"""

import json
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent / "species_data.json"
with open(_DATA_PATH, encoding="utf-8") as f:
    SPECIES_DB = json.load(f)


CONDITION_ADVICE = {
    "healthy": {
        "time": "Sáng sớm",
        "amount_ml": None,
        "method": "Tưới gốc khi lớp đất mặt đã khô, tránh để chậu ngập nước",
        "reason": "Cây đang được nhận diện là khỏe mạnh. Duy trì lịch chăm sóc ổn định và quan sát độ ẩm đất trước khi tưới.",
        "fertilizer": "Phân NPK cân bằng pha loãng",
        "fertilizer_frequency": "Theo hướng dẫn sản phẩm, thường 3-4 tuần/lần trong giai đoạn sinh trưởng",
    },
    "dehydration_wilting": {
        "time": "Tưới sớm và kiểm tra độ ẩm đất",
        "amount_ml": None,
        "method": "Tưới chậm vào gốc đến khi giá thể ẩm đều; tránh dội nước mạnh lên lá",
        "reason": "Có dấu hiệu thiếu nước/héo. Kiểm tra đất trước; nếu đất khô, tưới từ từ và theo dõi cây trong 12-24 giờ.",
        "fertilizer": "Tạm chưa bón ngay khi cây đang héo",
        "fertilizer_frequency": "Bón lại khi cây phục hồi",
    },
    "overwatering_rot": {
        "time": "Tạm ngừng tưới",
        "amount_ml": 0,
        "method": "Kiểm tra lỗ thoát nước, giá thể và phần rễ; chỉ tưới lại khi đất đã khô phù hợp",
        "reason": "Có dấu hiệu tưới quá nhiều/úng. Tiếp tục tưới có thể làm tình trạng nặng hơn; ưu tiên tăng thoáng khí và thoát nước.",
        "fertilizer": "Tạm hoãn bón phân",
        "fertilizer_frequency": "Chỉ bón lại sau khi cây ổn định",
    },
    "leaf_spot_fungal": {
        "time": "Sáng sớm, chỉ tưới khi cần",
        "amount_ml": None,
        "method": "Tưới vào gốc, giữ lá khô và tăng thông thoáng",
        "reason": "Có dấu hiệu đốm lá do nấm. Hạn chế làm ướt lá, loại bỏ lá bệnh nặng và theo dõi sự lan rộng.",
        "fertilizer": "Tạm giảm hoặc hoãn bón nếu cây đang suy yếu",
        "fertilizer_frequency": "Bón lại khi tình trạng ổn định",
    },
    "nutrient_deficiency": {
        "time": "Tưới theo nhu cầu của cây",
        "amount_ml": None,
        "method": "Giữ ẩm vừa phải, tránh úng",
        "reason": "Có dấu hiệu thiếu dinh dưỡng. Nên kiểm tra pH/giá thể và bổ sung dinh dưỡng cân đối thay vì tăng liều đột ngột.",
        "fertilizer": "NPK cân bằng có vi lượng, dùng ở liều thấp trước",
        "fertilizer_frequency": "Theo hướng dẫn sản phẩm; đánh giá lại sau 1-2 tuần",
    },
    "sunburn": {
        "time": "Sáng sớm",
        "amount_ml": None,
        "method": "Tưới gốc vừa đủ; chuyển cây khỏi nắng gắt và tăng ánh sáng gián tiếp từ từ",
        "reason": "Có dấu hiệu cháy nắng. Tổn thương cũ không hồi xanh; mục tiêu là tránh lá mới tiếp tục bị cháy.",
        "fertilizer": "Không cần tăng phân chỉ vì cháy nắng",
        "fertilizer_frequency": "Duy trì lịch bình thường sau khi cây ổn định",
    },
    "uncertain": {
        "time": "Kiểm tra đất trước khi tưới",
        "amount_ml": None,
        "method": "Sờ đất sâu 2-3 cm, xem mặt dưới lá và gốc; chưa nên can thiệp mạnh (đổi chậu, cắt tỉa, bón phân)",
        "reason": "",
        "fertilizer": "Tạm chưa bón cho đến khi xác định rõ tình trạng",
        "fertilizer_frequency": "Đánh giá lại sau khi chụp ảnh mới",
    },
}


def build_recommendation(analysis, weather: dict | None) -> dict:
    species_info = SPECIES_DB.get(analysis.species_key or "", {})
    category = species_info.get("category", "leaf")
    uncertain = bool(getattr(analysis, "condition_uncertain", False))
    advice = dict(CONDITION_ADVICE["uncertain" if uncertain else analysis.condition_key]
                  if uncertain or analysis.condition_key in CONDITION_ADVICE else CONDITION_ADVICE["healthy"])
    if uncertain:
        names = [t["name_vi"] for t in (getattr(analysis, "top_condition", None) or [])[:2]]
        lean = f" (nghiêng về: {' hoặc '.join(names)})" if names else ""
        advice["reason"] = (
            f"Kết quả chưa đủ chắc chắn{lean}. Hãy kiểm tra trực tiếp độ ẩm đất, mặt dưới lá và rễ, "
            "rồi chụp lại ảnh gần hơn, đủ sáng trước khi làm theo gợi ý."
        )

    reasons = [advice["reason"]]

    # Thời tiết chỉ dùng như tín hiệu phụ. Với cây trong nhà, người dùng vẫn phải
    # kiểm tra đất/giá thể trực tiếp.
    if weather:
        if weather.get("rain_next_hours") and analysis.condition_key != "dehydration_wilting":
            reasons.append("Nếu cây đặt ngoài trời/ban công, sắp có mưa nên chưa cần tưới thêm.")
        if weather.get("temp", 0) >= 34 and analysis.condition_key not in {"overwatering_rot", "leaf_spot_fungal"}:
            reasons.append(f"Nhiệt độ ngoài trời cao ({weather['temp']}°C); tránh nắng gắt và kiểm tra độ ẩm đất thường xuyên hơn.")
        if weather.get("humidity", 0) >= 80 and analysis.condition_key == "leaf_spot_fungal":
            reasons.append(f"Độ ẩm cao ({weather['humidity']}%) có thể làm bệnh nấm dễ lan; cần tăng thông thoáng.")
    else:
        reasons.append("Không lấy được dữ liệu thời tiết; khuyến nghị dựa trên ảnh và tình trạng cây.")

    return {
        "species": analysis.species_common_vi,
        "species_key": analysis.species_key,
        "species_latin": species_info.get("species_latin"),
        "category": category,
        "health_status": analysis.health_status,
        "condition": analysis.condition_key,
        "uncertain": uncertain,
        "condition_name_vi": analysis.condition_name_vi,
        # Giữ field disease_name_vi để frontend cũ/logic cũ không vỡ hoàn toàn.
        "disease_name_vi": None if analysis.condition_key == "healthy" else analysis.condition_name_vi,
        "weather": weather,
        "recommendation": {
            "time": advice["time"],
            "amount_ml": advice["amount_ml"],
            "method": advice["method"],
            "reason": " ".join(reasons),
        },
        "fertilizer_recommendation": {
            "type": advice["fertilizer"],
            "frequency": advice["fertilizer_frequency"],
            "note": "Khuyến nghị tham khảo; cần điều chỉnh theo kích thước chậu, giá thể và loài cụ thể.",
        },
    }
