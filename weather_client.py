"""
weather_client.py
Gọi OpenWeatherMap để lấy nhiệt độ, độ ẩm, khả năng mưa theo tọa độ.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


def get_weather(lat: float, lon: float) -> dict:
    """
    Trả về dict: {temp, humidity, rain_next_hours, condition}
    Nếu lỗi (mất mạng, hết quota...) trả về None — nơi gọi hàm cần tự xử lý fallback.
    """
    if not WEATHER_API_KEY:
        return None

    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
            "units": "metric",
            "lang": "vi",
        }
        resp = requests.get(url, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()

        current = data["list"][0]
        # Kiểm tra 2 mốc dự báo gần nhất (~6 giờ tới) xem có mưa không
        next_slots = data["list"][:2]
        rain_next_hours = any(
            "rain" in slot.get("weather", [{}])[0].get("main", "").lower()
            for slot in next_slots
        )

        return {
            "temp": round(current["main"]["temp"]),
            "humidity": current["main"]["humidity"],
            "rain_next_hours": rain_next_hours,
            "condition": current["weather"][0]["description"],
        }
    except Exception as e:
        print(f"[weather_client] Lỗi khi gọi Weather API: {e}")
        return None
