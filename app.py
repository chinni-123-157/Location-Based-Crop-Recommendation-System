import json
import os
from functools import lru_cache
from typing import Any

import requests
from flask import Flask, jsonify, render_template_string, request


APP_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(APP_DIR, "index.html")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
NOMINATIM_URL = os.environ.get("NOMINATIM_URL", "https://nominatim.openstreetmap.org/reverse")
OVERPASS_URL = os.environ.get("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
OSM_HEADERS = {
    "User-Agent": "CropSuitabilityMap/1.0 (local no-api-key building filter)",
}
BUILT_UP_LANDUSE_TYPES = {
    "residential",
    "commercial",
    "industrial",
    "retail",
    "construction",
    "brownfield",
    "garages",
    "railway",
}


def load_index_html() -> str:
    with open(INDEX_PATH, "r", encoding="utf-8") as file:
        return file.read()


app = Flask(__name__)


def point_in_rect(lat: float, lng: float, bounds: tuple[float, float, float, float]) -> bool:
    min_lat, max_lat, min_lng, max_lng = bounds
    return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng


def point_in_polygon(lat: float, lng: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        intersects = ((xi > lng) != (xj > lng)) and (
            lat < (yj - yi) * (lng - xi) / ((xj - xi) or 1e-9) + yi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def built_up_reason_from_tags(tags: dict[str, Any]) -> str | None:
    if "building" in tags:
        building_type = tags.get("building")
        if building_type and building_type != "yes":
            return f"Building detected nearby ({building_type}). Choose open farm land."
        return "Building detected nearby. Choose open farm land."

    landuse = tags.get("landuse")
    if landuse in BUILT_UP_LANDUSE_TYPES:
        return "Built-up urban land detected here. Choose open farm land."

    if "amenity" in tags or "shop" in tags or "office" in tags:
        return "This point looks like an urban facility, not open agricultural land."

    return None


@lru_cache(maxsize=2048)
def detect_built_up_area(lat: float, lng: float) -> tuple[bool, str]:
    rounded_lat = round(lat, 5)
    rounded_lng = round(lng, 5)

    try:
        reverse_response = requests.get(
            NOMINATIM_URL,
            params={
                "format": "jsonv2",
                "lat": rounded_lat,
                "lon": rounded_lng,
                "zoom": 18,
                "addressdetails": 1,
                "extratags": 1,
            },
            headers=OSM_HEADERS,
            timeout=5,
        )
        reverse_response.raise_for_status()
        reverse_data = reverse_response.json()
        reverse_class = str(reverse_data.get("class", "")).strip().lower()
        reverse_type = str(reverse_data.get("type", "")).strip().lower()
        reverse_addresstype = str(reverse_data.get("addresstype", "")).strip().lower()

        if reverse_class == "building" or reverse_addresstype == "building":
            return True, "Building detected at this point. Choose open farm land."
        if reverse_class == "landuse" and reverse_type in BUILT_UP_LANDUSE_TYPES:
            return True, "Built-up urban land detected here. Choose open farm land."
        if reverse_class in {"amenity", "shop", "office"}:
            return True, "This point looks like an urban facility, not open agricultural land."
    except requests.RequestException:
        pass

    overpass_query = f"""
    [out:json][timeout:8];
    (
      nwr(around:16,{rounded_lat},{rounded_lng})[building];
      nwr(around:16,{rounded_lat},{rounded_lng})[amenity];
      nwr(around:16,{rounded_lat},{rounded_lng})[shop];
      nwr(around:16,{rounded_lat},{rounded_lng})[office];
      nwr(around:24,{rounded_lat},{rounded_lng})[landuse~"residential|commercial|industrial|retail|construction|brownfield|garages|railway"];
    );
    out tags center qt 20;
    """.strip()

    try:
        overpass_response = requests.post(
            OVERPASS_URL,
            data=overpass_query,
            headers=OSM_HEADERS,
            timeout=8,
        )
        overpass_response.raise_for_status()
        overpass_data = overpass_response.json()
        for element in overpass_data.get("elements", []):
            reason = built_up_reason_from_tags(element.get("tags", {}))
            if reason:
                return True, reason
    except (requests.RequestException, ValueError):
        pass

    return False, ""


def classify_zone(lat: float, lng: float) -> dict[str, Any]:
    if not (6.0 <= lat <= 37.5 and 68.0 <= lng <= 97.5):
        return {"allowed": False, "reason": "This demo is limited to India-focused crop prediction areas."}

    mainland_polygon = [
        (8.1, 77.2),
        (8.4, 76.0),
        (9.8, 75.6),
        (11.5, 74.7),
        (14.0, 74.2),
        (17.2, 72.9),
        (19.0, 72.7),
        (21.0, 72.8),
        (23.0, 69.1),
        (24.8, 68.2),
        (27.8, 69.4),
        (30.8, 74.2),
        (32.8, 74.8),
        (34.2, 76.1),
        (35.3, 78.5),
        (34.2, 79.8),
        (31.2, 78.9),
        (29.0, 80.0),
        (27.5, 88.0),
        (26.2, 89.5),
        (24.7, 92.8),
        (24.2, 94.5),
        (22.7, 92.7),
        (21.8, 89.5),
        (20.5, 87.8),
        (18.8, 84.8),
        (17.1, 82.8),
        (15.5, 80.5),
        (13.3, 80.2),
        (11.1, 79.8),
        (9.5, 78.8),
        (8.1, 77.2),
    ]

    if not point_in_polygon(lat, lng, mainland_polygon):
        return {"allowed": False, "reason": "Outside the supported mainland farm area."}

    excluded_regions = [
        ((23.0, 30.5, 68.0, 73.8), "Desert-like region"),
        ((30.0, 34.8, 78.0, 80.5), "High-altitude cold desert zone"),
        ((9.45, 10.25, 76.2, 76.55), "Lake / backwater exclusion zone"),
        ((19.55, 20.2, 85.1, 85.6), "Lake exclusion zone"),
        ((26.8, 27.25, 74.95, 75.25), "Lake exclusion zone"),
    ]

    for bounds, reason in excluded_regions:
        if point_in_rect(lat, lng, bounds):
            return {"allowed": False, "reason": reason}

    west_coast = 8.0 <= lat <= 20.5 and 72.5 <= lng <= 76.8
    east_coast = 8.0 <= lat <= 22.5 and 79.2 <= lng <= 88.6
    kerala_coast = 8.0 <= lat <= 12.7 and 74.8 <= lng <= 77.2
    south_east_coast = 8.0 <= lat <= 13.8 and 78.0 <= lng <= 80.8
    andhra_delta = 15.2 <= lat <= 17.6 and 80.2 <= lng <= 82.5
    odisha_delta = 19.0 <= lat <= 21.2 and 84.6 <= lng <= 86.7
    bengal_delta = 21.2 <= lat <= 23.6 and 87.5 <= lng <= 89.8
    west_black_soil = 15.0 <= lat <= 22.5 and 72.0 <= lng <= 77.0
    north_alluvial = 24.0 <= lat <= 31.8 and 74.0 <= lng <= 88.5
    east_wet_belt = 20.0 <= lat <= 27.5 and 80.0 <= lng <= 89.5

    profile = {
        "zone_name": "inland mixed farming belt",
        "rainfall": "medium",
        "temperature": "warm-subtropical",
        "soil": "mixed alluvial",
        "irrigation": "moderate",
        "season_bias": "kharif and rabi",
        "coastal": False,
        "humid_tropical": False,
        "aquaculture_capable": False,
        "coconut_capable": False,
    }

    if kerala_coast or (west_coast and lat < 15.5):
        profile.update(
            {
                "zone_name": "humid tropical coastal belt",
                "rainfall": "high",
                "temperature": "hot-humid",
                "soil": "laterite and red loam",
                "irrigation": "good",
                "season_bias": "multiple seasons",
                "coastal": True,
                "humid_tropical": True,
                "aquaculture_capable": lat <= 10.9,
                "coconut_capable": True,
            }
        )
    elif bengal_delta or odisha_delta or andhra_delta:
        profile.update(
            {
                "zone_name": "deltaic coastal belt",
                "rainfall": "high",
                "temperature": "warm-humid",
                "soil": "deltaic alluvial",
                "irrigation": "good",
                "season_bias": "kharif dominant with rabi support",
                "coastal": True,
                "humid_tropical": andhra_delta,
                "aquaculture_capable": True,
                "coconut_capable": andhra_delta,
            }
        )
    elif south_east_coast:
        profile.update(
            {
                "zone_name": "southern east coastal belt",
                "rainfall": "medium-high",
                "temperature": "hot-humid",
                "soil": "coastal alluvial or red sandy loam",
                "irrigation": "moderate",
                "season_bias": "multiple seasons",
                "coastal": True,
                "humid_tropical": True,
                "aquaculture_capable": lat <= 12.4,
                "coconut_capable": True,
            }
        )
    elif west_black_soil:
        profile.update(
            {
                "zone_name": "black soil plateau",
                "rainfall": "low-medium",
                "temperature": "warm",
                "soil": "black cotton soil",
                "irrigation": "moderate",
            }
        )
    elif north_alluvial:
        profile.update(
            {
                "zone_name": "northern alluvial belt",
                "rainfall": "medium",
                "temperature": "cool-subtropical" if lat >= 28.0 else "subtropical",
                "soil": "alluvial",
                "irrigation": "good",
            }
        )
    elif east_wet_belt:
        profile.update(
            {
                "zone_name": "eastern wet belt",
                "rainfall": "high",
                "temperature": "warm-humid",
                "soil": "alluvial",
                "irrigation": "good",
            }
        )
    elif 12.5 <= lat <= 19.0:
        profile.update(
            {
                "zone_name": "semi-humid plateau",
                "rainfall": "medium",
                "temperature": "warm",
                "soil": "red loam or black soil",
            }
        )

    if east_coast and not profile["coastal"]:
        profile["coastal"] = True

    return {"allowed": True, "profile": profile}


def build_prompt(lat: float, lng: float, profile: dict[str, Any]) -> str:
    return f"""
You are an agricultural suitability engine.
Return JSON only. No markdown. No explanation.

Location:
- latitude: {lat}
- longitude: {lng}

Derived field profile:
{json.dumps(profile, indent=2)}

Task:
Recommend the top 3 suitable options for this exact land point in India.
Use aquaculture only when the profile says aquaculture_capable is true.
Use coconut only when the profile says coconut_capable is true.
Do not suggest desert-only or climate-impossible options.

Required JSON schema:
{{
  "summary": "short sentence",
  "crops": [
    {{
      "name": "crop name",
      "probability": 0,
      "investment_inr_per_acre": "low/medium/high with rupee range",
      "time_to_yield": "short phrase"
    }}
  ]
}}

Rules:
- probability must be integer percentage.
- crops array must have exactly 3 items.
- summary must be under 18 words.
- output only the fields in the schema.
""".strip()


def parse_ollama_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_prediction(data: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    aquaculture_terms = ("fish", "shrimp", "aquaculture", "prawn", "crab")
    crops = []

    for item in data.get("crops", []):
        name = str(item.get("name", "")).strip()
        name_lower = name.lower()
        if not name:
            continue
        if any(term in name_lower for term in aquaculture_terms) and not profile.get("aquaculture_capable"):
            continue
        if "coconut" in name_lower and not profile.get("coconut_capable"):
            continue
        if "oil palm" in name_lower and not profile.get("humid_tropical"):
            continue

        try:
            probability = int(item.get("probability", 0))
        except (TypeError, ValueError):
            continue

        investment = str(item.get("investment_inr_per_acre", "")).strip()
        time_to_yield = str(item.get("time_to_yield", "")).strip()
        if not investment or not time_to_yield:
            continue

        crops.append(
            {
                "name": name,
                "probability": max(0, min(100, probability)),
                "investment_inr_per_acre": investment,
                "time_to_yield": time_to_yield,
            }
        )

        if len(crops) == 3:
            break

    return {
        "summary": str(data.get("summary", "Top crops estimated for this land point.")).strip() or "Top crops estimated for this land point.",
        "crops": crops,
    }


def fallback_prediction(profile: dict[str, Any]) -> dict[str, Any]:
    if profile["zone_name"] == "humid tropical coastal belt" and profile["coconut_capable"]:
        crops = [
            {"name": "Coconut", "probability": 95, "investment_inr_per_acre": "INR 45,000 - INR 65,000 (initial)", "time_to_yield": "5 - 7 Years"},
            {"name": "Aquaculture (Shrimp/Fish)", "probability": 90, "investment_inr_per_acre": "INR 250,000 - INR 500,000", "time_to_yield": "4 - 6 Months"},
            {"name": "Banana", "probability": 84, "investment_inr_per_acre": "INR 55,000 - INR 80,000", "time_to_yield": "10 - 12 Months"},
        ]
    elif profile["zone_name"] == "deltaic coastal belt" and profile["coconut_capable"]:
        crops = [
            {"name": "Paddy (Rice)", "probability": 96, "investment_inr_per_acre": "INR 22,000 - INR 34,000", "time_to_yield": "4 - 5 Months"},
            {"name": "Aquaculture (Shrimp/Fish)", "probability": 91, "investment_inr_per_acre": "INR 250,000 - INR 500,000", "time_to_yield": "4 - 6 Months"},
            {"name": "Coconut", "probability": 83, "investment_inr_per_acre": "INR 45,000 - INR 65,000 (initial)", "time_to_yield": "5 - 7 Years"},
        ]
    elif profile["aquaculture_capable"]:
        crops = [
            {"name": "Paddy (Rice)", "probability": 96, "investment_inr_per_acre": "INR 22,000 - INR 34,000", "time_to_yield": "4 - 5 Months"},
            {"name": "Aquaculture (Shrimp/Fish)", "probability": 91, "investment_inr_per_acre": "INR 250,000 - INR 500,000", "time_to_yield": "4 - 6 Months"},
            {"name": "Jute", "probability": 82, "investment_inr_per_acre": "INR 18,000 - INR 26,000", "time_to_yield": "4 Months"},
        ]
    elif profile["coconut_capable"]:
        crops = [
            {"name": "Coconut", "probability": 94, "investment_inr_per_acre": "INR 45,000 - INR 65,000 (initial)", "time_to_yield": "5 - 7 Years"},
            {"name": "Banana", "probability": 86, "investment_inr_per_acre": "INR 55,000 - INR 80,000", "time_to_yield": "10 - 12 Months"},
            {"name": "Arecanut", "probability": 79, "investment_inr_per_acre": "INR 60,000 - INR 90,000 (initial)", "time_to_yield": "4 - 6 Years"},
        ]
    elif profile["zone_name"] == "black soil plateau":
        crops = [
            {"name": "Cotton", "probability": 86, "investment_inr_per_acre": "INR 24,000 - INR 38,000", "time_to_yield": "5 - 6 Months"},
            {"name": "Soybean", "probability": 79, "investment_inr_per_acre": "INR 11,000 - INR 18,000", "time_to_yield": "3 - 4 Months"},
            {"name": "Tur Dal", "probability": 72, "investment_inr_per_acre": "INR 9,000 - INR 15,000", "time_to_yield": "5 - 6 Months"},
        ]
    elif profile["zone_name"] == "northern alluvial belt":
        crops = [
            {"name": "Wheat", "probability": 84, "investment_inr_per_acre": "INR 12,000 - INR 20,000", "time_to_yield": "4 - 5 Months"},
            {"name": "Mustard", "probability": 78, "investment_inr_per_acre": "INR 8,000 - INR 13,000", "time_to_yield": "3 - 4 Months"},
            {"name": "Potato", "probability": 72, "investment_inr_per_acre": "INR 30,000 - INR 55,000", "time_to_yield": "3 - 4 Months"},
        ]
    elif profile["zone_name"] == "eastern wet belt":
        crops = [
            {"name": "Paddy (Rice)", "probability": 88, "investment_inr_per_acre": "INR 18,000 - INR 28,000", "time_to_yield": "4 - 5 Months"},
            {"name": "Maize", "probability": 74, "investment_inr_per_acre": "INR 14,000 - INR 22,000", "time_to_yield": "3 - 4 Months"},
            {"name": "Banana", "probability": 69, "investment_inr_per_acre": "INR 50,000 - INR 78,000", "time_to_yield": "10 - 12 Months"},
        ]
    else:
        crops = [
            {"name": "Maize", "probability": 78, "investment_inr_per_acre": "INR 14,000 - INR 22,000", "time_to_yield": "3 - 4 Months"},
            {"name": "Groundnut", "probability": 73, "investment_inr_per_acre": "INR 16,000 - INR 24,000", "time_to_yield": "4 - 5 Months"},
            {"name": "Pigeon Pea", "probability": 70, "investment_inr_per_acre": "INR 9,000 - INR 15,000", "time_to_yield": "5 - 6 Months"},
        ]

    return {"summary": "Top 3 options for this land point.", "crops": crops}


def generate_prediction(lat: float, lng: float, profile: dict[str, Any]) -> tuple[dict[str, Any], str]:
    prompt = build_prompt(lat, lng, profile)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        parsed = normalize_prediction(parse_ollama_json(data.get("response", "")), profile)
        if len(parsed["crops"]) != 3:
            raise ValueError("Invalid crop response length")
        return parsed, "ollama"
    except Exception:
        return fallback_prediction(profile), "fallback"


@app.get("/")
def index() -> str:
    return render_template_string(load_index_html(), model_name=OLLAMA_MODEL)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    lat = payload.get("lat")
    lng = payload.get("lng")

    if lat is None or lng is None:
        return jsonify({"ok": False, "error": "Latitude and longitude are required."}), 400

    try:
        lat = float(lat)
        lng = float(lng)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid coordinates."}), 400

    zone = classify_zone(lat, lng)
    if not zone["allowed"]:
        return jsonify(
            {
                "ok": False,
                "error": "Prediction blocked for this location.",
                "reason": zone["reason"],
            }
        ), 422

    is_built_up, built_up_reason = detect_built_up_area(lat, lng)
    if is_built_up:
        return jsonify(
            {
                "ok": False,
                "error": "Prediction blocked for this location.",
                "reason": built_up_reason,
            }
        ), 422

    result, source = generate_prediction(lat, lng, zone["profile"])
    return jsonify(
        {
            "ok": True,
            "source": source,
            "location": {"lat": round(lat, 5), "lng": round(lng, 5)},
            "profile": zone["profile"],
            "result": result,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
