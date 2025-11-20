import streamlit as st
import base64
import json
import requests
import os
# Optional: load .env if running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Configuration ---
st.set_page_config(page_title="Car Carbon AI", layout="wide")

# Constants
MODEL = "gpt-4o-mini"
API_URL = "https://api.openai.com/v1/chat/completions"

PROMPT_JSON = """
You are an image-understanding assistant. I will provide an image. 
Respond ONLY with JSON (no extra text). The JSON must have the following keys:
- make: string or null
- model: string or null
- year_range: string or null (e.g., "2016-2020")
- vehicle_class: one of ["compact", "midsize", "fullsize", "suv", "pickup", "van", "motorcycle", "bus", "truck", "unknown"]
- powertrain: one of ["gasoline", "diesel", "hybrid", "plug-in hybrid", "electric", "unknown"]
- confidence: number between 0 and 1 (estimate of how confident you are)

Make conservative guesses. If uncertain, put null or "unknown". Don't output any explanatory text — ONLY the JSON object.
"""

CARBON_TABLE = {
    "compact":    {"lifetime_tons_min": 30, "lifetime_tons_max": 50},
    "midsize":    {"lifetime_tons_min": 40, "lifetime_tons_max": 65},
    "fullsize":   {"lifetime_tons_min": 55, "lifetime_tons_max": 85},
    "suv":        {"lifetime_tons_min": 50, "lifetime_tons_max": 80},
    "pickup":     {"lifetime_tons_min": 60, "lifetime_tons_max": 100},
    "van":        {"lifetime_tons_min": 45, "lifetime_tons_max": 80},
    "motorcycle": {"lifetime_tons_min": 10, "lifetime_tons_max": 25},
    "bus":        {"lifetime_tons_min": 150, "lifetime_tons_max": 400},
    "truck":      {"lifetime_tons_min": 120, "lifetime_tons_max": 350},
    "unknown":    {"lifetime_tons_min": None, "lifetime_tons_max": None}
}

# --- Helper Functions ---

def encode_image(uploaded_file):
    """Encodes the uploaded Streamlit file object to Base64."""
    bytes_data = uploaded_file.getvalue()
    b64 = base64.b64encode(bytes_data).decode('utf-8')
    # Determine mime type
    mime = uploaded_file.type
    return f"data:{mime};base64,{b64}"

def extract_json_from_response(resp_json):
    """Robustly extracts JSON from OpenAI response."""
    candidates = []
    
    # Check standard ChatCompletion structure
    if "choices" in resp_json:
        try:
            choice_msg = resp_json["choices"][0].get("message", {}).get("content")
            if choice_msg:
                candidates.append(choice_msg)
        except Exception:
            pass

    # Fallback: raw string
    if not candidates:
        candidates.append(json.dumps(resp_json))

    full_text = "\n".join(candidates)
    
    # Find JSON substring
    decoder = json.JSONDecoder()
    for i in range(len(full_text)):
        try:
            obj, idx = decoder.raw_decode(full_text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
            
    return {"raw_text": full_text, "error": "no JSON object found"}

def call_vision_api(data_uri, api_key):
    """Sends request to OpenAI Chat Completions API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Standard OpenAI Chat Format
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_JSON},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ],
        "max_tokens": 300
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return None

def estimate_carbon(detection):
    cls = detection.get("vehicle_class", "unknown")
    powertrain = detection.get("powertrain", "unknown")
    cfg = CARBON_TABLE.get(cls, CARBON_TABLE["unknown"])
    
    # EV Adjustment
    if powertrain == "electric" and cls != "unknown" and cfg["lifetime_tons_min"] is not None:
        min_est = max(0, int(cfg["lifetime_tons_min"] * 0.7))
        max_est = int(cfg["lifetime_tons_max"] * 0.8)
        return {"lifetime_min_tons": min_est, "lifetime_max_tons": max_est, "note": "EV estimate adjusted"}
        
    return {
        "lifetime_min_tons": cfg["lifetime_tons_min"], 
        "lifetime_max_tons": cfg["lifetime_tons_max"], 
        "note": "Standard category estimate"
    }

def refined_estimate(detection, miles_per_year, years):
    cls = (detection or {}).get("vehicle_class", "unknown")
    cfg = CARBON_TABLE.get(cls, {})
    
    if not cfg or cfg.get("lifetime_tons_min") is None:
        return None

    mid_lifetime = (cfg["lifetime_tons_min"] + cfg["lifetime_tons_max"]) / 2.0
    manufacturing = mid_lifetime * 0.3
    use_phase_total = mid_lifetime * 0.7
    
    lifetime_miles = miles_per_year * years
    per_mile_tons = (use_phase_total / lifetime_miles) if lifetime_miles > 0 else 0
    annual_emissions = per_mile_tons * miles_per_year
    
    total_estimate = manufacturing + (annual_emissions * years)
    
    return {
        "manufacturing_tons": manufacturing,
        "annual_tons": annual_emissions,
        "total_lifetime_tons": total_estimate,
        "per_mile_g": per_mile_tons * 1e6
    }

# --- Main UI ---

st.title("🚗 Car Model & Carbon Recognition")
st.markdown("Upload a photo of a vehicle to identify it and estimate its carbon footprint.")

# 1. API Key Management
with st.sidebar:
    st.header("Settings")
    # Check secrets first, then environment, then user input
    env_key = os.getenv("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("API Key loaded from Secrets")
    elif env_key:
        api_key = env_key
        st.success("API Key loaded from Environment")
    else:
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        if not api_key:
            st.warning("Please enter an API Key to proceed.")

# 2. Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file and api_key:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("Analysis")
        analyze_btn = st.button("Identify Vehicle", type="primary")
        
        # Session state to hold results so they don't disappear on re-run
        if "detection_result" not in st.session_state:
            st.session_state.detection_result = None
            
        if analyze_btn:
            with st.spinner("Analyzing image with GPT-4o..."):
                data_uri = encode_image(uploaded_file)
                raw_resp = call_vision_api(data_uri, api_key)
                
                if raw_resp:
                    parsed = extract_json_from_response(raw_resp)
                    if "error" not in parsed:
                        st.session_state.detection_result = parsed
                    else:
                        st.error("Could not parse vehicle data from AI response.")
                        st.json(parsed)

        # Display Results
        result = st.session_state.detection_result
        if result:
            # Visual Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Make", result.get("make", "Unknown"))
            m2.metric("Model", result.get("model", "Unknown"))
            m3.metric("Year", result.get("year_range", "Unknown"))
            
            st.info(f"**Class:** {result.get('vehicle_class')} | **Powertrain:** {result.get('powertrain')}")
            
            # Base Carbon Estimate
            base_est = estimate_carbon(result)
            if base_est["lifetime_min_tons"]:
                st.divider()
                st.subheader("🌍 Carbon Footprint Estimate")
                st.write(f"Estimated Lifetime Emissions: **{base_est['lifetime_min_tons']} - {base_est['lifetime_max_tons']} tons CO₂e**")
                st.caption(f"Note: {base_est['note']}")

                # Calculator
                st.markdown("### 🧮 Refine Your Estimate")
                c1, c2 = st.columns(2)
                miles = c1.number_input("Miles Driven per Year", value=12000, step=1000)
                years = c2.number_input("Years of Ownership", value=12, step=1)
                
                refined = refined_estimate(result, miles, years)
                
                if refined:
                    st.success(f"**Total Projected Lifetime Emissions: {refined['total_lifetime_tons']:.1f} tons**")
                    
                    # Breakdown Chart
                    chart_data = {
                        "Source": ["Manufacturing", "Use Phase (Fuel/Grid)"],
                        "Emissions (Tons)": [refined["manufacturing_tons"], refined["annual_tons"] * years]
                    }
                    st.bar_chart(chart_data, x="Source", y="Emissions (Tons)")
                    
                    st.write(f"**Emissions per mile:** {refined['per_mile_g']:.1f} grams CO₂")
            else:
                st.warning("Could not estimate carbon data for this vehicle class.")

        # Debug expander
        with st.expander("View Raw API Response"):
            st.json(result)

elif uploaded_file and not api_key:
    st.error("Please provide an OpenAI API Key in the sidebar.")
