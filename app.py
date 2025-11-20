import streamlit as st
import base64
import json
import requests
import os

# --- Configuration ---
st.set_page_config(
    page_title="Car Carbon AI", 
    page_icon="🚗",
    layout="wide"
)

# Constants
MODEL = "gpt-4o-mini"
API_URL = "https://api.openai.com/v1/chat/completions"

PROMPT_JSON = """
You are an image-understanding assistant. I will provide an image of a vehicle. 
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
    mime = uploaded_file.type
    return f"data:{mime};base64,{b64}"

def extract_json_from_response(resp_json):
    """Robustly extracts JSON from OpenAI response."""
    candidates = []
    
    if "choices" in resp_json:
        try:
            choice_msg = resp_json["choices"][0].get("message", {}).get("content")
            if choice_msg:
                candidates.append(choice_msg)
        except Exception:
            pass

    if not candidates:
        candidates.append(json.dumps(resp_json))

    full_text = "\n".join(candidates)
    
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
        st.error(f"API Connection Error: {e}")
        return None

def estimate_carbon(detection):
    cls = detection.get("vehicle_class", "unknown")
    powertrain = detection.get("powertrain", "unknown")
    cfg = CARBON_TABLE.get(cls, CARBON_TABLE["unknown"])
    
    # EV Adjustment (Manufacturing higher, use phase lower)
    if powertrain == "electric" and cls != "unknown" and cfg["lifetime_tons_min"] is not None:
        min_est = max(0, int(cfg["lifetime_tons_min"] * 0.7))
        max_est = int(cfg["lifetime_tons_max"] * 0.8)
        return {"lifetime_min_tons": min_est, "lifetime_max_tons": max_est, "note": "EV estimate (adjusted for grid)"}
        
    return {
        "lifetime_min_tons": cfg["lifetime_tons_min"], 
        "lifetime_max_tons": cfg["lifetime_tons_max"], 
        "note": "Standard internal combustion estimate"
    }

def refined_estimate(detection, miles_per_year, years):
    cls = (detection or {}).get("vehicle_class", "unknown")
    cfg = CARBON_TABLE.get(cls, {})
    
    if not cfg or cfg.get("lifetime_tons_min") is None:
        return None

    mid_lifetime = (cfg["lifetime_tons_min"] + cfg["lifetime_tons_max"]) / 2.0
    # Rough heuristic: 30% emissions from manufacturing, 70% from driving
    manufacturing = mid_lifetime * 0.3
    use_phase_total_lifetime = mid_lifetime * 0.7
    
    # Calculate per-mile based on standard assumption (e.g. 150k lifetime miles)
    assumed_lifetime_miles = 150000
    per_mile_tons = use_phase_total_lifetime / assumed_lifetime_miles
    
    annual_emissions = per_mile_tons * miles_per_year
    total_estimate = manufacturing + (annual_emissions * years)
    
    return {
        "manufacturing_tons": manufacturing,
        "annual_tons": annual_emissions,
        "total_lifetime_tons": total_estimate,
        "per_mile_g": per_mile_tons * 1e6
    }

# --- Main UI Layout ---

st.title("🚗 Car Model & Carbon Recognition")
st.markdown("""
**Upload a photo of a vehicle.** AI will identify the Make/Model and estimate the carbon footprint over its lifetime.
""")

# --- 1. SECURE API KEY HANDLING ---
# This looks for the key in Streamlit Secrets. 
# It will NOT appear in the UI.
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    # Fallback for local testing if you have a .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    except ImportError:
        pass

if not api_key:
    st.error("🚨 **Configuration Error:** OpenAI API Key not found. Please set `OPENAI_API_KEY` in Streamlit Secrets.")
    st.stop()

# --- 2. APP LOGIC ---

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption="Your Upload", use_column_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        analyze_btn = st.button("🔍 Identify Vehicle", type="primary")
        
        # Use session state to keep results visible when interacting with sliders later
        if "detection_result" not in st.session_state:
            st.session_state.detection_result = None
            
        if analyze_btn:
            with st.spinner("Analyzing image features..."):
                data_uri = encode_image(uploaded_file)
                raw_resp = call_vision_api(data_uri, api_key)
                
                if raw_resp:
                    parsed = extract_json_from_response(raw_resp)
                    if "error" not in parsed:
                        st.session_state.detection_result = parsed
                    else:
                        st.error("Could not identify vehicle. Please try a clearer image.")
                        st.json(parsed)

        # Display Results if they exist
        result = st.session_state.detection_result
        if result:
            # Identity Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Make", result.get("make", "Unknown"))
            m2.metric("Model", result.get("model", "Unknown"))
            m3.metric("Year", result.get("year_range", "Unknown"))
            
            st.caption(f"Detected Class: {result.get('vehicle_class')} | Powertrain: {result.get('powertrain')}")
            
            st.divider()
            
            # Carbon Calculation
            base_est = estimate_carbon(result)
            if base_est["lifetime_min_tons"]:
                st.subheader("🌍 Carbon Footprint")
                st.write(f"Estimated Lifetime Emissions: **{base_est['lifetime_min_tons']} - {base_est['lifetime_max_tons']} tons CO₂e**")
                
                st.markdown("### 🧮 Personalize Estimate")
                c1, c2 = st.columns(2)
                miles = c1.number_input("Miles Driven / Year", value=12000, step=1000)
                years = c2.number_input("Years of Ownership", value=10, step=1)
                
                refined = refined_estimate(result, miles, years)
                
                if refined:
                    total = refined['total_lifetime_tons']
                    st.success(f"**Your Projected Total: {total:.1f} tons CO₂e**")
                    
                    # Simple Chart
                    chart_data = {
                        "Category": ["Manufacturing (Fixed)", "Driving (Variable)"],
                        "Emissions (Tons)": [refined["manufacturing_tons"], refined["annual_tons"] * years]
                    }
                    st.bar_chart(chart_data, x="Category", y="Emissions (Tons)")
                    
                    st.info(f"This is equivalent to **{refined['per_mile_g']:.0f} grams** of CO₂ emitted per mile.")
            else:
                st.warning("Carbon data unavailable for this specific vehicle type.")
