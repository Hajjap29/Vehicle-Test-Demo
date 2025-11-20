import streamlit as st
from dotenv import load_dotenv
import os
import base64
from openai import OpenAI

load_dotenv()
client = OpenAI()

def image_file_to_data_uri(path):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

PROMPT_JSON = """You are an image-understanding system. Extract vehicle info in JSON."""

def call_vision_api_with_image(data_uri, prompt_json=PROMPT_JSON):
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_json},
                        {"type": "input_image", "image_url": data_uri},
                    ]
                }
            ]
        )
        return response
    except Exception as e:
        return {"error": str(e)}

def extract_json_from_response(resp):
    try:
        text = resp.output_text
        import json
        return json.loads(text)
    except:
        return {"raw_text": str(resp)}

def analyze_image_file(path):
    data_uri = image_file_to_data_uri(path)
    resp = call_vision_api_with_image(data_uri)
    return extract_json_from_response(resp)

st.title("Vehicle Vision Analyzer")

uploaded = st.file_uploader("Upload vehicle image", type=["jpg","jpeg","png"])

if uploaded:
    st.image(uploaded)
    if st.button("Analyze"):
        with open("temp.jpg","wb") as f:
            f.write(uploaded.getvalue())
        st.json(analyze_image_file("temp.jpg"))
