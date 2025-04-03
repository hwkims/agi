import streamlit as st
import requests
import json
from PIL import Image
import io
import base64

# Ollama API 엔드포인트 (기본값: http://localhost:11434/api/generate)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def encode_image_to_base64(image):
    """PIL Image 객체를 base64 문자열로 인코딩합니다."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")  # 또는 image.format
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def call_ollama_api(image, prompt, model="gemma3:4b"):
    """Ollama API를 호출하여 이미지와 프롬프트에 대한 응답을 받습니다."""
    try:
        encoded_image = encode_image_to_base64(image)

        data = {
            "model": model,
            "prompt": prompt,
            "images": [encoded_image],
            "stream": False,  # 스트리밍 비활성화
            "format": "json" #json format으로 받음
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

        json_data = response.json()
        # 전체 응답 확인 (디버깅용)
        #st.write(json_data)
        return json_data["response"]


    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        st.error("Error: Ollama API returned an unexpected response format.")
        st.write(response.text)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {type(e).__name__} - {e}")
        return None
st.title("Gemma 3 이미지 인식 (Ollama API)")

uploaded_file = st.file_uploader("이미지를 업로드하세요.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    prompt = st.text_input("이미지에 대해 질문을 입력하세요:", "이미지에 무엇이 있나요?")
    model_choice = st.selectbox("모델 선택", ["gemma3:4b", "gemma3:12b", "gemma3:27b"])

    if st.button("실행"):
        with st.spinner("Gemma 3 모델 실행 중..."):
            result = call_ollama_api(image, prompt, model=model_choice)
            if result:
                st.subheader("결과:")
                st.write(result)
