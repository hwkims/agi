import streamlit as st
import requests
import json
import base64
import io
from PIL import Image
import os
import tempfile

# Ollama API 엔드포인트 및 모델 설정 (설정 파일 또는 환경 변수에서 읽어오는 것이 좋음)
OLLAMA_HOST = "192.168.0.5:11434"  # 실제 Jetson IP 주소로 변경
OLLAMA_MODEL = "gemma3:4b"  # 사용하는 모델 이름


# Ollama API 호출 함수
def ollama_api(prompt, image_path=None):
    data = {"prompt": prompt, "model": OLLAMA_MODEL, "stream": False, "format": "json"}

    if image_path:
        # 이미지 파일을 base64로 인코딩
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            data["images"] = [encoded_string]
        except FileNotFoundError:
            st.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return None
        except Exception as e:
            st.error(f"이미지 처리 중 오류 발생: {e}")
            return None

    try:
        response = requests.post(f"http://{OLLAMA_HOST}/api/generate", json=data)
        response.raise_for_status()  # 200 OK가 아니면 예외 발생
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama API 호출 오류: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Ollama API 응답이 JSON 형식이 아닙니다.")
        return None


# Streamlit 앱
st.title("Ollama 챗봇 (with Image Support)")

# 사용자 입력 (텍스트 또는 이미지)
input_type = st.radio("입력 유형 선택:", ("텍스트", "이미지"))

if input_type == "텍스트":
    user_input = st.text_input("질문을 입력하세요:")
    if st.button("질문하기"):
        if user_input:
            with st.spinner("답변 생성 중..."):
                response = ollama_api(user_input)
            if response:
                st.write("답변:")
                st.write(response)
        else:
            st.warning("질문을 입력해주세요.")

elif input_type == "이미지":
    uploaded_file = st.file_uploader("이미지를 업로드하세요:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_column_width=True)

        # 임시 파일로 저장 (경로 문제 해결)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            temp_image_path = tmp_file.name

        user_question = st.text_input("이미지에 대해 질문을 입력하세요 (선택 사항):")
        if st.button("질문하기"):
            with st.spinner("답변 생성 중..."):
                response = ollama_api(user_question, temp_image_path)
            if response:
                st.write("답변:")
                st.write(response)

            # 임시 파일 삭제
            os.remove(temp_image_path)
