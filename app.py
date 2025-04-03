import streamlit as st
import requests
import base64
import json
import re
import io
import asyncio
from edge_tts import Communicate
import logging
from PIL import Image
import streamlit.components.v1 as components

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# Ollama 서버 주소
OLLAMA_HOST = 'http://localhost:11434'
# 음성 설정
VOICE = "ko-KR-HyunsuNeural"

# 프롬프트 템플릿 (한국어, 이미지 설명 중심)
PROMPT_TEMPLATE = """
[INST]
당신은 이미지를 매우 잘 설명하는 어시스턴트입니다.  사용자의 입력을 바탕으로 다음 중 하나를 수행합니다:

1. 이미지가 주어지면, 이미지를 자세하고 정확하게 설명합니다.
2. 텍스트 입력이 주어지면, 짧고 간결하게 응답합니다.

{user_input}
[/INST]
"""

async def tts(text, voice=VOICE):
    """Edge TTS를 사용하여 텍스트를 음성으로 변환"""
    try:
        communicate = Communicate(text, voice)
        audio_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        audio_data.seek(0)
        return audio_data

    except Exception as e:
        logging.error(f"TTS Error: {e}")
        st.error(f"TTS 오류: {e}")
        return None


def query_ollama(prompt, context=None, image_data=None):
    """Ollama API 호출 (대화 및 이미지)"""
    data = {
        "model": "gemma3:4b",  # Multimodal model
        "prompt": prompt,
        "stream": False,
        "context": context or [],
        "options": {"temperature": 0.2, "top_p": 0.8},  # Adjusted for more creative descriptions
    }
    if image_data:
        data["images"] = [image_data]

    try:
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=data, stream=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama Request Error: {e}")
        return {"error": f"Ollama API 요청 오류: {e}"}

# Ollama 응답에서 JSON 파싱 부분 제거 - 이제 JSON 명령어 처리 안함
# def parse_ollama_response(response_text): ...


async def main():
    st.title("AI 대화 챗봇 (이미지 + 음성 인식)")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = []
    if "speech_active" not in st.session_state:
        st.session_state.speech_active = False

    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "audio" in message:
                st.audio(message["audio"])

    # 이미지 업로드
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_column_width=True)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # 이미지 설명
        with st.spinner("이미지 설명 생성 중..."):
            ollama_response = query_ollama(PROMPT_TEMPLATE.format(user_input="이 이미지를 설명해 주세요."), image_data=img_base64) # user_input 변경
            if "error" in ollama_response:
                st.error(f"이미지 설명 오류: {ollama_response['error']}")
            else:
                image_description = ollama_response["response"].strip()
                st.session_state.messages.append({"role": "assistant", "content": image_description})
                st.session_state.context = ollama_response.get('context', [])  # 컨텍스트는 계속 업데이트
                with st.chat_message("assistant"):
                    st.markdown(image_description)
                audio_data = await tts(image_description)
                if audio_data:
                    st.audio(audio_data)
                    st.session_state.messages[-1]["audio"] = audio_data

    # 음성 인식 활성화/비활성화 버튼
    if st.button("음성 인식 " + ("켜기" if not st.session_state.speech_active else "끄기")):
        st.session_state.speech_active = not st.session_state.speech_active
        if st.session_state.speech_active:
            st.write("음성 인식이 활성화되었습니다. 마이크에 말씀하세요.")
        else:
            st.write("음성 인식이 비활성화되었습니다.")

    # 사용자 입력 (텍스트)
    user_input = st.chat_input("무엇이든 물어보세요")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Ollama에 쿼리 (이미지/텍스트)
        with st.spinner("답변 생성 중..."):
            ollama_response = query_ollama(
                PROMPT_TEMPLATE.format(user_input=user_input),
                st.session_state.context,  # 컨텍스트는 항상 전달
                image_data=img_base64 if uploaded_file else None,
            )
            if "error" in ollama_response:
                st.error(ollama_response["error"])
                assistant_response_content = "죄송해요, 무슨 말씀인지 잘 모르겠어요."
            else:
                # JSON 파싱 제거, 바로 response 사용
                assistant_response_content = ollama_response["response"].strip()
                st.session_state.context = ollama_response.get('context', []) # 컨텍스트 업데이트

        with st.chat_message("assistant"):
            st.markdown(assistant_response_content)
            with st.spinner("음성 생성 중..."):
                audio_data = await tts(assistant_response_content)
                if audio_data:
                    st.audio(audio_data)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response_content,
                        "audio": audio_data
                    })
                else: # tts 실패시에도 텍스트는 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response_content
                    })


    # 음성 인식 (Web Speech API, JavaScript -> Streamlit)
    components.html(
        f"""
<script>
const inputField = window.parent.document.querySelector('textarea[data-testid="stChatInput"]');
let recognition;

function startRecognition() {{
  recognition = new window.webkitSpeechRecognition(); // Chrome
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'ko-KR';

  recognition.onresult = (event) => {{
    const result = event.results[0][0].transcript;
    inputField.value = result;
    inputField.dispatchEvent(new Event('input', {{ bubbles: true }}));
    const sendButton = window.parent.document.querySelector('button[data-testid="send-message-button"]');
    if (sendButton) {{
      sendButton.click();
    }} else {{
      console.error('Send button not found!');
    }}
  }};

  recognition.onerror = (event) => {{
    console.error('Speech recognition error:', event.error);
  }};

  recognition.onend = () => {{
      console.log("ended")
  }}

  recognition.start();
  console.log('Speech recognition started');
}}

if ({'true' if st.session_state.speech_active else 'false'}) {{
  if (!recognition) {{
    startRecognition();
  }}
}} else if(recognition) {{
    recognition.stop()
}}

</script>
""",
        height=0,
    )

if __name__ == "__main__":
    asyncio.run(main())
