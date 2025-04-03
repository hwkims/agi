import streamlit as st
import requests
import base64
import json
import re
import io
import asyncio
from edge_tts import Communicate
import logging
import time
from PIL import Image  # PIL(Pillow) 사용


# 로깅 설정
logging.basicConfig(level=logging.INFO)

# Ollama 서버 주소
OLLAMA_HOST = 'http://localhost:11434'
# 음성 설정
VOICE = "ko-KR-HyunsuNeural"

# 프롬프트 템플릿 (명령어 인식 추가)
PROMPT_TEMPLATE = """
[INST]
You are a helpful assistant that can also understand images.  You can both speak and understand spoken language.  You also understand simple JSON commands.  Here is the user's input, which might include text, a spoken command, or a description of an image:

{user_input}

If the input is a clear command related to facial analysis or forward movement (like "analyze the face," "move forward," "go ahead," etc.), output a JSON object with a "command" key.  Example:

Input: "앞으로 가"
Output: {{"command": "forward"}}

Input: "얼굴 분석해 줘"
Output: {{"command": "analyze_face"}}

Otherwise, just respond conversationally to the input as text. Do *not* include any JSON if the input is not a specific, recognizable command.  Keep your conversational replies *very* short (one sentence, or a few words).
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
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False,
        "context": context or [],
        "options": {"temperature": 0.2, "top_p": 0.8},
    }
    if image_data:
        data["images"] = [image_data]  # 이미지 데이터 추가

    try:
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=data, stream=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama Request Error: {e}")
        return {"error": f"Ollama API 요청 오류: {e}"}

def parse_ollama_response(response_text):
    """Ollama 응답에서 JSON 파싱 (명령어 처리)"""
    try:
        # JSON 명령어 파싱 시도
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)  # JSON 객체 반환
        else:
            return {"response": response_text.strip()}  # 일반 텍스트 응답
    except json.JSONDecodeError:
        return {"response": response_text.strip()}  # JSON 파싱 실패 시


async def main():
    st.title("AI 대화 챗봇 (이미지 + 음성 인식)")


    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = []


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
        # 이미지 base64 인코딩
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")  # 또는 "PNG"
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # 즉시 이미지 분석 (필요하다면)
        with st.spinner("이미지 분석 중..."):
            ollama_response = query_ollama(PROMPT_TEMPLATE.format(user_input="Analyze this face."), image_data=img_base64) # image_data 추가
            if "error" in ollama_response:
                st.error(f"이미지 분석 오류: {ollama_response['error']}")
            else:
                parsed_response = parse_ollama_response(ollama_response['response'])
                if "command" in parsed_response and parsed_response["command"] == "analyze_face":
                    st.session_state.messages.append({"role": "assistant", "content": "얼굴 분석 완료."})
                    st.session_state.context = ollama_response.get('context', [])  # 문맥 업데이트
                    audio_data = await tts("얼굴 분석 완료.")
                    if audio_data:
                        st.audio(audio_data)
                        st.session_state.messages[-1]["audio"] = audio_data #메시지에 추가.


    # 사용자 입력
    user_input = st.chat_input("무엇이든 물어보세요")
    if user_input:
        # 사용자 메시지 처리 및 표시
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Ollama에 쿼리 (텍스트)
        with st.spinner("답변 생성 중..."):
            ollama_response = query_ollama(PROMPT_TEMPLATE.format(user_input=user_input), st.session_state.context, image_data=img_base64 if uploaded_file else None)
            if "error" in ollama_response:
                st.error(ollama_response["error"])
                assistant_response_content = "죄송해요, 무슨 말씀인지 잘 모르겠어요."
                parsed_response = {"response": assistant_response_content}  # 에러 시에도 파싱된 응답처럼
            else:
                parsed_response = parse_ollama_response(ollama_response["response"])
                st.session_state.context = ollama_response.get('context', [])  # 대화 컨텍스트 업데이트


        # 챗봇 응답 처리
        with st.chat_message("assistant"):
            if "command" in parsed_response:
                # 명령어 처리 (여기서는 간단한 예시)
                if parsed_response["command"] == "forward":
                    st.markdown("앞으로 이동합니다.")  # 실제 로봇 제어 코드는 여기에
                    assistant_response_content = "앞으로 이동합니다."
                elif parsed_response["command"] == "analyze_face":
                    st.markdown("얼굴을 분석합니다.") # 위에서 이미 처리했어야함.
                    assistant_response_content = "얼굴을 분석합니다."
                else:
                    st.markdown(f"알 수 없는 명령어: {parsed_response['command']}")
                    assistant_response_content = f"알 수 없는 명령어: {parsed_response['command']}"
            else:
                # 일반 텍스트 응답
                assistant_response_content = parsed_response["response"]
                st.markdown(assistant_response_content)

            # TTS 실행 (비동기)
            with st.spinner("음성 생성 중..."):
                audio_data = await tts(assistant_response_content)
                if audio_data:
                    st.audio(audio_data)
                    # 오디오 데이터와 함께 메시지 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response_content,
                        "audio": audio_data
                    })
                else:
                    # TTS 실패 시에도 텍스트 메시지는 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response_content,
                    })


     # 음성 인식 결과 표시 (JavaScript -> Streamlit)
    import streamlit.components.v1 as components
    components.html(
        """
<script>
const inputField = window.parent.document.querySelector('textarea[data-testid="stChatInput"]');
const constraints = { audio: true };

function handleSuccess(stream) {
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();

     // 오디오 데이터 처리 (1초마다)
    const audioChunks = [];
    mediaRecorder.addEventListener("dataavailable", event => {
      audioChunks.push(event.data);
      if(audioChunks.length > 10){
        audioChunks.shift() // 10개 이상이면, 오래된것 삭제.
      }
    });

    mediaRecorder.addEventListener("stop", () => {
      const audioBlob = new Blob(audioChunks);
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);

      reader.onloadend = () => {
        const base64Audio = reader.result;

         // base64 오디오 데이터를 Streamlit (Python)으로 전송
        const event = new CustomEvent('audioData', {detail: base64Audio});
        window.parent.document.dispatchEvent(event);

      }
    });

     // 2초 후 녹음 중지 (데모용)
      setTimeout(() => {
        mediaRecorder.stop();

      }, 20000); // 20초 (데모용)
}


navigator.mediaDevices.getUserMedia(constraints)
    .then(handleSuccess);


// Streamlit -> JS 이벤트 (세션 업데이트 등)
window.parent.document.addEventListener('DOMContentLoaded', function(event) {
    console.log('Streamlit app loaded');  // 디버깅
    // 여기에 Streamlit -> JS 이벤트 핸들러 (필요한 경우)
});



// CustomEvent listener (JS -> Python)
document.addEventListener('audioData', (e) => {
   //  console.log("Received audioData:", e.detail);  // 디버깅

      // STT API 호출 (여기서는 Google Cloud Speech-to-Text 예시)
      // 실제 사용 시에는 API 키, 설정 등이 필요

      fetch('https://your-stt-api-endpoint', {  // <--- 실제 STT API 엔드포인트로 변경!
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({audio: e.detail})
      })
      .then(response => response.json())
      .then(data => {
        //   console.log("STT Result:", data); // 디버깅
          if (data.transcript) {
              // 인식된 텍스트를 Streamlit chat input에 추가
              inputField.value = data.transcript;
              inputField.dispatchEvent(new Event('input', { bubbles: true })); // Trigger input event

              // UI 업데이트 (트리거) - send message 버튼 클릭
              const sendButton = window.parent.document.querySelector('button[data-testid="send-message-button"]');

              if(sendButton){
                  // console.log('send button found, click!')
                  sendButton.click();

              } else {
                  console.error('Send button not found!');
              }


          }
      })
      .catch(error => console.error('STT Error:', error));
});

</script>
""",
        height=0, # 표시안함
    )

if __name__ == "__main__":
    asyncio.run(main())
