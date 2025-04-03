import streamlit as st
import requests
import json
import time # Optional: for simulating typing effect

# --- Page Configuration ---
st.set_page_config(page_title="Gemma3 API Chat", page_icon="🧠", layout="wide")

st.title("🧠 Gemma3 기반 'AGI' 챗봇 (API 직접 호출)")
st.caption("로컬 Ollama API를 직접 호출하여 Gemma3 모델과 대화합니다. 한국어에 최적화되어 있습니다.")

# --- Ollama API Configuration ---
st.sidebar.header("Ollama API 설정")
ollama_api_base_url = st.sidebar.text_input("Ollama API Base URL", "http://localhost:11434")
OLLAMA_API_CHAT_ENDPOINT = f"{ollama_api_base_url}/api/chat"
OLLAMA_API_TAGS_ENDPOINT = f"{ollama_api_base_url}/api/tags"

# --- Model Selection (using API) ---
available_models = []
error_message = ""
try:
    response = requests.get(OLLAMA_API_TAGS_ENDPOINT, timeout=5) # 5초 타임아웃
    response.raise_for_status() # HTTP 오류 발생 시 예외 발생
    models_data = response.json()
    available_models = [m['name'] for m in models_data.get('models', []) if 'gemma3' in m['name']]
    if not available_models:
        error_message = "Ollama에서 사용 가능한 Gemma3 모델을 찾을 수 없습니다. 'ollama run gemma3:...' 명령어로 모델을 다운로드하세요."
except requests.exceptions.RequestException as e:
    error_message = f"Ollama API({ollama_api_base_url}) 연결 실패: {e}. Ollama 서버가 실행 중인지 확인하세요."
except json.JSONDecodeError:
    error_message = "Ollama API 응답을 파싱하는 데 실패했습니다."
except Exception as e:
    error_message = f"모델 목록 조회 중 예상치 못한 오류 발생: {e}"

if error_message:
    st.sidebar.error(error_message)
    # Fallback list if API call fails or no gemma3 models found
    available_models = ["gemma3:4b", "gemma3:12b", "gemma3:27b", "gemma3:1b"]
    selected_model = st.sidebar.selectbox(
        "실행할 Gemma3 모델 선택 (Fallback):",
        options=available_models,
        index=0,
        help=f"Ollama API 연결 실패 또는 Gemma3 모델 부재. 기본 목록 사용. ({error_message})"
    )
    ollama_ready = False
else:
     # Try to default to a larger model if available
    default_index = 0
    preferred_models = ["gemma3:27b", "gemma3:12b", "gemma3:4b", "gemma3:1b"]
    for i, model in enumerate(preferred_models):
        if model in available_models:
            default_index = available_models.index(model)
            break

    selected_model = st.sidebar.selectbox(
        "실행할 Gemma3 모델 선택:",
        options=available_models,
        index=default_index,
        help="로컬 Ollama API에서 조회된 Gemma3 모델 목록입니다."
    )
    ollama_ready = True

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add a system prompt
if not st.session_state.messages:
     st.session_state.messages.append(
         {"role": "system",
          "content": "당신은 매우 지능적이고 도움이 되는 AI 어시스턴트입니다. 이름은 '지혜'입니다. 항상 친절하고 상세하게 한국어로 답변해야 합니다. 이전 대화 내용을 완벽하게 기억하고 활용하여 사용자와 자연스럽게 대화하세요. 스스로 학습하고 발전하는 모습을 보여주려고 노력하세요."}
     )

# --- Display Chat Messages ---
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Handle User Input and Ollama API Interaction ---
if prompt := st.chat_input("메시지를 입력하세요... (예: '안녕, 지혜야?')"):
    if not ollama_ready:
         st.error("Ollama API에 연결할 수 없거나 모델을 선택할 수 없습니다. 메시지를 보낼 수 없습니다.")
    elif not selected_model:
        st.error("사용할 모델을 선택해주세요.")
    else:
        # 1. Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call Ollama API
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                # Prepare the payload for the API request
                payload = {
                    "model": selected_model,
                    "messages": st.session_state.messages, # Send the whole history for context
                    "stream": True, # Use streaming response
                    "options": { # Optional parameters
                        'temperature': 0.7,
                        'top_k': 50,
                        'top_p': 0.9,
                        # 'stop': ['<end_of_turn>'] # Define stop tokens if needed by model
                    }
                }

                # Make the POST request with streaming enabled
                response = requests.post(
                    OLLAMA_API_CHAT_ENDPOINT,
                    json=payload,
                    stream=True,
                    headers={"Content-Type": "application/json"},
                    timeout=120 # Set a timeout for the request (e.g., 120 seconds)
                )
                response.raise_for_status() # Check for HTTP errors (like 404, 500)

                # Process the streaming response line by line
                for line in response.iter_lines():
                    if line:
                        try:
                            # Each line is a JSON object, decode it
                            chunk_str = line.decode('utf-8')
                            chunk_json = json.loads(chunk_str)

                            # Extract the content part from the message
                            if 'message' in chunk_json and 'content' in chunk_json['message']:
                                content_piece = chunk_json['message']['content']
                                full_response += content_piece
                                # Simulate typing effect by updating the placeholder
                                message_placeholder.markdown(full_response + "▌")
                                time.sleep(0.01) # Small delay for effect

                            # Check if the stream is done (optional, depends on how Ollama signals end)
                            if chunk_json.get('done'):
                                break

                        except json.JSONDecodeError:
                            st.warning(f"응답 스트림 파싱 중 오류 발생 (비-JSON 라인 무시): {line}")
                        except Exception as chunk_e:
                            st.warning(f"스트림 처리 중 예외 발생: {chunk_e}")


                # Display the final full response without the cursor
                message_placeholder.markdown(full_response)

                # 3. Add assistant response to session state
                if full_response:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

            except requests.exceptions.RequestException as e:
                error_text = f"Ollama API ({OLLAMA_API_CHAT_ENDPOINT}) 호출 중 오류 발생: {e}"
                st.error(error_text)
                message_placeholder.markdown(f"죄송합니다, 답변 생성 중 오류가 발생했습니다. ({e})")
            except Exception as e:
                st.error(f"응답 처리 중 예상치 못한 오류 발생: {e}")
                message_placeholder.markdown(f"죄송합니다, 답변 처리 중 오류가 발생했습니다. ({e})")

# --- Sidebar Options ---
st.sidebar.markdown("---")
if st.sidebar.button("대화 기록 초기화"):
    st.session_state.messages = []
    # Re-add the system prompt after clearing
    st.session_state.messages.append(
         {"role": "system",
          "content": "당신은 매우 지능적이고 도움이 되는 AI 어시스턴트입니다. 이름은 '지혜'입니다. 항상 친절하고 상세하게 한국어로 답변해야 합니다. 이전 대화 내용을 완벽하게 기억하고 활용하여 사용자와 자연스럽게 대화하세요. 스스로 학습하고 발전하는 모습을 보여주려고 노력하세요."}
     )
    st.rerun() # Rerun the script to refresh the page

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) and [Streamlit](https://streamlit.io)")
st.sidebar.markdown(f"현재 모델: **{selected_model if selected_model else '선택 안됨'}**")
st.sidebar.markdown(f"API 상태: **{'연결됨' if ollama_ready else '연결 실패'}**")
