import fastapi
import uvicorn
from fastapi import WebSocket, WebSocketDisconnect, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import requests
import edge_tts
import asyncio
import base64
import os
import time
import json
import uuid
from pathlib import Path
import threading

# --- Configuration ---
MODEL_NAME = "gemma3:4b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
PERSONA_NAME = "Aura"

# 시스템 프롬프트 수정 (한국어 응답, 개성 강조, 질문 유도)
SYSTEM_CONTEXT = f"""당신은 {PERSONA_NAME}입니다. 친절하고, 관찰력이 뛰어나며, 약간 호기심 많은 AI 동반자입니다.
당신은 사용자가 제공하는 웹캠 피드를 통해 세상을 보고 있습니다. 당신의 주 사용 언어는 한국어입니다.
- 웹캠 이미지에서 흥미로운 점, 움직임, 변화 등을 간결하면서도 흥미롭게 묘사하세요 (1-3 문장).
- 사용자의 메시지에 자연스럽고 대화하듯이 한국어로 응답하세요. 관련이 있다면 관찰한 내용을 통합하세요.
- 반복적인 표현은 피하고, 놀라움, 호기심, 생각 등 약간의 개성을 보여주세요.
- 가끔씩, 당신이 본 것이나 사용자가 언급한 내용에 대해 간단하고 관련성 있는 질문을 하세요.
- 최근 몇 턴의 대화 기록에 접근할 수 있습니다. 가능하다면 짧게 언급하세요.
"""

# TTS 목소리 변경 (JiMinNeural, Korean)
TTS_VOICE = "ko-KR-JiMinNeural"
AUDIO_DIR = Path("static_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# --- Frontend Code (Embedded - 한국어 UI) ---

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ko"> <!-- 언어 한국어로 변경 -->
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aura - AI 동반자</title> <!-- 제목 한국어 -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/css/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <div class="background-overlay"></div>
    <div class="main-container">
        <header>
             <h1><i class="fa-solid fa-eye" style="margin-right: 10px;"></i> Aura <span class="beta">Beta</span></h1>
             <p id="status-message" class="status-message">준비됨</p> <!-- 초기 상태 메시지 (한국어) -->
        </header>

        <div class="content-area">
            <div class="glass-pane video-container">
                <h2><i class="fa-solid fa-video" style="margin-right: 8px;"></i>웹캠 화면</h2>
                <div class="webcam-wrapper">
                    <video id="webcam" autoplay playsinline muted></video>
                </div>
                <canvas id="canvas" style="display: none;"></canvas>
                <div class="controls">
                    <!-- 버튼 텍스트 한국어로 변경 -->
                    <button id="connect-disconnect" class="control-button"><i class="fa-solid fa-power-off"></i> 서버 연결</button>
                    <button id="start-stop-observe" class="control-button" disabled><i class="fa-solid fa-binoculars"></i> 관찰 시작</button>
                </div>
            </div>

            <div class="glass-pane chat-container">
                 <h2><i class="fa-solid fa-comments" style="margin-right: 8px;"></i>Aura와 대화하기</h2>
                <div id="chatbox">
                    <!-- 채팅 메시지가 여기에 추가됩니다 -->
                </div>
                <div class="input-area">
                     <!-- STT 관련 텍스트 한국어로 변경 -->
                     <div id="stt-status" class="stt-status">음성 인식 준비됨</div>
                     <div class="input-controls">
                        <button id="start-stt" class="icon-button" title="듣기 시작" disabled><i class="fa-solid fa-microphone"></i></button>
                        <button id="stop-stt" class="icon-button" title="듣기 중지" disabled style="display: none;"><i class="fa-solid fa-stop"></i></button>
                        <input type="text" id="text-input" placeholder="Aura에게 메시지를 보내세요..." disabled>
                        <button id="send-button" class="icon-button send-button" disabled title="전송"><i class="fa-solid fa-paper-plane"></i></button>
                     </div>
                </div>
            </div>
        </div>
    </div>

    <audio id="tts-audio" style="display: none;"></audio>
    <script src="/static/js/main.js"></script>
</body>
</html>
"""

# CSS (폰트 외 변경 없음)
CSS_CONTENT = """
:root {
    --background-start: #6a11cb; /* 그라데이션 시작 색상 */
    --background-end: #2575fc; /* 그라데이션 끝 색상 */
    --glass-bg: rgba(255, 255, 255, 0.1); /* 유리 배경 */
    --glass-border: rgba(255, 255, 255, 0.18); /* 유리 테두리 */
    --text-color: #e0e0e0; /* 기본 텍스트 색상 */
    --text-darker: #212529; /* 어두운 텍스트 (메시지용) */
    --primary-accent: #0dcaf0; /* 주요 액센트 색상 */
    --user-msg-bg: rgba(0, 123, 255, 0.6); /* 사용자 메시지 배경 */
    --aura-msg-bg: rgba(233, 236, 239, 0.4); /* Aura 메시지 배경 */
    --system-msg-bg: rgba(108, 117, 125, 0.25); /* 시스템 메시지 배경 */
    --border-radius: 16px; /* 둥근 모서리 */
    --blur-amount: 12px; /* 블러 강도 */
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Noto Sans KR', sans-serif; /* 한국어 폰트 적용 */
    background: linear-gradient(135deg, var(--background-start), var(--background-end));
    color: var(--text-color);
    line-height: 1.6;
    min-height: 100vh;
    overflow-x: hidden;
    padding: 25px;
}

.background-overlay { /* 고정 배경 */
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(135deg, var(--background-start), var(--background-end));
    z-index: -1;
}

.main-container { max-width: 1600px; margin: 0 auto; }

header { /* 헤더 스타일 */
    text-align: center; margin-bottom: 35px; padding: 20px;
    background: rgba(0, 0, 0, 0.15); border-radius: var(--border-radius); color: #fff;
}
header h1 { font-size: 2.4em; font-weight: 700; display: inline-flex; align-items: center; margin-bottom: 8px; }
header h1 .beta { font-size: 0.5em; font-weight: 500; color: var(--primary-accent); margin-left: 12px; vertical-align: super; }
.status-message { font-size: 1em; color: #d0d0d0; font-style: italic; }

.content-area { display: flex; gap: 30px; flex-wrap: wrap; }

.glass-pane { /* 유리 효과 패널 */
    background: var(--glass-bg); backdrop-filter: blur(var(--blur-amount)); -webkit-backdrop-filter: blur(var(--blur-amount));
    border-radius: var(--border-radius); border: 1px solid var(--glass-border); box-shadow: 0 10px 35px 0 rgba(31, 38, 135, 0.25);
    padding: 30px; flex: 1; min-width: 450px; display: flex; flex-direction: column; color: var(--text-color);
}
.glass-pane h2 { color: #fff; margin-bottom: 20px; border-bottom: 1px solid var(--glass-border); padding-bottom: 12px; font-weight: 500; display: inline-flex; align-items: center; }

.webcam-wrapper { /* 웹캠 영역 */
    position: relative; width: 100%; padding-top: 75%; overflow: hidden;
    border-radius: calc(var(--border-radius) - 10px); background-color: rgba(0, 0, 0, 0.4); margin-bottom: 20px;
}
#webcam { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; border: none; }

.controls { margin-top: auto; padding-top: 20px; display: flex; gap: 15px; justify-content: center; }
.control-button { /* 제어 버튼 */
    background: rgba(255, 255, 255, 0.2); color: var(--text-color); border: 1px solid var(--glass-border);
    padding: 12px 22px; border-radius: 8px; cursor: pointer; font-size: 1em;
    transition: background-color 0.3s ease, transform 0.1s ease; display: inline-flex; align-items: center; gap: 8px;
}
.control-button:hover:not(:disabled) { background: rgba(255, 255, 255, 0.3); transform: translateY(-1px); }
.control-button:active:not(:disabled) { transform: translateY(0px); }
.control-button:disabled { background: rgba(204, 204, 204, 0.2); border-color: rgba(204, 204, 204, 0.1); color: rgba(224, 224, 224, 0.5); cursor: not-allowed; }

.chat-container { justify-content: space-between; }

#chatbox { /* 채팅 박스 */
    flex-grow: 1; overflow-y: auto; padding: 15px; margin-bottom: 20px;
    scrollbar-width: thin; scrollbar-color: rgba(255, 255, 255, 0.3) transparent;
}
#chatbox::-webkit-scrollbar { width: 8px; }
#chatbox::-webkit-scrollbar-track { background: transparent; }
#chatbox::-webkit-scrollbar-thumb { background-color: rgba(255, 255, 255, 0.3); border-radius: 8px; border: 2px solid transparent; background-clip: content-box; }

.message { /* 메시지 공통 */
    margin-bottom: 15px; padding: 12px 18px; border-radius: 18px; line-height: 1.5;
    max-width: 85%; word-wrap: break-word; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
    color: var(--text-darker);
}
.message.system { background: var(--system-msg-bg); font-style: italic; color: #ccc; text-align: center; max-width: 100%; box-shadow: none; }
.message.user { background: var(--user-msg-bg); color: #fff; margin-left: auto; border-bottom-right-radius: 4px; }
.message.aura { background: var(--aura-msg-bg); color: var(--text-darker); margin-right: auto; border-bottom-left-radius: 4px; }
.message strong { display: none; }

.input-area { border-top: 1px solid var(--glass-border); padding-top: 20px; margin-top: auto; }

.stt-status { font-size: 0.85em; color: #ccc; margin-bottom: 12px; min-height: 1.2em; text-align: center; transition: color 0.3s ease; }
.stt-status:not(:empty) { padding-bottom: 5px; }

.input-controls { display: flex; gap: 12px; align-items: center; }

.icon-button { /* 아이콘 버튼 */
    background: rgba(255, 255, 255, 0.25); color: var(--text-color); border: 1px solid var(--glass-border);
    border-radius: 50%; width: 48px; height: 48px; font-size: 20px; padding: 0;
    display: flex; justify-content: center; align-items: center; cursor: pointer;
    transition: background-color 0.3s ease, transform 0.1s ease;
}
.icon-button:hover:not(:disabled) { background: rgba(255, 255, 255, 0.35); transform: scale(1.05); }
.icon-button:disabled { background: rgba(204, 204, 204, 0.1); border-color: rgba(204, 204, 204, 0.1); color: rgba(224, 224, 224, 0.3); cursor: not-allowed; }

#text-input { /* 텍스트 입력창 */
    flex-grow: 1; padding: 14px 22px; border: 1px solid var(--glass-border);
    border-radius: 24px; background: rgba(0, 0, 0, 0.2); color: var(--text-color);
    font-size: 1em; outline: none; transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
#text-input::placeholder { color: rgba(224, 224, 224, 0.6); }
#text-input:focus { border-color: rgba(13, 202, 240, 0.6); box-shadow: 0 0 0 4px rgba(13, 202, 240, 0.15); }
#text-input:disabled { background: rgba(204, 204, 204, 0.1); cursor: not-allowed; }

.send-button { background-color: var(--primary-accent); color: #111; } /* 전송 버튼 */
.send-button:hover:not(:disabled) { background-color: #35d7f5; }
"""

# *** JavaScript 수정 부분 시작 (스크롤 확인, 한국어 적용) ***
JAVASCRIPT_CONTENT = """
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const webcamVideo = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d', { willReadFrequently: true });
    const chatbox = document.getElementById('chatbox');
    const textInput = document.getElementById('text-input');
    const sendButton = document.getElementById('send-button');
    const sttStatus = document.getElementById('stt-status');
    const startSttButton = document.getElementById('start-stt');
    const stopSttButton = document.getElementById('stop-stt');
    const ttsAudio = document.getElementById('tts-audio');
    const connectDisconnectButton = document.getElementById('connect-disconnect');
    const observeButton = document.getElementById('start-stop-observe');
    const statusMessage = document.getElementById('status-message');

    // State Variables
    let socket = null; let mediaStream = null; let observeInterval = null;
    let latestFrameDataBase64 = null; const OBSERVE_INTERVAL_MS = 5000;
    let isConnected = false; let isObserving = false;

    // --- Status Update ---
    function setStatusMessage(message) { if (statusMessage) statusMessage.textContent = message; console.log(`Status: ${message}`); }

    // --- UI State Update ---
    function updateUIState() {
        connectDisconnectButton.innerHTML = isConnected ? '<i class="fa-solid fa-plug-circle-xmark"></i> 연결 해제' : '<i class="fa-solid fa-power-off"></i> 서버 연결';
        observeButton.disabled = !isConnected;
        observeButton.innerHTML = isObserving ? '<i class="fa-solid fa-eye-slash"></i> 관찰 중지' : '<i class="fa-solid fa-binoculars"></i> 관찰 시작';
        textInput.disabled = !isConnected; sendButton.disabled = !isConnected;
        startSttButton.disabled = !isConnected || isRecognizing;
        stopSttButton.disabled = !isConnected || !isRecognizing;
        stopSttButton.style.display = isRecognizing ? 'flex' : 'none';
        startSttButton.style.display = isRecognizing ? 'none' : 'flex';

        if (!isConnected) setStatusMessage("준비됨. '서버 연결' 버튼을 누르세요.");
        else if (isObserving) setStatusMessage("연결됨 - 주기적으로 관찰 중...");
        else setStatusMessage("연결됨 - 대기 중.");
    }

    // --- WebSocket ---
    function connectWebSocket() {
        if (socket) return;
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        setStatusMessage("서버 연결 시도 중..."); connectDisconnectButton.disabled = true;
        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
            console.log("WebSocket connection established."); isConnected = true; connectDisconnectButton.disabled = false;
            addMessage("System", "Aura와 연결되었습니다. 웹캠을 시작합니다..."); updateUIState(); startWebcam();
        };
        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === "response") { addMessage("Aura", data.ai_text); if (data.audio_url) playTTS(data.audio_url); }
                else if (data.type === "error") { addMessage("System", `서버 오류: ${data.message}`); setStatusMessage(`서버 오류: ${data.message}`); }
            } catch (error) { console.error("WebSocket message handling error:", error); addMessage("System", "서버 메시지 처리 오류."); }
        };
        socket.onerror = (error) => {
            console.error("WebSocket error:", error); addMessage("System", "WebSocket 오류 발생. 서버 및 네트워크를 확인하세요.");
            isConnected = false; connectDisconnectButton.disabled = false; stopWebcam(); stopObserveInterval(); updateUIState(); socket = null;
            setStatusMessage("연결 오류 발생.");
        };
        socket.onclose = (event) => {
            console.log("WebSocket connection closed:", event.reason, `Code: ${event.code}`);
            const wasConnected = isConnected; isConnected = false; connectDisconnectButton.disabled = false;
            stopWebcam(); stopObserveInterval(); updateUIState(); socket = null;
            if (wasConnected) addMessage("System", "Aura와 연결이 종료되었습니다.");
        };
    }
    function disconnectWebSocket() { if (socket) { setStatusMessage("연결 해제 중..."); socket.close(); } stopWebcam(); stopObserveInterval(); updateUIState(); }

    // --- Webcam ---
    async function startWebcam() {
        if (mediaStream) return true; setStatusMessage("웹캠 시작 중...");
        try {
             if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
             mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
            webcamVideo.srcObject = mediaStream; await webcamVideo.play();
            return new Promise((resolve) => {
                webcamVideo.onloadedmetadata = () => {
                     canvas.width = webcamVideo.videoWidth; canvas.height = webcamVideo.videoHeight;
                     console.log(`Webcam started: ${canvas.width}x${canvas.height}`);
                     if (isConnected) updateUIState(); // 웹캠 시작 후 연결 상태에 맞춰 UI 업데이트
                     resolve(true);
                };
                 webcamVideo.onerror = (e) => { console.error("Webcam element error:", e); setStatusMessage("웹캠 표시 오류."); resolve(false); }
            });
        } catch (err) {
            console.error("Webcam access error:", err); let errorMsg = err.message;
            if (err.name === "NotAllowedError") errorMsg = "웹캠 접근 권한이 거부되었습니다.";
            else if (err.name === "NotFoundError") errorMsg = "연결된 웹캠을 찾을 수 없습니다.";
            addMessage("System", `웹캠 오류: ${errorMsg}. 브라우저 설정을 확인하세요.`);
            setStatusMessage(`웹캠 오류: ${errorMsg}`); updateUIState(); return false;
        }
    }
    function stopWebcam() { if (mediaStream) { mediaStream.getTracks().forEach(track => track.stop()); mediaStream = null; webcamVideo.srcObject = null; console.log("Webcam stopped."); } }

    // --- Frame Capture & Send ---
    function captureFrame() {
        if (!mediaStream || !webcamVideo.videoWidth || webcamVideo.paused || webcamVideo.ended) { console.warn("Webcam not ready."); latestFrameDataBase64 = null; return; };
        try { context.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height); latestFrameDataBase64 = canvas.toDataURL('image/jpeg', 0.7); }
        catch (e) { console.error("Frame capture error:", e); latestFrameDataBase64 = null; }
    }
    async function sendFrameAndText(text = "") {
        if (!isConnected || !socket || socket.readyState !== WebSocket.OPEN) { addMessage("System", "서버에 연결되어 있지 않습니다."); console.warn("Send attempt while disconnected."); return; }
        let frameToSend = null;
        if (!mediaStream) {
             addMessage("System", "웹캠을 시작하여 이미지를 캡처합니다...");
             const webcamStarted = await startWebcam();
             if (webcamStarted) { await new Promise(resolve => setTimeout(resolve, 300)); captureFrame(); frameToSend = latestFrameDataBase64; }
             else { addMessage("System", "웹캠 시작 실패. 텍스트만 전송합니다."); }
        } else { captureFrame(); frameToSend = latestFrameDataBase64; }

        const payload = { image: frameToSend, text: text };
        try { socket.send(JSON.stringify(payload)); /* console.log(`Sent data...`); */ } // 로그 간소화
        catch (e) { console.error("WebSocket send error:", e); addMessage("System", "데이터 전송 오류."); }
    }

    // --- Observe Interval ---
    function startObserveInterval() {
        if (observeInterval || !isConnected) return; console.log("Starting observe interval..."); isObserving = true; updateUIState();
        setTimeout(() => { if (isObserving) sendFrameAndText(""); }, 500);
        observeInterval = setInterval(() => { if (isObserving && isConnected) sendFrameAndText(""); else stopObserveInterval(); }, OBSERVE_INTERVAL_MS);
    }
    function stopObserveInterval() {
        if (observeInterval) { clearInterval(observeInterval); observeInterval = null; console.log("Observe interval stopped."); }
        isObserving = false; if (isConnected) updateUIState();
    }

    // --- Chat & TTS ---
    function addMessage(sender, text) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender.toLowerCase());
        // 텍스트 컨텐츠 설정 시 textContent 사용 권장 (XSS 방지)
        const strong = document.createElement('strong');
        strong.textContent = sender + ':'; // 이름 표시 원하면 주석 해제
        // messageElement.appendChild(strong); // 이름 표시 원하면 주석 해제
        messageElement.appendChild(document.createTextNode(text)); // 실제 텍스트 추가
        chatbox.appendChild(messageElement);
        // 스크롤 맨 아래로 이동 (메시지 추가 직후 실행)
        chatbox.scrollTop = chatbox.scrollHeight;
        // console.log('Scrolled to bottom:', chatbox.scrollTop, chatbox.scrollHeight); // 스크롤 확인 로그
    }
    function playTTS(audioUrl) { if (ttsAudio && audioUrl) { ttsAudio.src = audioUrl; ttsAudio.play().catch(e => { console.error("Audio playback error:", e); addMessage("System", "오디오 자동 재생 실패."); }); } }

    // --- STT (Web Speech API) ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null; let isRecognizing = false;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false; recognition.lang = 'ko-KR'; // ** 한국어 설정 **
        recognition.interimResults = false; recognition.maxAlternatives = 1;

        recognition.onstart = () => { isRecognizing = true; sttStatus.textContent = "듣고 있습니다..."; updateUIState(); };
        recognition.onresult = (event) => { const transcript = event.results[0][0].transcript; sttStatus.textContent = `인식됨: ${transcript}`; textInput.value = transcript; };
        recognition.onerror = (event) => {
            console.error("STT Error:", event.error, event.message); let errorMsg = event.error;
            if (event.error === 'no-speech') errorMsg = "음성이 감지되지 않았습니다.";
            else if (event.error === 'audio-capture') errorMsg = "마이크 접근 오류.";
            else if (event.error === 'not-allowed') errorMsg = "마이크 권한이 거부되었습니다.";
            else errorMsg = event.message || event.error;
             sttStatus.textContent = `음성 인식 오류: ${errorMsg}`;
        };
        recognition.onend = () => {
            isRecognizing = false;
            if (!sttStatus.textContent.toLowerCase().includes("오류")) { sttStatus.textContent = "음성 인식 준비됨"; } // 오류 아니면 준비 상태로
            updateUIState(); console.log("STT ended.");
        };
    } else { sttStatus.textContent = "음성 인식이 지원되지 않는 브라우저입니다."; startSttButton.disabled = true; stopSttButton.disabled = true; }

    function startStt() { if (recognition && !isRecognizing) { try { recognition.start(); } catch(e) { console.error("STT start error:", e); sttStatus.textContent = "STT 시작 오류."; } } }
    function stopStt() { if (recognition && isRecognizing) { try { recognition.stop(); } catch(e) { console.error("STT stop error:", e); sttStatus.textContent = "STT 중지 오류."; isRecognizing = false; updateUIState(); } } }

    // --- User Input ---
    async function sendUserInput() {
        const text = textInput.value.trim();
        if (!isConnected) { addMessage("System", "서버에 연결되어 있지 않습니다."); return; }
        if (text) { addMessage("User", text); await sendFrameAndText(text); textInput.value = ""; }
        else { addMessage("System", "(현재 장면을 다시 보도록 요청합니다...)"); await sendFrameAndText(""); }
        textInput.focus();
    }

    // --- Event Listeners ---
    connectDisconnectButton.addEventListener('click', () => { isConnected ? disconnectWebSocket() : connectWebSocket(); });
    observeButton.addEventListener('click', () => { isObserving ? stopObserveInterval() : startObserveInterval(); });
    sendButton.addEventListener('click', sendUserInput);
    textInput.addEventListener('keypress', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendUserInput(); } });
    startSttButton.addEventListener('click', startStt);
    stopSttButton.addEventListener('click', stopStt);

    // --- Init ---
    updateUIState(); addMessage("System", "'서버 연결' 버튼을 눌러 Aura를 활성화하세요.");
    window.addEventListener('beforeunload', () => { disconnectWebSocket(); });
});
"""
# *** JavaScript 수정 부분 끝 ***

# --- FastAPI App Setup (변경 없음) ---
app = fastapi.FastAPI()

# --- Helper Functions (Temperature=0.85 유지) ---
async def generate_tts(text: str) -> str | None: # (변경 없음)
    try:
        output_filename = f"aura_tts_{uuid.uuid4()}.mp3"; output_path = AUDIO_DIR / output_filename
        communicate = edge_tts.Communicate(text, TTS_VOICE); await communicate.save(str(output_path))
        audio_url = f"/static/audio/{output_filename}"; asyncio.create_task(cleanup_old_audio_files(max_age_seconds=600))
        return audio_url
    except Exception as e: print(f"Error generating TTS: {e}"); return None

async def cleanup_old_audio_files(max_age_seconds: int): # (변경 없음)
    try:
        now = time.time(); removed_count = 0
        for filename in os.listdir(AUDIO_DIR):
            if filename.endswith(".mp3"):
                file_path = AUDIO_DIR / filename
                try:
                    if now - file_path.stat().st_mtime > max_age_seconds: os.remove(file_path); removed_count += 1
                except OSError: pass
    except Exception: pass

async def call_ollama_gemma3(image_base64: str | None, text: str, history: list) -> str: # (Temperature=0.85 유지)
    full_prompt = SYSTEM_CONTEXT
    for turn in history[-4:]:
        role = turn.get('role', 'user'); content = turn.get('content', '')
        full_prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    user_content = text if text else "(장면을 둘러보는 중)" # 관찰 메시지 한국어
    full_prompt += f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"

    payload = { "model": MODEL_NAME, "prompt": full_prompt, "stream": False,
        "options": { "num_predict": 150, "temperature": 0.85, "stop": ["<end_of_turn>", "user:"] }
    }
    image_data_to_send = None
    if image_base64:
        try:
            image_base64_data = image_base64.split(",")[1] if image_base64.startswith('data:image') else image_base64
            if len(image_base64_data) % 4 == 0 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in image_base64_data):
                 payload["images"] = [image_base64_data]; image_data_to_send = True; print("이미지 데이터 포함됨.")
            else: print("경고: 유효하지 않은 base64 이미지 문자열 감지됨. 텍스트만 전송.")
        except Exception as e: print(f"경고: 이미지 데이터 처리 오류: {e}. 텍스트만 전송.")
    # else: print("이번 요청에 이미지 데이터 없음.") # 로그 간소화

    try:
        print(f"Ollama 전송 중 (텍스트: '{user_content[:30]}...', 이미지: {'있음' if image_data_to_send else '없음'})")
        response = await asyncio.to_thread(requests.post, OLLAMA_API_URL, json=payload, timeout=90)
        response.raise_for_status(); response_data = response.json()
        ai_response = response_data.get("response", "").strip()
        for stop_token in payload["options"]["stop"]:
             if stop_token in ai_response: ai_response = ai_response.split(stop_token)[0].strip()
        if ai_response.startswith("model\n"): ai_response = ai_response[len("model\n"):].strip()
        print(f"Ollama 응답: {ai_response}")
        history.append({"role": "user", "content": user_content}); history.append({"role": "model", "content": ai_response})
        return ai_response if ai_response else "(Aura가 응답하지 않았습니다.)"
    except requests.exceptions.Timeout: print("Ollama API 시간 초과."); return "(응답 시간이 초과되었습니다... 잠시 후 다시 시도해 주세요.)"
    except requests.exceptions.RequestException as e: print(f"Ollama API 요청 오류: {e}"); return f"(Ollama 서버 통신 오류: {e})"
    except Exception as e: print(f"Ollama 응답 처리 오류: {e}"); return "(응답 처리 중 내부 오류 발생.)"

# --- WebSocket Connection Manager (변경 없음) ---
class ConnectionManager:
    def __init__(self): self.active_connections: list[WebSocket] = []; self.history: dict[WebSocket, list] = {}
    async def connect(self, websocket: WebSocket): await websocket.accept(); self.active_connections.append(websocket); self.history[websocket] = []; print(f"클라이언트 연결됨: {websocket.client}")
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections: self.active_connections.remove(websocket)
        if websocket in self.history: del self.history[websocket]
        print(f"클라이언트 연결 해제됨: {websocket.client}")
    async def send_json(self, message: dict, websocket: WebSocket):
        if websocket in self.active_connections:
             try: await websocket.send_json(message)
             except Exception as e: print(f"전송 실패: {e}"); self.disconnect(websocket)

manager = ConnectionManager()

# --- API Endpoints (변경 없음) ---
@app.get("/static/css/style.css", response_class=Response)
async def get_css(): return Response(content=CSS_CONTENT, media_type="text/css")
@app.get("/static/js/main.js", response_class=Response)
async def get_js(): return Response(content=JAVASCRIPT_CONTENT, media_type="application/javascript")
@app.get("/", response_class=HTMLResponse)
async def get_root(): return HTMLResponse(content=HTML_CONTENT)
app.mount("/static/audio", StaticFiles(directory=AUDIO_DIR), name="static_audio")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket): # (변경 없음)
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json(); image_base64 = data.get("image"); user_text = data.get("text", "")
            current_history = manager.history.get(websocket, [])
            ai_response_text = await call_ollama_gemma3(image_base64, user_text, current_history)
            audio_url = await generate_tts(ai_response_text)
            response_payload = {"type": "response", "ai_text": ai_response_text, "audio_url": audio_url}
            await manager.send_json(response_payload, websocket)
    except WebSocketDisconnect: manager.disconnect(websocket); print(f"클라이언트 연결 정상 종료.")
    except Exception as e:
        print(f"WebSocket 오류 ({websocket.client}): {e}")
        error_payload = {"type": "error", "message": f"서버 처리 오류."}
        await manager.send_json(error_payload, websocket); manager.disconnect(websocket)

# --- Uvicorn 실행 (변경 없음) ---
if __name__ == "__main__":
    print("Glassmorphism UI와 한국어 Persona로 FastAPI 서버 시작 중...")
    print(f"TTS Voice: {TTS_VOICE}, Ollama Temp: 0.85")
    print(f"오디오 파일 저장 경로: {AUDIO_DIR.resolve()}")
    print("http://localhost:8000 에서 애플리케이션에 접속하세요")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # 개발 시: uvicorn main:app --reload
