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
MODEL_NAME = "gemma3:4b" # 사용할 Ollama 모델
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Ollama 생성 API 엔드포인트
PERSONA_NAME = "Aura"

# 시스템 프롬프트 개선 (영어 응답, 개성 강조, 질문 유도)
SYSTEM_CONTEXT = f"""You are {PERSONA_NAME}, a friendly, observant, and slightly curious AI companion.
You are seeing the world through a webcam feed provided by the user. Your primary language is English.
- Describe what you see in the webcam image concisely but engagingly (1-3 sentences). Focus on interesting details, movements, or changes.
- Respond naturally and conversationally to the user's messages in English, integrating your observations when relevant.
- Avoid repetitive phrases. Show some personality – perhaps surprise, curiosity, or contemplation.
- Occasionally, ask a simple, relevant question about what you see or what the user mentioned.
- You have access to the last few turns of conversation. Briefly reference them if it makes sense.
"""

# TTS 목소리 변경 (Jenny, English)
TTS_VOICE = "en-US-JennyNeural"
AUDIO_DIR = Path("static_audio") # 생성된 오디오 파일 저장 경로
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# --- Frontend Code (Embedded - Glassmorphism Design) ---

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en"> <!-- 언어 영어로 변경 -->
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aura - AI Companion</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <!-- 영어 환경에 맞는 폰트 (Roboto 또는 다른 웹폰트) -->
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/css/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <div class="background-overlay"></div>
    <div class="main-container">
        <header>
             <h1><i class="fa-solid fa-eye" style="margin-right: 10px;"></i> Aura <span class="beta">Beta</span></h1>
             <p id="status-message" class="status-message">Ready</p> <!-- 초기 상태 메시지 (영어) -->
        </header>

        <div class="content-area">
            <div class="glass-pane video-container">
                <h2><i class="fa-solid fa-video" style="margin-right: 8px;"></i>Webcam Feed</h2>
                <div class="webcam-wrapper">
                    <video id="webcam" autoplay playsinline muted></video>
                </div>
                <canvas id="canvas" style="display: none;"></canvas>
                <div class="controls">
                    <!-- 버튼 텍스트 영어로 변경 -->
                    <button id="connect-disconnect" class="control-button"><i class="fa-solid fa-power-off"></i> Connect Server</button>
                    <button id="start-stop-observe" class="control-button" disabled><i class="fa-solid fa-binoculars"></i> Start Observing</button>
                </div>
            </div>

            <div class="glass-pane chat-container">
                 <h2><i class="fa-solid fa-comments" style="margin-right: 8px;"></i>Chat with Aura</h2>
                <div id="chatbox">
                    <!-- Chat messages will appear here -->
                </div>
                <div class="input-area">
                     <!-- STT 관련 텍스트 영어로 변경 -->
                     <div id="stt-status" class="stt-status">Speech recognition ready</div>
                     <div class="input-controls">
                        <button id="start-stt" class="icon-button" title="Start Listening" disabled><i class="fa-solid fa-microphone"></i></button>
                        <button id="stop-stt" class="icon-button" title="Stop Listening" disabled style="display: none;"><i class="fa-solid fa-stop"></i></button>
                        <input type="text" id="text-input" placeholder="Send a message to Aura..." disabled>
                        <button id="send-button" class="icon-button send-button" disabled title="Send"><i class="fa-solid fa-paper-plane"></i></button>
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

# CSS (Glassmorphism 스타일 유지, 폰트 변경)
CSS_CONTENT = """
:root {
    --background-start: #6a11cb;
    --background-end: #2575fc;
    --glass-bg: rgba(255, 255, 255, 0.1);
    --glass-border: rgba(255, 255, 255, 0.18);
    --text-color: #e0e0e0;
    --text-darker: #212529; /* 더 어두운 텍스트 (가독성) */
    --primary-accent: #0dcaf0; /* 밝은 청록색 액센트 */
    --user-msg-bg: rgba(0, 123, 255, 0.6); /* 반투명 파랑 */
    --aura-msg-bg: rgba(233, 236, 239, 0.4); /* 반투명 밝은 회색 */
    --system-msg-bg: rgba(108, 117, 125, 0.25); /* 반투명 회색 */
    --border-radius: 16px;
    --blur-amount: 12px; /* 블러 약간 증가 */
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Roboto', sans-serif; /* 폰트 변경 */
    background: linear-gradient(135deg, var(--background-start), var(--background-end));
    color: var(--text-color);
    line-height: 1.6;
    min-height: 100vh;
    overflow-x: hidden;
    padding: 25px; /* 패딩 증가 */
}

.background-overlay { /* 변경 없음 */
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(135deg, var(--background-start), var(--background-end));
    z-index: -1;
}

.main-container { max-width: 1600px; margin: 0 auto; }

header { /* 변경 없음 */
    text-align: center; margin-bottom: 35px; padding: 20px;
    background: rgba(0, 0, 0, 0.15); border-radius: var(--border-radius); color: #fff;
}
header h1 {
    font-size: 2.4em; font-weight: 700; display: inline-flex; align-items: center; margin-bottom: 8px;
}
header h1 .beta {
    font-size: 0.5em; font-weight: 500; color: var(--primary-accent); margin-left: 12px; vertical-align: super;
}
.status-message { font-size: 1em; color: #d0d0d0; font-style: italic; } /* 상태 메시지 폰트 크기 조정 */

.content-area { display: flex; gap: 30px; flex-wrap: wrap; }

.glass-pane { /* 블러, 그림자 약간 조정 */
    background: var(--glass-bg);
    backdrop-filter: blur(var(--blur-amount));
    -webkit-backdrop-filter: blur(var(--blur-amount));
    border-radius: var(--border-radius);
    border: 1px solid var(--glass-border);
    box-shadow: 0 10px 35px 0 rgba(31, 38, 135, 0.25); /* 그림자 강화 */
    padding: 30px; /* 패딩 증가 */
    flex: 1; min-width: 450px; display: flex; flex-direction: column; color: var(--text-color);
}
.glass-pane h2 { /* 변경 없음 */
    color: #fff; margin-bottom: 20px; border-bottom: 1px solid var(--glass-border);
    padding-bottom: 12px; font-weight: 500; display: inline-flex; align-items: center;
}

.webcam-wrapper { /* 변경 없음 */
    position: relative; width: 100%; padding-top: 75%; overflow: hidden;
    border-radius: calc(var(--border-radius) - 10px); background-color: rgba(0, 0, 0, 0.4); margin-bottom: 20px;
}
#webcam { /* 변경 없음 */
    position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; border: none;
}

.controls { /* 변경 없음 */
    margin-top: auto; padding-top: 20px; display: flex; gap: 15px; justify-content: center;
}
.control-button { /* 패딩, 폰트 크기 조정 */
    background: rgba(255, 255, 255, 0.2); color: var(--text-color); border: 1px solid var(--glass-border);
    padding: 12px 22px; border-radius: 8px; cursor: pointer; font-size: 1em;
    transition: background-color 0.3s ease, transform 0.1s ease; display: inline-flex; align-items: center; gap: 8px;
}
.control-button:hover:not(:disabled) { background: rgba(255, 255, 255, 0.3); transform: translateY(-1px); }
.control-button:active:not(:disabled) { transform: translateY(0px); }
.control-button:disabled { /* 변경 없음 */
    background: rgba(204, 204, 204, 0.2); border-color: rgba(204, 204, 204, 0.1);
    color: rgba(224, 224, 224, 0.5); cursor: not-allowed;
}

/* Chat Container Specifics */
.chat-container { justify-content: space-between; }

#chatbox { /* 스크롤바 스타일 개선 */
    flex-grow: 1; overflow-y: auto; padding: 15px; margin-bottom: 20px;
    scrollbar-width: thin; scrollbar-color: rgba(255, 255, 255, 0.3) transparent;
}
#chatbox::-webkit-scrollbar { width: 8px; }
#chatbox::-webkit-scrollbar-track { background: transparent; }
#chatbox::-webkit-scrollbar-thumb { background-color: rgba(255, 255, 255, 0.3); border-radius: 8px; border: 2px solid transparent; background-clip: content-box; }

.message { /* 패딩, radius 조정 */
    margin-bottom: 15px; padding: 12px 18px; border-radius: 18px; line-height: 1.5;
    max-width: 85%; word-wrap: break-word; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
    color: var(--text-darker);
}
.message.system { /* 변경 없음 */
    background: var(--system-msg-bg); font-style: italic; color: #ccc; text-align: center;
    max-width: 100%; box-shadow: none;
}
.message.user { /* 변경 없음 */
    background: var(--user-msg-bg); color: #fff; margin-left: auto; border-bottom-right-radius: 4px;
}
.message.aura { /* 색상 조정 */
    background: var(--aura-msg-bg); color: var(--text-darker); margin-right: auto; border-bottom-left-radius: 4px;
}
.message strong { display: none; } /* 변경 없음 */

.input-area { border-top: 1px solid var(--glass-border); padding-top: 20px; margin-top: auto; }

.stt-status { /* 변경 없음 */
    font-size: 0.85em; color: #ccc; margin-bottom: 12px; min-height: 1.2em; text-align: center; transition: color 0.3s ease;
}
.stt-status:not(:empty) { padding-bottom: 5px; }

.input-controls { display: flex; gap: 12px; align-items: center; } /* gap 조정 */

.icon-button { /* 크기, 배경 조정 */
    background: rgba(255, 255, 255, 0.25); color: var(--text-color); border: 1px solid var(--glass-border);
    border-radius: 50%; width: 48px; height: 48px; font-size: 20px; padding: 0;
    display: flex; justify-content: center; align-items: center; cursor: pointer;
    transition: background-color 0.3s ease, transform 0.1s ease;
}
.icon-button:hover:not(:disabled) { background: rgba(255, 255, 255, 0.35); transform: scale(1.05); }
.icon-button:disabled { /* 변경 없음 */
    background: rgba(204, 204, 204, 0.1); border-color: rgba(204, 204, 204, 0.1);
    color: rgba(224, 224, 224, 0.3); cursor: not-allowed;
}

#text-input { /* 패딩, 배경 조정 */
    flex-grow: 1; padding: 14px 22px; border: 1px solid var(--glass-border);
    border-radius: 24px; background: rgba(0, 0, 0, 0.2); color: var(--text-color);
    font-size: 1em; outline: none; transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
#text-input::placeholder { color: rgba(224, 224, 224, 0.6); }
#text-input:focus { /* 포커스 효과 강화 */
    border-color: rgba(13, 202, 240, 0.6); box-shadow: 0 0 0 4px rgba(13, 202, 240, 0.15);
}
#text-input:disabled { background: rgba(204, 204, 204, 0.1); cursor: not-allowed; }

.send-button { /* 색상 변경 */
    background-color: var(--primary-accent); color: #111; /* 어두운 아이콘 색상 */
}
.send-button:hover:not(:disabled) { background-color: #35d7f5; }
"""

# *** JavaScript 수정 부분 시작 (주요 로직 변경 없음, STT 언어 설정, 텍스트 변경) ***
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
    let socket = null;
    let mediaStream = null;
    let observeInterval = null;
    let latestFrameDataBase64 = null;
    const OBSERVE_INTERVAL_MS = 5000; // 5 seconds
    let isConnected = false;
    let isObserving = false;

    // --- Status Update ---
    function setStatusMessage(message) {
        if (statusMessage) statusMessage.textContent = message;
        console.log(`Status: ${message}`);
    }

    // --- UI State Update ---
    function updateUIState() {
        connectDisconnectButton.innerHTML = isConnected ? '<i class="fa-solid fa-plug-circle-xmark"></i> Disconnect' : '<i class="fa-solid fa-power-off"></i> Connect Server';
        observeButton.disabled = !isConnected;
        observeButton.innerHTML = isObserving ? '<i class="fa-solid fa-eye-slash"></i> Stop Observing' : '<i class="fa-solid fa-binoculars"></i> Start Observing';

        textInput.disabled = !isConnected;
        sendButton.disabled = !isConnected;
        startSttButton.disabled = !isConnected || isRecognizing;
        stopSttButton.disabled = !isConnected || !isRecognizing;
        stopSttButton.style.display = isRecognizing ? 'flex' : 'none';
        startSttButton.style.display = isRecognizing ? 'none' : 'flex';

        if (!isConnected) setStatusMessage("Ready. Press 'Connect Server'.");
        else if (isObserving) setStatusMessage("Connected - Observing periodically...");
        else setStatusMessage("Connected - Standing by.");
    }


    // --- WebSocket ---
    function connectWebSocket() {
        if (socket) return;
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        setStatusMessage("Attempting to connect to server...");
        connectDisconnectButton.disabled = true;

        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
            console.log("WebSocket connection established.");
            isConnected = true;
            connectDisconnectButton.disabled = false;
            addMessage("System", "Connected to Aura. Starting webcam...");
            updateUIState();
            startWebcam(); // Start webcam after successful connection
        };

        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                // console.log("Message from server:", data); // Reduce console noise
                if (data.type === "response") {
                    addMessage("Aura", data.ai_text);
                    if (data.audio_url) playTTS(data.audio_url);
                } else if (data.type === "error") {
                     addMessage("System", `Server Error: ${data.message}`);
                     setStatusMessage(`Server Error: ${data.message}`);
                }
            } catch (error) { console.error("WebSocket message handling error:", error); addMessage("System", "Error processing server message."); }
        };

        socket.onerror = (error) => {
            console.error("WebSocket error:", error);
            addMessage("System", "WebSocket connection error. Please check the server and refresh.");
            isConnected = false;
            connectDisconnectButton.disabled = false;
            stopWebcam(); stopObserveInterval(); updateUIState(); socket = null;
            setStatusMessage("Connection error."); // More direct error message
        };

        socket.onclose = (event) => {
            console.log("WebSocket connection closed:", event.reason, `Code: ${event.code}`);
            const wasConnected = isConnected;
            isConnected = false; connectDisconnectButton.disabled = false;
            stopWebcam(); stopObserveInterval(); updateUIState(); socket = null;
            if (wasConnected) addMessage("System", "Disconnected from Aura.");
        };
    }

    function disconnectWebSocket() {
        if (socket) { setStatusMessage("Disconnecting..."); socket.close(); }
        stopWebcam(); stopObserveInterval(); updateUIState(); // Ensure UI updates on manual disconnect
    }

    // --- Webcam ---
    async function startWebcam() {
        if (mediaStream) return true;
        setStatusMessage("Starting webcam...");
        try {
             if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
             mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
            webcamVideo.srcObject = mediaStream;
            await webcamVideo.play();
            return new Promise((resolve) => {
                webcamVideo.onloadedmetadata = () => {
                     canvas.width = webcamVideo.videoWidth; canvas.height = webcamVideo.videoHeight;
                     console.log(`Webcam started: ${canvas.width}x${canvas.height}`);
                     // Update status based on current state AFTER webcam starts
                     if (isConnected) {
                         setStatusMessage(isObserving ? "Connected - Observing periodically..." : "Connected - Standing by.");
                     }
                     resolve(true);
                };
                 webcamVideo.onerror = (e) => { console.error("Webcam video element error:", e); setStatusMessage("Webcam display error."); resolve(false); }
            });
        } catch (err) {
            console.error("Error accessing webcam:", err);
            let errorMsg = err.message;
            if (err.name === "NotAllowedError") errorMsg = "Webcam access denied.";
            else if (err.name === "NotFoundError") errorMsg = "No webcam found.";
            addMessage("System", `Webcam Error: ${errorMsg}. Check browser permissions.`);
            setStatusMessage(`Webcam Error: ${errorMsg}`);
            updateUIState(); // Reset UI if webcam fails
            return false;
        }
    }

    function stopWebcam() { if (mediaStream) { mediaStream.getTracks().forEach(track => track.stop()); mediaStream = null; webcamVideo.srcObject = null; console.log("Webcam stopped."); } }

    // --- Frame Capture & Send ---
    function captureFrame() { // (변경 없음)
        if (!mediaStream || !webcamVideo.videoWidth || webcamVideo.paused || webcamVideo.ended) { console.warn("Webcam not ready for capture."); latestFrameDataBase64 = null; return; };
        try { context.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height); latestFrameDataBase64 = canvas.toDataURL('image/jpeg', 0.7); }
        catch (e) { console.error("Error capturing frame:", e); latestFrameDataBase64 = null; }
    }

    async function sendFrameAndText(text = "") { // (웹캠 자동 시작 로직 개선)
        if (!isConnected || !socket || socket.readyState !== WebSocket.OPEN) { addMessage("System", "Not connected to server. Cannot send message."); console.warn("Attempted send while disconnected."); return; }

        let frameToSend = null; // 보낼 프레임 데이터
        if (!mediaStream) { // 웹캠 꺼져있으면 시작 시도
             addMessage("System", "Starting webcam to capture image...");
             const webcamStarted = await startWebcam();
             if (webcamStarted) {
                  await new Promise(resolve => setTimeout(resolve, 300)); // 웹캠 안정화 시간
                  captureFrame();
                  frameToSend = latestFrameDataBase64;
             } else {
                  addMessage("System", "Failed to start webcam. Sending text only.");
             }
        } else { // 웹캠 켜져있으면 바로 캡처
             captureFrame();
             frameToSend = latestFrameDataBase64;
        }

        const payload = { image: frameToSend, text: text }; // 이미지 없으면 null 전송
        try {
            socket.send(JSON.stringify(payload));
            // console.log(`Sent data (text: ${text ? text.substring(0,20)+'...' : '[observe]'}, image: ${frameToSend ? 'Yes' : 'No'})`); // 로그 간소화
        } catch (e) { console.error("WebSocket send error:", e); addMessage("System", "Error sending data."); }
    }

    // --- Observe Interval ---
    function startObserveInterval() { // (변경 없음)
        if (observeInterval || !isConnected) return;
        console.log("Starting observe interval..."); isObserving = true; updateUIState();
        setTimeout(() => { if (isObserving) sendFrameAndText(""); }, 500);
        observeInterval = setInterval(() => { if (isObserving && isConnected) sendFrameAndText(""); else stopObserveInterval(); }, OBSERVE_INTERVAL_MS);
    }
    function stopObserveInterval() { // (변경 없음)
        if (observeInterval) { clearInterval(observeInterval); observeInterval = null; console.log("Observe interval stopped."); }
        isObserving = false; if (isConnected) updateUIState();
    }

    // --- Chat & TTS ---
    function addMessage(sender, text) { // (변경 없음)
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender.toLowerCase());
        messageElement.innerHTML = `<strong>${sender}:</strong> ${text}`; // Simple text assumed
        chatbox.appendChild(messageElement);
        chatbox.scrollTop = chatbox.scrollHeight; // Ensure scroll to bottom
    }
    function playTTS(audioUrl) { // (변경 없음)
        if (ttsAudio && audioUrl) {
            ttsAudio.src = audioUrl;
            ttsAudio.play().catch(e => { console.error("Audio playback error:", e); addMessage("System", "Audio playback failed. Check browser settings."); });
        }
    }

    // --- STT (Web Speech API) ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null; let isRecognizing = false;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = 'en-US'; // ** 영어로 설정 **
        recognition.interimResults = false; recognition.maxAlternatives = 1;

        recognition.onstart = () => { isRecognizing = true; sttStatus.textContent = "Listening..."; updateUIState(); };
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            sttStatus.textContent = `Recognized: ${transcript}`;
            textInput.value = transcript;
        };
        recognition.onerror = (event) => {
            console.error("STT Error:", event.error, event.message);
            let errorMsg = event.error;
            if (event.error === 'no-speech') errorMsg = "No speech detected.";
            else if (event.error === 'audio-capture') errorMsg = "Microphone error.";
            else if (event.error === 'not-allowed') errorMsg = "Microphone access denied.";
            else errorMsg = event.message || event.error;
             sttStatus.textContent = `STT Error: ${errorMsg}`;
        };
        recognition.onend = () => {
            isRecognizing = false;
            // 상태 메시지 업데이트 (오류 아니면 준비 상태로)
            if (!sttStatus.textContent.toLowerCase().includes("error")) {
                 sttStatus.textContent = "Speech recognition ready";
            }
            updateUIState(); // 버튼 상태 복구
            console.log("STT ended.");
        };
    } else {
        sttStatus.textContent = "Speech recognition not supported by this browser.";
        startSttButton.disabled = true; stopSttButton.disabled = true;
    }

    function startStt() { // (연결 확인 불필요)
        if (recognition && !isRecognizing) {
            try { recognition.start(); }
            catch(e) { console.error("STT start error:", e); sttStatus.textContent = "STT start error."; }
        }
    }
    function stopStt() { // (변경 없음)
        if (recognition && isRecognizing) {
             try { recognition.stop(); }
             catch(e) { console.error("STT stop error:", e); sttStatus.textContent = "STT stop error."; isRecognizing = false; updateUIState(); }
        }
    }

    // --- User Input ---
    async function sendUserInput() { // (async 유지)
        const text = textInput.value.trim();
        if (!isConnected) { // 연결 안되어 있으면 전송 불가
             addMessage("System", "Not connected. Please connect to the server first.");
             return;
        }
        if (text) {
            addMessage("User", text); await sendFrameAndText(text); textInput.value = "";
        } else {
             addMessage("System", "(Asking Aura to just observe the current scene...)");
             await sendFrameAndText(""); // 빈 텍스트는 관찰 요청
        }
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
    updateUIState(); // Set initial UI based on state variables
    addMessage("System", "Press 'Connect Server' to activate Aura.");
    window.addEventListener('beforeunload', () => { disconnectWebSocket(); });
});
"""
# *** JavaScript 수정 부분 끝 ***

# --- FastAPI App Setup (변경 없음) ---
app = fastapi.FastAPI()

# --- Helper Functions (Temperature=0.85, 이미지 처리 보강) ---
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
        # if removed_count > 0: print(f"Cleaned up {removed_count} old audio files.")
    except Exception: pass

async def call_ollama_gemma3(image_base64: str | None, text: str, history: list) -> str:
    """Ollama Gemma3 모델 API 호출 (generate, temp=0.85, 이미지 처리 개선)"""
    full_prompt = SYSTEM_CONTEXT
    for turn in history[-4:]: # 최근 2턴 유지
        role = turn.get('role', 'user'); content = turn.get('content', '')
        full_prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"

    user_content = text if text else "(Just observing the scene)" # 관찰 메시지 명확화
    full_prompt += f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
    full_prompt += "<start_of_turn>model\n" # 모델 응답 시작

    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_predict": 150, # 응답 길이 조금 더 증가
            "temperature": 0.85, # ** Temperature 0.85 **
            "stop": ["<end_of_turn>", "user:"]
        }
    }

    # 이미지 데이터 처리 및 검증 강화
    image_data_to_send = None
    if image_base64:
        try:
            # 데이터 URI 스키마 제거 (존재할 경우)
            if image_base64.startswith('data:image'):
                image_base64_data = image_base64.split(",")[1]
            else:
                image_base64_data = image_base64 # 이미 순수 데이터일 경우

            # 간단한 Base64 유효성 검사 (길이 및 문자) - 완벽하진 않음
            if len(image_base64_data) % 4 == 0 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in image_base64_data):
                 payload["images"] = [image_base64_data]
                 image_data_to_send = True # 이미지 포함 플래그
                 print("Image data included in payload.")
            else:
                 print("Warning: Invalid base64 image string detected. Sending text only.")
        except Exception as e:
             print(f"Warning: Error processing image data: {e}. Sending text only.")
    else:
         print("No image data provided for this request.")


    try:
        print(f"Sending to Ollama (Text: '{user_content[:30]}...', Image: {'Yes' if image_data_to_send else 'No'})")
        response = await asyncio.to_thread(
            requests.post, OLLAMA_API_URL, json=payload, timeout=90
        )
        response.raise_for_status()
        response_data = response.json()
        ai_response = response_data.get("response", "").strip()

        # 응답 후처리
        for stop_token in payload["options"]["stop"]:
             if stop_token in ai_response: ai_response = ai_response.split(stop_token)[0].strip()
        if ai_response.startswith("model\n"): ai_response = ai_response[len("model\n"):].strip()

        print(f"Ollama response: {ai_response}")
        # 히스토리 업데이트
        history.append({"role": "user", "content": user_content})
        history.append({"role": "model", "content": ai_response})

        return ai_response if ai_response else "(Aura didn't respond.)"

    except requests.exceptions.Timeout:
        print("Ollama API call timed out."); return "(Response took too long... Please try again.)"
    except requests.exceptions.RequestException as e:
        print(f"Ollama API request error: {e}"); return f"(Error communicating with Ollama: {e})"
    except Exception as e:
        print(f"Error processing Ollama response: {e}"); return "(An internal error occurred.)"


# --- WebSocket Connection Manager (변경 없음) ---
class ConnectionManager:
    def __init__(self): self.active_connections: list[WebSocket] = []; self.history: dict[WebSocket, list] = {}
    async def connect(self, websocket: WebSocket): await websocket.accept(); self.active_connections.append(websocket); self.history[websocket] = []; print(f"Client connected: {websocket.client}")
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections: self.active_connections.remove(websocket)
        if websocket in self.history: del self.history[websocket]
        print(f"Client disconnected: {websocket.client}")
    async def send_json(self, message: dict, websocket: WebSocket):
        if websocket in self.active_connections:
             try: await websocket.send_json(message)
             except Exception as e: print(f"Send failed: {e}"); self.disconnect(websocket)

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
            data = await websocket.receive_json()
            image_base64 = data.get("image") # Can be null
            user_text = data.get("text", "")
            current_history = manager.history.get(websocket, [])
            ai_response_text = await call_ollama_gemma3(image_base64, user_text, current_history)
            audio_url = await generate_tts(ai_response_text)
            response_payload = {"type": "response", "ai_text": ai_response_text, "audio_url": audio_url}
            await manager.send_json(response_payload, websocket)
    except WebSocketDisconnect: manager.disconnect(websocket); print(f"Client disconnected gracefully.")
    except Exception as e:
        print(f"WebSocket Error for {websocket.client}: {e}")
        error_payload = {"type": "error", "message": f"Server processing error."}
        await manager.send_json(error_payload, websocket); manager.disconnect(websocket)


# --- Uvicorn 실행 (변경 없음) ---
if __name__ == "__main__":
    print("Starting FastAPI server with Enhanced Glassmorphism UI & Persona...")
    print(f"TTS Voice: {TTS_VOICE}, Ollama Temp: 0.85")
    print(f"Audio files will be saved in: {AUDIO_DIR.resolve()}")
    print("Access the application at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # For development: uvicorn main:app --reload
