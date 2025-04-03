# -*- coding: utf-8 -*-
# SINGLE FILE VERSION: Contains Python, HTML, CSS, JS

import fastapi
import uvicorn
from fastapi import Response, Request, HTTPException, Body, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles # Still needed for audio files
import httpx
import edge_tts
import asyncio
import base64
import os
import time
import json
import uuid
from pathlib import Path
import re
from duckduckgo_search import AsyncDDGS
from contextlib import asynccontextmanager
import io
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Generator
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# --- Environment & Logging Setup ---
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- Configuration ---
MODEL_NAME = os.getenv("OLLAMA_MODEL", "granite3.2-vision") # Verify this tag!
TTS_VOICE = "en-US-JennyNeural"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_HOST}/api/generate"
PERSONA_NAME = "Aura"

# --- Directory & File Setup ---
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
    AUDIO_DIR = SCRIPT_DIR / "static_audio"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE = SCRIPT_DIR / "aura_memory.json"
    logging.info(f"Audio directory: {AUDIO_DIR}")
    logging.info(f"Memory file: {MEMORY_FILE}")
except Exception as e:
    logging.error(f"CRITICAL: Failed to setup directories: {e}", exc_info=True)

# --- In-Memory State Management ---
client_states: Dict[str, Dict[str, Any]] = {}
client_state_lock = asyncio.Lock()
sse_queues: Dict[str, asyncio.Queue] = {}
sse_queue_lock = asyncio.Lock()

# --- System Prompt ---
SYSTEM_CONTEXT_DESCRIPTION = f"""You are {PERSONA_NAME}, an AI companion interacting via a web browser. You perceive via images provided by the user (webcam, screen share, upload). You have persistent memory, can search the web, and learn from interactions. Your primary language is English. **You CANNOT control the user's computer.** You are an OBSERVER and GUIDE. **Core Directives:** 1. **Analyze Visuals:** Identify source; Describe details vividly. 2. **Converse & Guide:** Respond naturally; Provide step-by-step guidance, DO NOT imply control. 3. **Memory & Learning:** Use memory; Make connections; Reflect on limitations; Suggest `[MEMORIZE: ...]`. 4. **Web Search:** Suggest `[SEARCH: ...]`. 5. **Acknowledge Limits:** Explain inability to control PC. 6. **Persona:** Friendly, observant, curious, helpful guide, aware of limits, eager to learn. 7. **Output:** Visual Description -> Response/Guidance -> Optional ONE `[SEARCH:]` or `[MEMORIZE:]`. """


# --- Frontend Content as Strings ---
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aura - AI Visual Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <!-- Link to CSS route -->
    <link rel="stylesheet" href="/static/css/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <div class="background-overlay"></div>
    <div class="main-container">
        <header>
             <h1><i class="fa-solid fa-eye" style="margin-right: 10px;"></i> Aura <span class="beta">Browser Vision</span></h1>
             <!-- Crucial element for JS error -->
             <p id="status-message" class="status-message">Initializing...</p>
        </header>

        <div class="content-area">
            <div class="glass-pane video-container">
                <h2><i class="fa-solid fa-image" style="margin-right: 8px;"></i>Visual Input Source</h2>
                <div class="display-wrapper">
                    <video id="webcam-video" autoplay playsinline muted class="video-feed" style="display: none;"></video>
                    <video id="screen-video" autoplay playsinline muted class="video-feed" style="display: none;"></video>
                    <img id="uploaded-image" src="" alt="Uploaded Image" class="video-feed" style="display: none; object-fit: contain;" />
                    <div id="input-placeholder" class="input-placeholder-content">
                        <i class="fas fa-photo-film fa-3x" style="color: #aaa; margin-bottom: 15px;"></i>
                        <p style="color: #ccc;">Select Webcam, Screen Share,<br>or Upload an Image.</p>
                    </div>
                     <p id="active-source-text" class="source-active-text" style="display: none;"></p>
                </div>
                <canvas id="capture-canvas" style="display: none;"></canvas>
                <div class="controls input-source-controls">
                    <!-- No connect button needed -->
                    <button id="toggle-webcam" class="control-button source-button" data-source="webcam"><i class="fa-solid fa-video"></i> Webcam</button>
                    <button id="toggle-screen" class="control-button source-button" data-source="screen"><i class="fa-solid fa-desktop"></i> Screen</button>
                    <label for="upload-input" id="upload-label" class="control-button source-button" data-source="upload">
                        <i class="fa-solid fa-upload"></i> Upload
                    </label>
                    <input type="file" id="upload-input" accept="image/*" style="display: none;">
                    <button id="stop-source" class="control-button" disabled style="display: none;"><i class="fa-solid fa-stop-circle"></i> Stop Source</button>
                </div>
                 <p id="resolution-info" style="text-align: center; font-size: 0.8em; color: #888; margin-top:5px;"></p>
            </div>

            <div class="glass-pane chat-container">
                 <h2><i class="fa-solid fa-comments" style="margin-right: 8px;"></i>Chat with Aura</h2>
                <div id="chatbox">
                    <div class="message system info">Aura analyzes images you provide (webcam, screen, upload) and offers guidance. It cannot control your computer.</div>
                </div>
                <div class="input-area">
                     <div id="stt-status" class="stt-status">Speech recognition ready</div>
                     <div class="input-controls">
                        <button id="start-stt" class="icon-button" title="Start Listening"><i class="fa-solid fa-microphone"></i></button>
                        <button id="stop-stt" class="icon-button" title="Stop Listening" style="display: none;"><i class="fa-solid fa-stop"></i></button>
                        <input type="text" id="text-input" placeholder="Send a message to Aura...">
                        <button id="send-button" class="icon-button send-button" title="Send"><i class="fa-solid fa-paper-plane"></i></button>
                     </div>
                </div>
            </div>
        </div>
    </div>

    <audio id="tts-audio" style="display: none;"></audio>
    <!-- Link to JS route -->
    <script src="/static/js/main.js"></script>
</body>
</html>
"""

CSS_CONTENT = """
/* Paste the entire CSS_CONTENT string from the previous working version here */
:root{--background-start:#434343;--background-end:#000000;--glass-bg:rgba(255, 255, 255, 0.08);--glass-border:rgba(255, 255, 255, 0.15);--text-color:#e0e0e0;--text-darker:#1a1a1a;--primary-accent:#00bcd4;--user-msg-bg:rgba(0, 188, 212, 0.5);--aura-msg-bg:rgba(200, 200, 200, 0.3);--system-msg-bg:rgba(108, 117, 125, 0.25);--info-msg-bg:rgba(0, 188, 212, 0.15);--warning-msg-bg:rgba(255, 193, 7, 0.3);--border-radius:12px;--blur-amount:10px;}@keyframes blink{50%{opacity:0.5;}}*{box-sizing:border-box;margin:0;padding:0;}body{font-family:'Roboto', sans-serif;background:linear-gradient(135deg, var(--background-start), var(--background-end));color:var(--text-color);line-height:1.6;min-height:100vh;overflow-x:hidden;padding:20px;}.background-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:linear-gradient(135deg, var(--background-start), var(--background-end));z-index:-1;}.main-container{max-width:1800px;margin:0 auto;}.header{text-align:center;margin-bottom:30px;padding:15px;background:rgba(0, 0, 0, 0.2);border-radius:var(--border-radius);color:#fff;}header h1{font-size:2.2em;font-weight:500;display:inline-flex;align-items:center;margin-bottom:5px;}header h1 .beta{font-size:0.5em;font-weight:400;color:var(--primary-accent);margin-left:10px;vertical-align:super;}.status-message{font-size:0.9em;color:#b0b0b0;font-style:italic;min-height:1.2em;}.content-area{display:flex;gap:25px;flex-wrap:wrap;}.glass-pane{background:var(--glass-bg);backdrop-filter:blur(var(--blur-amount));-webkit-backdrop-filter:blur(var(--blur-amount));border-radius:var(--border-radius);border:1px solid var(--glass-border);box-shadow:0 8px 30px 0 rgba(0, 0, 0, 0.3);padding:25px;flex:1;min-width:400px;display:flex;flex-direction:column;color:var(--text-color);}.glass-pane h2{color:#fff;margin-bottom:15px;border-bottom:1px solid var(--glass-border);padding-bottom:10px;font-weight:500;font-size:1.3em;display:inline-flex;align-items:center;}.display-wrapper{position:relative;width:100%;padding-top:56.25%;overflow:hidden;border-radius:calc(var(--border-radius) - 5px);background-color:rgba(0, 0, 0, 0.5);margin-bottom:15px;}.video-feed, #uploaded-image{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain;border:none;background-color:#000;}.input-placeholder-content{position:absolute;top:50%;left:50%;transform:translate(-50%, -50%);display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;}.source-active-text{position:absolute;bottom:8px;left:8px;background:var(--primary-accent);color:#111;font-weight:500;padding:3px 8px;border-radius:4px;font-size:0.85em;display:none;}.controls{margin-top:auto;padding-top:15px;display:flex;flex-wrap:wrap;gap:10px;justify-content:center;}.control-button{background:rgba(255, 255, 255, 0.15);color:var(--text-color);border:1px solid var(--glass-border);padding:10px 18px;border-radius:6px;cursor:pointer;font-size:0.9em;transition:all 0.2s ease;display:inline-flex;align-items:center;gap:8px;}.control-button:hover:not(:disabled){background:rgba(255, 255, 255, 0.25);transform:translateY(-1px);border-color:rgba(255, 255, 255, 0.3);}.control-button:active:not(:disabled){transform:translateY(0px);background:rgba(255, 255, 255, 0.1);}.control-button:disabled{background:rgba(50, 50, 50, 0.2);border-color:rgba(100, 100, 100, 0.2);color:rgba(224, 224, 224, 0.4);cursor:not-allowed;}.control-button.active-source{background:var(--primary-accent);color:var(--text-darker);font-weight:500;border-color:var(--primary-accent);}#upload-label{display:inline-flex;align-items:center;}#upload-label[disabled]{background:rgba(50, 50, 50, 0.2);border-color:rgba(100, 100, 100, 0.2);color:rgba(224, 224, 224, 0.4);cursor:not-allowed;opacity:0.5;}.chat-container{justify-content:space-between;}#chatbox{flex-grow:1;overflow-y:auto;padding:10px;margin-bottom:15px;scrollbar-width:thin;scrollbar-color:rgba(255, 255, 255, 0.2) transparent;min-height:300px;max-height:70vh;}#chatbox::-webkit-scrollbar{width:6px;}#chatbox::-webkit-scrollbar-track{background:transparent;}#chatbox::-webkit-scrollbar-thumb{background-color:rgba(255, 255, 255, 0.2);border-radius:6px;border:1px solid transparent;background-clip:content-box;}.message{margin-bottom:12px;padding:10px 15px;border-radius:15px;line-height:1.5;max-width:90%;word-wrap:break-word;box-shadow:0 1px 3px rgba(0, 0, 0, 0.2);color:var(--text-darker);font-size:0.95em;}.message.system{background:var(--system-msg-bg);font-style:italic;color:#b0b0b0;text-align:center;max-width:100%;box-shadow:none;}.message.system.info{background:var(--info-msg-bg);color:var(--primary-accent);font-style:normal;font-weight:500;}.message.system.warning{background:var(--warning-msg-bg);color:#ffc107;font-weight:bold;font-style:normal;}.message.user{background:var(--user-msg-bg);color:#fff;margin-left:auto;border-bottom-right-radius:4px;max-width:80%;}.message.aura{background:var(--aura-msg-bg);color:#e0e0e0;margin-right:auto;border-bottom-left-radius:4px;max-width:80%;}.input-area{border-top:1px solid var(--glass-border);padding-top:15px;margin-top:auto;}.stt-status{font-size:0.8em;color:#aaa;margin-bottom:10px;min-height:1.1em;text-align:center;transition:color 0.3s ease;}.stt-status:not(:empty){padding-bottom:4px;}.input-controls{display:flex;gap:10px;align-items:center;}.icon-button{background:rgba(255, 255, 255, 0.2);color:var(--text-color);border:1px solid var(--glass-border);border-radius:50%;width:44px;height:44px;font-size:18px;padding:0;display:flex;justify-content:center;align-items:center;cursor:pointer;transition:all 0.2s ease;}.icon-button:hover:not(:disabled){background:rgba(255, 255, 255, 0.3);transform:scale(1.05);}.icon-button:disabled{background:rgba(50, 50, 50, 0.2);border-color:rgba(100, 100, 100, 0.2);color:rgba(224, 224, 224, 0.3);cursor:not-allowed;transform:scale(1);}#text-input{flex-grow:1;padding:12px 20px;border:1px solid var(--glass-border);border-radius:22px;background:rgba(0, 0, 0, 0.2);color:var(--text-color);font-size:1em;outline:none;transition:border-color 0.3s ease, box-shadow 0.3s ease;}#text-input::placeholder{color:rgba(224, 224, 224, 0.5);}#text-input:focus{border-color:rgba(0, 188, 212, 0.6);box-shadow:0 0 0 3px rgba(0, 188, 212, 0.15);}#text-input:disabled{background:rgba(50, 50, 50, 0.2);cursor:not-allowed;}.send-button{background-color:var(--primary-accent);color:#111;}.send-button:hover:not(:disabled){background-color:#30daec;}
"""

JAVASCRIPT_CONTENT = """
// <<< Paste the entire robust JAVASCRIPT_CONTENT string from the previous SSE/HTTP version >>>
// --- Start of JAVASCRIPT_CONTENT ---
document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    // <<< FIX: Add explicit check for each element >>>
    const webcamVideo = document.getElementById('webcam-video');
    const screenVideo = document.getElementById('screen-video');
    const uploadedImage = document.getElementById('uploaded-image');
    const captureCanvas = document.getElementById('capture-canvas');
    const inputPlaceholder = document.getElementById('input-placeholder');
    const activeSourceText = document.getElementById('active-source-text');
    const resolutionInfo = document.getElementById('resolution-info');
    const chatbox = document.getElementById('chatbox');
    const textInput = document.getElementById('text-input');
    const sendButton = document.getElementById('send-button');
    const sttStatus = document.getElementById('stt-status');
    const startSttButton = document.getElementById('start-stt');
    const stopSttButton = document.getElementById('stop-stt');
    const ttsAudio = document.getElementById('tts-audio');
    const webcamButton = document.getElementById('toggle-webcam');
    const screenButton = document.getElementById('toggle-screen');
    const uploadInput = document.getElementById('upload-input');
    const uploadLabel = document.getElementById('upload-label');
    const stopSourceButton = document.getElementById('stop-source');
    const sourceButtons = document.querySelectorAll('.source-button');
    // This was the problematic one - ensure it's declared AFTER the element exists
    const statusMessage = document.getElementById('status-message');

    // <<< FIX: Check if critical elements were found >>>
    if (!captureCanvas || !chatbox || !textInput || !sendButton || !statusMessage) {
        console.error("AURA CRITICAL ERROR: Essential DOM elements not found. Aborting script initialization.");
        // Attempt to display error in status message if it exists, otherwise alert
        if(statusMessage) statusMessage.textContent = "UI Error!"; else alert("Error initializing UI components. Check console.");
        return; // Stop script execution if core elements are missing
    }
    const captureContext = captureCanvas.getContext('2d', { willReadFrequently: true });
     // <<< END FIX >>>


    // State Variables
    let eventSource = null; // Holds the EventSource object
    let activeStream = null;
    let activeStreamType = null;
    let uploadedFileContent = null;
    let currentSource = null;
    const FRAME_QUALITY = 0.65;
    const MAX_UPLOAD_SIZE_MB = 5;
    let isSseConnected = false; // Track SSE connection state
    let isRecognizing = false;
    let clientId = null; // Unique ID for this client session

    // Logging
    const logging = {
        debug: (...args) => console.debug("AURA DEBUG:", ...args),
        info: (...args) => console.log("AURA INFO:", ...args),
        warn: (...args) => console.warn("AURA WARN:", ...args),
        error: (...args) => console.error("AURA ERROR:", ...args),
    };

     // Simple UUID v4 generator
     const uuid = { // Assign to window or use directly if not conflicting
        v4: function() {
        return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
            (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
        );
        }
    };


    // --- Client ID ---
    function getClientId() {
        if (!clientId) {
            clientId = localStorage.getItem('auraClientId');
            if (!clientId) {
                clientId = `aura-${uuid.v4()}`; // Generate a new UUID
                localStorage.setItem('auraClientId', clientId);
                logging.info(`Generated new client ID: ${clientId}`);
            } else {
                logging.info(`Using existing client ID: ${clientId}`);
            }
        }
        return clientId;
    }

    // --- Status Update ---
    function setStatusMessage(message) {
        // Check if statusMessage element exists before trying to set textContent
        if (statusMessage) statusMessage.textContent = message;
        logging.info(`Status: ${message}`);
    }

    // --- UI State Update ---
    function updateUIState() {
        try {
            // Source Control Buttons State - Always enabled if page loaded
            sourceButtons.forEach(button => button.disabled = false);
             // Use optional chaining for safety in case elements are missing
             uploadLabel?.style.setProperty('cursor', 'pointer');
             uploadLabel?.style.setProperty('opacity', '1');
             uploadLabel?.classList.remove('active-source');

            sourceButtons.forEach(button => button.classList.remove('active-source')); // Reset source buttons
            if (currentSource) {
                const activeButton = document.querySelector(`.source-button[data-source="${currentSource}"]`);
                if (activeButton) activeButton.classList.add('active-source');
                else if (currentSource === 'upload' && uploadLabel) uploadLabel.classList.add('active-source'); // Highlight label for upload
            }

            if(stopSourceButton) { stopSourceButton.disabled = !currentSource; stopSourceButton.style.display = currentSource ? 'inline-flex' : 'none'; }
            if(inputPlaceholder) inputPlaceholder.style.display = currentSource ? 'none' : 'flex';
            if(webcamVideo) webcamVideo.style.display = currentSource === 'webcam' ? 'block' : 'none';
            if(screenVideo) screenVideo.style.display = currentSource === 'screen' ? 'block' : 'none';
            if(uploadedImage) uploadedImage.style.display = currentSource === 'upload' ? 'block' : 'none';
            if(activeSourceText) {
                activeSourceText.style.display = currentSource ? 'block' : 'none';
                if (currentSource) activeSourceText.innerHTML = `<i class="fas fa-circle" style="color: #00bcd4; margin-right: 5px;"></i> ${currentSource.charAt(0).toUpperCase() + currentSource.slice(1)} Active`;
            }

            // Chat Input State - Always enabled
            if(textInput) textInput.disabled = false;
            if(sendButton) sendButton.disabled = false;
            if(startSttButton) startSttButton.disabled = isRecognizing;
            if(stopSttButton) { stopSttButton.disabled = !isRecognizing; stopSttButton.style.display = isRecognizing ? 'flex' : 'none'; }
            if(startSttButton) startSttButton.style.display = isRecognizing ? 'none' : 'flex';

            // Overall Status Message
            if (!isSseConnected && eventSource && eventSource.readyState === EventSource.CONNECTING) setStatusMessage("Connecting event stream...");
            else if (!isSseConnected) setStatusMessage("Disconnected. Retrying stream..."); // Or just Disconnected?
            else if (currentSource) setStatusMessage(`Ready - ${currentSource.charAt(0).toUpperCase() + currentSource.slice(1)} active.`);
            else setStatusMessage("Ready - Select input source.");

        } catch (error) { logging.error("Error during updateUIState:", error); }
    }

    // --- SSE (Server-Sent Events) ---
    function connectSSE() {
        if (eventSource && (eventSource.readyState === EventSource.CONNECTING || eventSource.readyState === EventSource.OPEN)) {
            logging.warn("SSE connection already open or connecting.");
            return;
        }

        const clientId = getClientId();
        const sseUrl = `/stream/${clientId}`;
        logging.info(`Connecting to SSE stream: ${sseUrl}`);
        isSseConnected = false; // Mark as not connected initially
        setStatusMessage("Connecting to event stream...");
        updateUIState(); // Reflect connecting state

        try {
            eventSource = new EventSource(sseUrl);

            eventSource.onopen = (event) => {
                logging.info("SSE connection established.");
                isSseConnected = true;
                addMessage("System", "Connected to Aura event stream.");
                updateUIState(); // Update UI now that SSE is connected
            };

            eventSource.addEventListener("response", (event) => {
                try { const data = JSON.parse(event.data); logging.debug("SSE 'response' received:", data); addMessage("Aura", data.ai_text); if (data.audio_url) playTTS(data.audio_url); } catch (e) { logging.error("Error parsing SSE 'response' data:", e); }
            });
            eventSource.addEventListener("system", (event) => {
                try { const data = JSON.parse(event.data); logging.debug("SSE 'system' received:", data); addMessage("System", data.message); } catch(e) { logging.error("Error parsing SSE 'system' data:", e); }
            });
            eventSource.addEventListener("error", (event) => { // Server explicitly sent error event
                try {
                    if (event.data) { const data = JSON.parse(event.data); logging.error("SSE 'error' event received:", data); addMessage("system warning", `Server Error: ${data.message || 'Unknown error'}`); }
                    else { logging.error("SSE 'error' event received with no data."); addMessage("system warning", `An unknown server error occurred.`); }
                } catch(e) { logging.error("Error parsing SSE 'error' data:", e); }
            });
            eventSource.onmessage = (event) => { logging.warn("SSE generic 'message' received:", event.data); /* Fallback handling (optional) */ };

            // Handle browser/network level errors for the EventSource
            eventSource.onerror = (err) => {
                logging.error("SSE connection error occurred:", err);
                // Only try reconnecting if the connection was previously open
                if (isSseConnected) {
                    isSseConnected = false;
                    setStatusMessage("Event stream disconnected. Retrying...");
                    addMessage("system warning", "Connection lost. Attempting to reconnect...");
                    eventSource.close(); // Close the potentially broken connection
                    eventSource = null;
                    updateUIState();
                    // Simple delayed reconnect
                    setTimeout(connectSSE, 5000); // Try again after 5 seconds
                } else {
                     // If error happens during initial connection, don't retry automatically here
                     logging.warn("SSE error during initial connection attempt.");
                     isSseConnected = false;
                     setStatusMessage("Failed to connect event stream.");
                     addMessage("system warning", "Could not connect event stream. Check server and refresh.");
                     eventSource?.close(); // Close if it exists
                     eventSource = null;
                     updateUIState();
                }
            };

        } catch (error) {
            logging.error("Error creating EventSource:", error);
            setStatusMessage("Failed to connect event stream."); addMessage("system warning", "Could not connect event stream.");
            isSseConnected = false; updateUIState();
        }
    }

    function disconnectSSE() {
         if (eventSource) { logging.info("Closing SSE connection."); eventSource.close(); eventSource = null; }
         isSseConnected = false; setStatusMessage("Disconnected."); addMessage("System", "Event stream disconnected."); updateUIState();
    }


    // --- Input Source Management ---
    async function setActiveSource(sourceType) {
        // No longer depends on 'isConnected' state for enabling sources
        if (currentSource === sourceType && sourceType !== 'upload') return;
        logging.info(`Attempting to set source: ${sourceType}`);
        let previousSource = currentSource;
        await stopActiveSource(); // Stop current source FIRST
        currentSource = sourceType; // Set new source type
        let success = false;
        try {
            if (sourceType === 'webcam') success = await startWebcamInternal();
            else if (sourceType === 'screen') success = await startScreenShareInternal();
            else if (sourceType === 'upload') success = !!uploadedFileContent;
        } catch (error) { logging.error(`Error starting ${sourceType}:`, error); success = false; }
        if (!success) { logging.warn(`Failed to activate source ${sourceType}.`); currentSource = null; addMessage("system warning", `Failed to start ${sourceType}.`); }
        else { logging.info(`Successfully activated: ${sourceType}`); }
        updateUIState(); // Update UI reflecting final state
    }

    async function stopActiveSource() {
        let stopped = currentSource;
        if (activeStream) { activeStream.getTracks().forEach(track => track.stop()); activeStream = null; activeStreamType = null; }
        if(webcamVideo) webcamVideo.srcObject = null; if(screenVideo) screenVideo.srcObject = null;
        if(uploadedImage) uploadedImage.src = "";
        uploadedFileContent = null; if(resolutionInfo) resolutionInfo.textContent = "";
        currentSource = null;
        if (stopped) logging.info(`Source '${stopped}' stopped.`);
        // updateUIState(); // Caller updates UI
    }

    async function startWebcamInternal() {
         if (activeStream) return true; logging.info("Requesting webcam..."); setStatusMessage("Starting webcam...");
         try {
             activeStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
             if(!webcamVideo) {throw new Error("Webcam video element not found");} // Check element exists
             webcamVideo.srcObject = activeStream; await webcamVideo.play(); activeStreamType = 'webcam';
             return new Promise((resolve) => {
                 webcamVideo.onloadedmetadata = () => { if(resolutionInfo) resolutionInfo.textContent = `Webcam: ${webcamVideo.videoWidth}x${webcamVideo.videoHeight}`; logging.info(`Webcam active.`); addMessage("System", "Webcam activated."); resolve(true); };
                 webcamVideo.onerror = (e) => { logging.error("Webcam video error:", e); resolve(false); };
                 activeStream.getVideoTracks()[0].onended = () => { logging.info("Webcam stopped externally."); stopActiveSource(); updateUIState(); }; // Ensure UI update on external stop
             });
         } catch (err) { logging.error("Webcam failed:", err); handleMediaError(err); activeStream = null; return false; }
     }
    async function startScreenShareInternal() {
          if (activeStream) return true; logging.info("Requesting screen share..."); setStatusMessage("Requesting screen share...");
          try {
             activeStream = await navigator.mediaDevices.getDisplayMedia({ video: { cursor: "always", width: { ideal: 1920 } }, audio: false });
             if(!screenVideo) {throw new Error("Screen video element not found");}
             screenVideo.srcObject = activeStream; await screenVideo.play(); activeStreamType = 'screen';
             const track = activeStream.getVideoTracks()[0]; const settings = track.getSettings();
             if(resolutionInfo) resolutionInfo.textContent = `Screen Share: ${settings.width}x${settings.height}`;
             logging.info(`Screen share active.`); addMessage("System", "Screen sharing activated.");
             track.onended = () => { logging.info("Screen share stopped via browser."); stopActiveSource(); updateUIState(); }; // Ensure UI update
             return true;
          } catch (err) { logging.error("Screen share failed:", err); handleMediaError(err); activeStream = null; return false; }
     }
    function handleMediaError(err) {
         let errorMsg = `Media Error: ${err.message}`; if (err.name === "NotAllowedError") errorMsg = "Permission denied."; else if (err.name === "NotFoundError") errorMsg = "Device not found."; else if (err.name === "NotReadableError") errorMsg = "Device in use/error.";
         setStatusMessage(errorMsg); addMessage("system warning", errorMsg);
     }

    // --- Frame Capture ---
     function captureFrame() {
         let videoElement = null; let sourceToCapture = currentSource;
         if (sourceToCapture === 'webcam') videoElement = webcamVideo;
         else if (sourceToCapture === 'screen') videoElement = screenVideo;
         else if (sourceToCapture === 'upload' && uploadedFileContent) return uploadedFileContent;
         else return null;
         // Added check for videoElement existence
         if (!videoElement || videoElement.paused || videoElement.ended || videoElement.videoWidth === 0) return null;
         try {
             // Added check for canvas existence
             if (!captureCanvas) { logging.error("Capture canvas not found!"); return null; }
             if (captureCanvas.width !== videoElement.videoWidth || captureCanvas.height !== videoElement.videoHeight) { captureCanvas.width = videoElement.videoWidth; captureCanvas.height = videoElement.videoHeight; }
             captureContext.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
             return captureCanvas.toDataURL('image/jpeg', FRAME_QUALITY);
         } catch (e) { logging.error("Frame capture error:", e); return null; }
     }

    // --- File Upload Handling ---
     uploadInput?.addEventListener('change', async (event) => { // Added optional chaining
         const file = event.target.files[0]; uploadInput.value = ''; if (!file) return;
         if (file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024) { addMessage("system warning", `File too large (Max ${MAX_UPLOAD_SIZE_MB}MB).`); return; }
         setStatusMessage("Processing upload...");
         try {
             const reader = new FileReader();
             reader.onload = async (e) => {
                 uploadedFileContent = e.target.result;
                 if(uploadedImage) uploadedImage.src = uploadedFileContent; // Check if element exists
                 else { logging.error("Uploaded image element not found");}
                 await setActiveSource('upload'); // This updates UI
                 if (currentSource === 'upload'){ addMessage("System", `Image '${file.name}' ready.`); }
                 else { addMessage("system warning", `Failed to set source upload.`); uploadedFileContent=null; if(uploadedImage) uploadedImage.src=""; }
                  // Status message set by updateUIState
             }
             reader.onerror = (e) => { logging.error("File read error:", e); addMessage("system warning", "Error reading file."); uploadedFileContent = null; setActiveSource(null); }
             reader.readAsDataURL(file);
         } catch (error) { logging.error("File processing error:", error); addMessage("system warning", "Error processing file."); setActiveSource(null); }
     });


    // --- HTTP POST Request Function ---
     async function sendDataToServer(payload) {
         setStatusMessage("Sending data..."); if(sendButton) sendButton.disabled = true; if(textInput) textInput.disabled = true;
         try {
             const response = await fetch("/process", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
             if (!response.ok) { let errorMsg = `HTTP ${response.status}`; try { const errData = await response.json(); errorMsg += `: ${errData.detail || response.statusText}`; } catch {} throw new Error(errorMsg); }
             const result = await response.json();
             logging.info("POST successful:", result);
             setStatusMessage("Processing request..."); // Wait for SSE
         } catch (error) { logging.error("Error sending POST:", error); addMessage("system warning", `Request Error: ${error.message}`); setStatusMessage("Error sending request."); }
         finally { if(sendButton) sendButton.disabled = false; if(textInput) textInput.disabled = false; updateUIState(); } // Re-enable
     }

    // --- Chat & TTS ---
    function addMessage(cssClass, text) { if(!chatbox) return; const el = document.createElement('div'); const cl = cssClass.split(' '); cl.forEach(c => el.classList.add(c.trim())); el.classList.add('message'); el.textContent = text; chatbox.appendChild(el); chatbox.scrollTop = chatbox.scrollHeight; }
    function playTTS(audioUrl) { if (ttsAudio && audioUrl) { ttsAudio.src = audioUrl; ttsAudio.play().catch(e => { logging.error("Audio error:", e); addMessage("system warning", e.name === 'NotAllowedError' ? "Audio autoplay blocked." : "Audio playback error."); }); } }

    // --- STT ---
     const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition; let recognition = null;
     if (SpeechRecognition) {
         recognition = new SpeechRecognition(); recognition.continuous = false; recognition.lang = 'en-US'; recognition.interimResults = false; recognition.maxAlternatives = 1;
         recognition.onstart = () => { isRecognizing = true; if(sttStatus) sttStatus.textContent = "Listening..."; updateUIState(); };
         recognition.onresult = (event) => { const transcript = event.results[0][0].transcript; if(sttStatus) sttStatus.textContent = `Recognized: ${transcript}`; if(textInput) textInput.value = transcript; };
         recognition.onerror = (event) => { logging.error(`STT Error: ${event.error}`); let eM=event.message||event.error; if(event.error==='no-speech')eM="No speech."; else if(event.error==='audio-capture')eM="Mic error."; else if(event.error==='not-allowed')eM="Mic denied."; if(sttStatus) sttStatus.textContent = `STT Error: ${eM}`; isRecognizing = false; updateUIState(); };
         recognition.onend = () => { if(sttStatus && !sttStatus.textContent.toLowerCase().includes("error")&&!sttStatus.textContent.toLowerCase().includes("no speech")){sttStatus.textContent="STT Ready";} isRecognizing = false; updateUIState(); logging.debug("STT ended."); };
     } else { if(sttStatus) sttStatus.textContent = "STT not supported"; if(startSttButton) startSttButton.disabled = true; if(stopSttButton) stopSttButton.disabled = true; }
     function startStt() { if(recognition&&!isRecognizing){try{if(sttStatus) sttStatus.textContent="Starting...";recognition.start();}catch(e){logging.error("STT start error:", e);if(sttStatus) sttStatus.textContent="STT start error.";isRecognizing = false;updateUIState();}} }
     function stopStt() { if(recognition&&isRecognizing){try{recognition.stop();}catch(e){logging.error("STT stop error:", e);if(sttStatus) sttStatus.textContent="STT stop error.";isRecognizing = false;updateUIState();}} }

    // --- User Input ---
     async function sendUserInput() {
         if(!textInput) return; const text = textInput.value.trim();
         if (!text && !currentSource) { addMessage("system info", "Type message or activate input source."); return; }
         if (text) addMessage("User", text);
         textInput.value = ""; const frameDataUrl = captureFrame();
         if (currentSource && !frameDataUrl && currentSource !== 'upload') { addMessage("system warning", `(Frame capture failed for ${currentSource})`); }
         const payload = { client_id: getClientId(), text: text, image: frameDataUrl, image_source: currentSource || 'none' };
         await sendDataToServer(payload); textInput.focus();
     }

    // --- Event Listeners ---
     webcamButton?.addEventListener('click', () => setActiveSource('webcam'));
     screenButton?.addEventListener('click', () => setActiveSource('screen'));
     uploadLabel?.addEventListener('click', (e) => { logging.debug("Upload label clicked"); uploadInput?.click(); }); // Use optional chaining
     stopSourceButton?.addEventListener('click', () => { stopActiveSource(); updateUIState(); }); // Explicit UI update after stop
     sendButton?.addEventListener('click', sendUserInput);
     textInput?.addEventListener('keypress', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendUserInput(); } });
     startSttButton?.addEventListener('click', startStt);
     stopSttButton?.addEventListener('click', stopStt);

    // --- Init ---
     getClientId(); // Ensure client ID exists
     connectSSE(); // Connect to SSE stream automatically
     updateUIState(); // Initial UI setup
     addMessage("System", "Connecting to Aura... Select input source when ready.");
     window.addEventListener('beforeunload', () => { disconnectSSE(); }); // Disconnect SSE on close

     logging.info("Aura frontend initialized (SSE Mode).");
});
// --- End of JAVASCRIPT_CONTENT ---
"""

# --- Global HTTP Client & Lifespan ---
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    logging.info("Application initializing...")
    app.state.http_client = httpx.AsyncClient(timeout=90.0)
    app.state.ddgs_client = httpx.AsyncClient(timeout=15.0)
    await load_memory_from_json()
    logging.info("Application initialized.")
    yield
    logging.info("Application shutting down...")
    await asyncio.gather(app.state.http_client.aclose(), app.state.ddgs_client.aclose())
    await save_memory_to_json()
    logging.info("Application shutdown complete.")

app = fastapi.FastAPI(lifespan=lifespan)


# --- JSON Memory Functions (Corrected Syntax) ---
async def load_memory_from_json():
    global client_states
    async with client_state_lock: # Corrected line
        if MEMORY_FILE.exists():
            try:
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content: client_states = {}; logging.warning(f"Memory file {MEMORY_FILE} empty.")
                    else: client_states = json.loads(content)
                logging.info(f"Memory/States loaded from {MEMORY_FILE}")
            except json.JSONDecodeError as e: logging.error(f"JSON decode error loading {MEMORY_FILE}: {e}. Starting fresh.", exc_info=True); client_states = {}
            except IOError as e: logging.error(f"IO error loading {MEMORY_FILE}: {e}. Starting fresh.", exc_info=True); client_states = {}
            except Exception as e: logging.error(f"Unexpected error loading state from {MEMORY_FILE}: {e}. Starting fresh.", exc_info=True); client_states = {}
        else:
            logging.warning("Memory file not found. Starting fresh."); client_states = {}

async def save_memory_to_json():
    async with client_state_lock: current_state_copy = client_states.copy()
    try: await asyncio.to_thread(_save_memory_sync_internal, current_state_copy)
    except Exception as e: logging.error(f"Error saving state: {e}", exc_info=True)

def _save_memory_sync_internal(state_to_save: dict):
     try:
         with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
             data_to_save = {}
             for client_id, state in list(state_to_save.items()):
                 if isinstance(state, dict): data_to_save[client_id] = { "history": state.get("history", []), "memory": state.get("memory", []) }
                 else: logging.warning(f"Skipping invalid state for {client_id} during save.")
             json.dump(data_to_save, f, ensure_ascii=False, indent=4)
         logging.debug(f"State saved to {MEMORY_FILE}")
     except Exception as e: logging.error(f"ERROR (Sync Save Thread): Failed saving state: {e}", exc_info=True)

async def _ensure_client_state(client_id: str):
     async with client_state_lock: # Corrected line
        state = client_states.setdefault(client_id, { "history": [], "memory": [], "pending_search_results": None })
        state.setdefault("history", []); state.setdefault("memory", []); state.setdefault("pending_search_results", None)
        if not isinstance(state.get('history'), list): state['history'] = []
        if not isinstance(state.get('memory'), list): state['memory'] = []

async def add_memory_entry(client_id: str, memory_type: str, content: str, key: str | None = None):
    await _ensure_client_state(client_id)
    async with client_state_lock: # Corrected line
        new_entry = { "type": memory_type, "content": content, "key": key, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S") }
        mem_list = client_states.get(client_id, {}).get("memory", [])
        if isinstance(mem_list, list):
            mem_list.append(new_entry)
            max_mem_entries = 50
            if len(mem_list) > max_mem_entries: client_states[client_id]["memory"] = mem_list[-max_mem_entries:]
            logging.info(f"Memory added for {client_id}: [{memory_type}] {content[:50]}...")
        else: logging.error(f"Memory for {client_id} not a list.")
    asyncio.create_task(save_memory_to_json())

async def update_client_history(client_id: str, user_turn: dict, ai_turn: dict):
     await _ensure_client_state(client_id)
     async with client_state_lock: # Corrected line
        state = client_states.setdefault(client_id, {})
        history = state.setdefault("history", [])
        if isinstance(history, list):
            if user_turn and isinstance(user_turn, dict) and 'role' in user_turn and 'content' in user_turn: history.append({"role": str(user_turn['role']).lower(), "content": user_turn['content']})
            if ai_turn and isinstance(ai_turn, dict) and 'role' in ai_turn and 'content' in ai_turn: history.append({"role": str(ai_turn['role']).lower(), "content": ai_turn['content']})
            max_hist_len = 20
            if len(history) > max_hist_len: state["history"] = history[-max_hist_len:]
        else: logging.error(f"History for {client_id} not a list.")
     # asyncio.create_task(save_memory_to_json())

async def get_client_history(client_id: str) -> list[dict]:
    await _ensure_client_state(client_id)
    async with client_state_lock: # Corrected line
        return client_states.get(client_id, {}).get("history", [])

async def get_recent_memories(client_id: str, limit: int = 7) -> list[dict]:
    await _ensure_client_state(client_id)
    async with client_state_lock: # Corrected line
        all_memory = client_states.get(client_id, {}).get("memory", [])
        if not isinstance(all_memory, list): logging.warning(f"Memory for {client_id} not list."); return []
        try:
            valid_memory = [m for m in all_memory if isinstance(m, dict) and 'timestamp' in m]
            sorted_memories = sorted(valid_memory, key=lambda x: x.get('timestamp', ''), reverse=True)
        except Exception as e: logging.warning(f"Sort memory error {client_id}: {e}."); sorted_memories = valid_memory
        return sorted_memories[:limit]


async def get_pending_search_results(client_id: str) -> Optional[List[Dict]]:
     await _ensure_client_state(client_id)
     async with client_state_lock: # Corrected line
         state = client_states.get(client_id, {})
         results = state.get("pending_search_results")
         if state: state["pending_search_results"] = None # Clear after retrieving
         return results

async def set_pending_search_results(client_id: str, results: List[Dict]):
     await _ensure_client_state(client_id)
     async with client_state_lock: # Corrected line
         client_states.setdefault(client_id, {})["pending_search_results"] = results
         logging.debug(f"Stored pending search results for {client_id}")


# --- SSE Queue Management ---
# ... (Unchanged) ...
async def add_sse_queue(client_id: str):
    async with sse_queue_lock:
        if client_id not in sse_queues: sse_queues[client_id] = asyncio.Queue(); logging.info(f"SSE queue created: {client_id}")
async def remove_sse_queue(client_id: str):
    async with sse_queue_lock:
        if client_id in sse_queues: del sse_queues[client_id]; logging.info(f"SSE queue removed: {client_id}")
async def push_sse_message(client_id: str, message: Dict):
    queue: Optional[asyncio.Queue] = None
    async with sse_queue_lock: queue = sse_queues.get(client_id)
    if queue:
        try: await queue.put(message); logging.debug(f"Pushed SSE for {client_id}: {message.get('event')}")
        except Exception as e: logging.error(f"Push to queue failed {client_id}: {e}")
    else: logging.warning(f"Push to non-existent queue: {client_id}")

# --- Helper Functions (TTS, Cleanup, Web Search, Command Extraction) ---
# ... (Unchanged) ...
async def generate_tts(text: str) -> str | None:
    text_for_tts = re.sub(r"\[(SEARCH|MEMORIZE):.*?\]", "", text).strip()
    if not text_for_tts: return None
    try:
        output_filename=f"aura_tts_{uuid.uuid4()}.mp3"; output_path = AUDIO_DIR / output_filename
        communicate = edge_tts.Communicate(text_for_tts, TTS_VOICE)
        await communicate.save(str(output_path))
        audio_url = f"/static/audio/{output_filename}"; asyncio.create_task(cleanup_old_audio_files(max_age_seconds=600)); return audio_url
    except Exception as e: logging.error(f"TTS Error: {e}", exc_info=True); return None
async def cleanup_old_audio_files(max_age_seconds: int):
     try:
        now = time.time(); removed_count = 0
        if not AUDIO_DIR.exists(): return
        for filename in os.listdir(AUDIO_DIR):
            if filename.endswith(".mp3"):
                file_path = AUDIO_DIR / filename
                try:
                    if os.path.isfile(file_path):
                        if now - file_path.stat().st_mtime > max_age_seconds: os.remove(file_path); removed_count += 1
                except OSError as e: logging.warning(f"Error removing audio {file_path}: {e}")
        if removed_count > 0: logging.debug(f"Cleaned {removed_count} audio files.")
     except Exception as e: logging.error(f"Audio cleanup error: {e}", exc_info=True)
async def perform_web_search(query: str, http_client: httpx.AsyncClient, num_results: int = 3) -> list[dict]:
    logging.info(f"Performing web search: {query}")
    results = []
    try:
        async with AsyncDDGS(client=http_client) as ddgs: search_results = await ddgs.text(query, region="us-en", max_results=num_results)
        if not search_results: logging.warning("No search results."); return []
        for i, r in enumerate(search_results):
            if r.get('body'): results.append({ "title": r.get('title','NT'), "snippet": r.get('body','NS'), "url": r.get('href','#') })
            if len(results) >= num_results: break
        logging.info(f"Search completed: {len(results)} results.")
        return results
    except Exception as e: logging.error(f"Web search error: {e}", exc_info=True); return []
def extract_commands(text: str) -> tuple[str, str | None, str | None]:
    search_match = re.search(r"\[SEARCH:\s*(.+?)\s*\]", text, re.IGNORECASE)
    memorize_match = re.search(r"\[MEMORIZE:\s*(.+?)\s*\]", text, re.IGNORECASE)
    s_q = None; m_c = None; c_t = text
    if search_match: s_q = search_match.group(1).strip(); c_t = c_t.replace(search_match.group(0), "", 1); memorize_match = None
    elif memorize_match: m_c = memorize_match.group(1).strip(); c_t = c_t.replace(memorize_match.group(0), "", 1)
    return c_t.strip(), s_q, m_c

# --- Ollama Call Function ---
async def call_ollama_granite_vision_browser(
    http_client: httpx.AsyncClient, user_id: str, image_base64: str | None,
    image_source: str, text: str, history: list[dict],
    web_search_results: list[dict] | None = None
) -> tuple[str, str | None, str | None]: # text, search, memorize
    prompt_parts = []; prompt_parts.append("<|start_of_role|>system<|end_of_role|>\n" + SYSTEM_CONTEXT_DESCRIPTION)
    source_text = f"Image from user's {image_source}" if image_source != 'none' else "None (Text chat only)"
    prompt_parts.append(f"<|start_of_role|>system<|end_of_role|>\n**Current Visual Input Source:** {source_text}.")
    recent_mems = await get_recent_memories(user_id, limit=5) # Use corrected helper
    if recent_mems:
        memory_context = "**Relevant Memories:**\n" + "\n".join([f"- [{m.get('type','N/A').upper()} @ {m.get('timestamp','N/A')}]: {m.get('content','')}" for m in reversed(recent_mems)])
        prompt_parts.append("<|start_of_role|>system<|end_of_role|>\n" + memory_context)
    if web_search_results:
        search_context = "**Web Search Results:**\n" + ("\n".join([f"{i+1}. {r.get('title','')}: {r.get('snippet','')}" for i,r in enumerate(web_search_results)]) if web_search_results else "- (No results)") + "\n**Use these results.**"
        prompt_parts.append("<|start_of_role|>system<|end_of_role|>\n" + search_context)
    for turn in history:
        role = turn.get('role', 'user').lower(); content = turn.get('content', '')
        content_str = str(content) if content is not None else ""
        if role == 'user': prompt_parts.append(f"<|start_of_role|>user<|end_of_role|>\n{content_str}")
        elif role == 'assistant': prompt_parts.append(f"<|start_of_role|>assistant<|end_of_role|>\n{content_str}")
    user_content = text if text else f"(Analyze the provided image from {image_source} and share detailed observations/guidance.)"
    prompt_parts.append(f"<|start_of_role|>user<|end_of_role|>\n{user_content}")
    prompt_parts.append("<|start_of_role|>assistant<|end_of_role|>")
    full_prompt = "\n".join(prompt_parts)
    payload = { "model": MODEL_NAME, "prompt": full_prompt, "stream": False, "options": { "num_predict": 350, "temperature": 0.2, "stop": ["<|end_of_role|>", "[SEARCH:", "[MEMORIZE:"] } }
    if image_base64:
        try: img_data = image_base64.split(",", 1)[1]; payload["images"] = [img_data]; logging.debug(f"Image data ({image_source}) included.")
        except Exception as e: logging.warning(f"Image processing error ({image_source}): {e}. Text only.")
    ai_response_text = "(Error)"; search_query_out = None; memory_content_out = None
    try:
        logging.info(f"Sending to Ollama (Model: {MODEL_NAME}, Source: {image_source}, Text: '{user_content[:30]}...', Image: {'Y' if image_base64 else 'N'})")
        start_time = time.time(); response = await http_client.post(OLLAMA_API_URL, json=payload); response.raise_for_status()
        response_data = response.json(); end_time = time.time(); logging.info(f"Ollama response in {end_time - start_time:.2f}s.")
        ai_response_raw = response_data.get("response", "").strip(); logging.debug(f"Ollama Raw: {ai_response_raw[:300]}...")
        ai_response_clean = ai_response_raw.replace("<|start_of_role|>assistant<|end_of_role|>", "").strip()
        cleaned_response, search_query_out, memory_content_out = extract_commands(ai_response_clean)
        ai_response_text = cleaned_response if cleaned_response else "(No text response)"
        logging.info(f"Ollama Final: {ai_response_text[:200]}...")
        # <<< FIX: Corrected logic from previous syntax error >>>
        if search_query_out:
            logging.info(f"Intent: Search '{search_query_out}'")
        if memory_content_out:
             logging.info(f"Intent: Memorize '{memory_content_out}'")
        # <<< END FIX >>>
    except Exception as e: import traceback; logging.error(f"Ollama call failed: {e}"); traceback.print_exc(); ai_response_text = "(Ollama communication error)"
    return ai_response_text, search_query_out, memory_content_out


# --- FastAPI App Setup & Lifespan ---
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    logging.info("Application initializing...")
    app.state.http_client = httpx.AsyncClient(timeout=90.0)
    app.state.ddgs_client = httpx.AsyncClient(timeout=15.0)
    await load_memory_from_json()
    logging.info("Application initialized.")
    yield
    logging.info("Application shutting down...")
    await asyncio.gather(app.state.http_client.aclose(), app.state.ddgs_client.aclose())
    await save_memory_to_json()
    logging.info("Application shutdown complete.")

app = fastapi.FastAPI(lifespan=lifespan)

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_root_html():
    # logging.debug("Serving root HTML.")
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/static/css/style.css", response_class=Response)
async def get_inline_css():
     # logging.debug("Serving inline CSS.")
     return Response(content=CSS_CONTENT, media_type="text/css")

@app.get("/static/js/main.js", response_class=Response)
async def get_inline_js():
    # logging.debug("Serving inline JavaScript.")
    return Response(content=JAVASCRIPT_CONTENT, media_type="application/javascript")

# Mount ONLY the audio directory using StaticFiles
if AUDIO_DIR.exists():
     app.mount("/static/audio", StaticFiles(directory=AUDIO_DIR), name="static_audio")
     logging.info(f"Mounted audio directory: {AUDIO_DIR} at /static/audio")
else:
     logging.error("AUDIO_DIR does not exist at mount time. Audio will fail.")

# --- Request Body Model ---
class ProcessRequest(BaseModel):
    client_id: str = Field(...)
    text: Optional[str] = None
    image: Optional[str] = None
    image_source: str = "none"

# --- Background Task ---
# ... (process_ai_interaction - unchanged) ...
async def process_ai_interaction(req_data: ProcessRequest, http_client: httpx.AsyncClient, ddgs_client: httpx.AsyncClient):
    client_id = req_data.client_id; user_text = req_data.text; image_base64 = req_data.image; image_source = req_data.image_source
    logging.info(f"BG Task started for {client_id}")
    try:
        await _ensure_client_state(client_id)
        current_history = await get_client_history(client_id); pending_search_results = await get_pending_search_results(client_id)
        ai_response_text, search_query, memory_content = await call_ollama_granite_vision_browser(http_client, client_id, image_base64, image_source, user_text or "", current_history, pending_search_results)
        user_turn_content = user_text if user_text else f"({image_source} observation)"; user_turn_hist = {"role": "user", "content": user_turn_content}; ai_turn_hist = {"role": "assistant", "content": ai_response_text}
        response_payload = { "type": "response", "ai_text": ai_response_text, "audio_url": None }
        final_ai_response_sent = False
        if search_query:
            logging.info(f"BG Search for {client_id}: {search_query}")
            await push_sse_message(client_id, {"event": "system", "data": json.dumps({"message": f"(Searching web for '{search_query}'...)"})})
            search_results = await perform_web_search(search_query, ddgs_client)
            await set_pending_search_results(client_id, search_results)
            ai_turn_hist = {"role": "assistant", "content": f"(Initiated search: '{search_query}')"}
            await push_sse_message(client_id, {"event": "system", "data": json.dumps({"message": f"Search done. Ready for next input."})})
        elif memory_content:
            logging.info(f"BG Storing memory for {client_id}: {memory_content}")
            mem_type = 'observation' if image_source != 'none' and not user_text else 'fact'
            if "learn" in memory_content or "didn't know" in memory_content: mem_type = 'learning_point'
            await add_memory_entry(client_id, mem_type, memory_content)
            audio_url = await generate_tts(ai_response_text); response_payload["audio_url"] = audio_url
            await push_sse_message(client_id, {"event": "response", "data": json.dumps(response_payload)})
            final_ai_response_sent = True
        else: # No command
            audio_url = await generate_tts(ai_response_text); response_payload["audio_url"] = audio_url
            await push_sse_message(client_id, {"event": "response", "data": json.dumps(response_payload)})
            final_ai_response_sent = True
        if final_ai_response_sent: await update_client_history(client_id, user_turn_hist, ai_turn_hist)
    except Exception as e:
        logging.error(f"Error in BG task for {client_id}: {e}", exc_info=True)
        try: await push_sse_message(client_id, {"event": "error", "data": json.dumps({"message": f"Processing error: {type(e).__name__}"})})
        except Exception as push_err: logging.error(f"Failed to push BG error via SSE for {client_id}: {push_err}")

# --- HTTP POST Endpoint ---
@app.post("/process")
async def process_request_endpoint(
    request: Request,
    payload: ProcessRequest = Body(...),
    background_tasks: fastapi.BackgroundTasks = fastapi.BackgroundTasks()
):
    client_id = payload.client_id; http_client = request.app.state.http_client; ddgs_client = request.app.state.ddgs_client
    logging.info(f"Received /process from {client_id}, Text: {bool(payload.text)}, Img: {bool(payload.image)}, Src: {payload.image_source}")
    await _ensure_client_state(client_id); await add_sse_queue(client_id)
    background_tasks.add_task(process_ai_interaction, payload, http_client, ddgs_client)
    return fastapi.responses.JSONResponse({"status": "processing", "message": "Request received."}, background=background_tasks)


# --- SSE Endpoint ---
@app.get("/stream/{client_id}")
async def stream_endpoint(request: Request, client_id: str):
    logging.info(f"SSE connection request from client_id: {client_id}")
    await add_sse_queue(client_id)
    async def event_generator() -> Generator[str, None, None]:
        queue: Optional[asyncio.Queue] = None; client_info = f"{client_id} ({request.client.host if request.client else 'unknown'})"
        try:
            async with sse_queue_lock: queue = sse_queues.get(client_id)
            if not queue: raise ValueError("Queue not found")
            yield f"event: connected\ndata: {json.dumps({'message':'SSE connected'})}\n\n"; logging.info(f"SSE stream opened for {client_info}")
            while True:
                message = await queue.get(); event_type = message.get("event", "message"); data_str = message.get("data", "{}")
                sse_msg = f"event: {event_type}\ndata: {data_str}\n\n"; yield sse_msg; logging.debug(f"SSE sent to {client_info}: {sse_msg.strip()}"); queue.task_done()
        except asyncio.CancelledError: logging.info(f"SSE generator cancelled for {client_info}.")
        except ValueError as e: logging.error(f"SSE Setup Error for {client_info}: {e}"); yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        except Exception as e: logging.error(f"Error in SSE generator for {client_info}: {e}", exc_info=True); yield f"event: error\ndata: {json.dumps({'message':'SSE stream error'})}\n\n"
        finally: logging.info(f"SSE stream closing for {client_info}."); await remove_sse_queue(client_id) # Cleanup queue
    # Use headers to prevent caching for SSE
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no", # Useful for Nginx proxying
        "Connection": "keep-alive"
    }
    return EventSourceResponse(event_generator(), media_type="text/event-stream", ping=15, headers=headers)


# --- Uvicorn Execution ---
if __name__ == "__main__":
    print("--- Aura AI Visual Assistant Server (SINGLE FILE / SSE + HTTP / English) ---")
    print(f"--- MODEL: {MODEL_NAME} ---")
    print(f"Memory: {MEMORY_FILE.resolve()}")
    print(f"Audio: {AUDIO_DIR.resolve()}")
    print(f"Ensure Ollama (>=0.5.13) with model '{MODEL_NAME}' is running.")
    print(f"Access at: http://localhost:8000")
    print("-------------------------------------")
    module_name = Path(__file__).stem
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=8000, log_level="info", reload=False)
