# 🤖 Titan Voice Assistant (BT-7274)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-blue)
![Status](https://img.shields.io/badge/Status-Operational-success)
![Offline%20TTS](https://img.shields.io/badge/TTS-Piper%20Offline-orange)
![LLM](https://img.shields.io/badge/LLM-OpenRouter-purple)

> *“Protocol 1: Link to Pilot.”*  
> *“Protocol 2: Uphold the Mission.”*  
> *“Protocol 3: Protect the Pilot.”*  

Titan is a **systems-level AI desktop assistant**, inspired by **BT‑7274 (Titanfall)**.  
It combines **offline speech**, **wake-word detection**, **LLM intelligence**, **vision**, and **deep OS automation** into a single coherent architecture.

This README is intentionally **long and exhaustive**.  
It exists so you can:
- Understand *every subsystem*
- Defend this project in **college reviews / vivas**
- Extend it without breaking things

---

## 🎬 Demo (Recommended)
*(Add GIFs here later)*
```text
/demo/
 ├── wake_word.gif
 ├── spotify_control.gif
 ├── screen_analysis.gif
```

---

## 🚀 Features Overview

### 🎙 Voice Interface
- Wake word: **“Hey BT”**
- Push‑to‑Talk fallback (F7)
- Dual personalities:
  - **BT Mode** – calm, logical
  - **Scorch Mode** – aggressive, tactical

### 🧠 Intelligence
- Exact keyword command execution
- LLM‑based intent classification
- Conversational fallback (OpenRouter)
- Context‑aware confirmations

### 🖥 OS Control
- App launch / termination
- Window focus, minimize, maximize
- Volume & brightness control
- Lock / shutdown / restart
- Wi‑Fi on/off

### 🌐 Media & Web
- Spotify playback + lyrics
- Website launching
- Google / YouTube search

### 👁 Vision AI
- Screenshot capture
- OCR‑like text reading
- Scene understanding

---

## 🧱 Architecture (High‑Level)

```text
┌────────────┐
│ Wake Word  │  ← Porcupine
└─────┬──────┘
      ↓
┌────────────┐
│ Speech Rec │  ← speech_recognition
└─────┬──────┘
      ↓
┌────────────┐
│ Command    │  ← Exact match
│ Processor  │  ← LLM intent
└─────┬──────┘
      ↓
┌────────────┐
│ Action     │  ← OS / Web / Media
│ Executor   │
└─────┬──────┘
      ↓
┌────────────┐
│ Piper TTS  │  ← Offline voice
└────────────┘
```

---

## 📦 requirements.txt

```txt
openai
pvporcupine
speechrecognition
sounddevice
soundfile
pillow
psutil
pyautogui
pynput
spotipy
screen-brightness-control
nltk
deep-translator
gtts
playsound
pycaw
python-Levenshtein
requests
```

---

## 🧠 LLM Intent System (Deep Explanation)

Titan uses LLMs **only when deterministic parsing fails**.

### Intent Classification Prompt (Conceptual)
```text
You are an intent classifier.
Return ONLY the intent key.

User: "play some music"
Output: play_music
```

### Intent Flow
1. Try exact keyword match
2. If not found → send query to LLM
3. LLM returns **single intent token**
4. Intent maps to command type
5. Data extracted via regex
6. Action executed locally

### Why This Is Fast
- No LLM call for known commands
- No embeddings / vector DB
- Stateless classification

---

## 👁 Vision Pipeline

```text
Screenshot
 → Resize (50%)
 → JPEG compression
 → Base64
 → Vision LLM
 → Spoken summary
```

Why it’s efficient:
- Reduces token cost
- Faster response
- OCR + description combined

---

## 🛡️ Security & Privacy

### What Stays Local
- Microphone audio
- Wake‑word detection
- Text‑to‑speech
- System control
- Screenshots (temporary)

### What Goes Online
- LLM queries (text only)
- Vision analysis (compressed image)
- Spotify API calls

### No:
- Continuous audio streaming
- Keylogging
- Background uploads

⚠ **Important**
- API keys live in `config.json`
- Never commit them publicly

---

## ⚙️ Setup Checklist

### 1️⃣ Python
Python **3.10+** recommended

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Configure `config.json`
You MUST update:
- Piper paths
- App paths
- Spotify credentials
- Porcupine key

### 4️⃣ Run
```bash
python main.py
```

Expected voice:
> *“BT‑7274 online and ready for combat.”*

---

## 🧩 How to Extend

### Add New Command
1. Add entry in `config.json`
2. Map intent (optional)
3. Handle in `execute_action()`

### Add New Voice
- Add Piper ONNX model
- Update paths
- Switch via voice command

---

## 🧪 Debugging Tips

| Issue | Likely Cause |
|-----|------------|
| Wake word fails | Mic / Porcupine key |
| No voice output | Piper path |
| LLM not responding | API key |
| Spotify error | OAuth cache |

Logs are timestamped for clarity.

---

## 🏁 Final Words

This is **not a chatbot**.  
This is a **desktop AI operator**.

> *“The Pilot is in control.”*  

Build responsibly. Extend fearlessly.

— **Titan 🤖**
