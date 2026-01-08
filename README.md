# 🤖 B.T. 2.0 - Vanguard-Class Voice Assistant

> **"Trust me, Pilot. I will not lose you again."** — BT-7274

A Titanfall 2's Titan voice assistant powered by OpenRouter, featuring dual personality modes (BT and Scorch) with advanced wake-word detection, speech recognition, TTS synthesis, and intelligent command execution.
<p align="center">
  <img src="https://media1.tenor.com/m/BnaAWfRhrO0AAAAC/titanfall-2.gif" alt="Titanfall 2">
</p>


## 📋 Table of Contents

- [What is B.T. 2.0?](#what-is-bt-20)
- [Quick Start](#quick-start)
- [Installation & Setup](#installation--setup)
- [API Configuration](#api-configuration)
- [Pipeline Architecture](#pipeline-architecture)
- [Folder Structure](#folder-structure)
- [Decision Logic](#decision-logic)
- [Features & Capabilities](#features--capabilities)
- [Command Reference](#command-reference)
- [Debugging & Scaling](#debugging--scaling)
- [Troubleshooting](#troubleshooting)

---

## What is B.T. 2.0?

B.T. 2.0 is a **voice-driven desktop assistant** designed to control your computer through natural language commands. It features:

- **Dual AI Personalities**: Switch between BT-7274 (logical, mission-focused Vanguard) and Scorch (aggressive, thermite-loving Ogre)
- **Wake-Word Detection**: Always listening for "Hey BT" via Picovoice Porcupine
- **Speech Recognition**: Google Speech-to-Text for command parsing
- **Voice Synthesis**: Piper TTS with dual voice models for BT and Scorch
- **LLM Integration**: OpenAI API for intent classification and conversational fallback
- **Smart Command Execution**: 40+ system, media, web, and utility commands

> **Pilot Protocol 3 Engaged**: Protect the civilian population. Secure the area. Link with the Titan.

---

## Quick Start

### Prerequisites
- **Python 3.9+**
- **Windows 10/11** (uses `win32gui`, `psutil`, etc.)
- **Microphone** for audio input
- **Internet connection** (for APIs and TTS)

### Installation in 5 Minutes

```bash
# 1. Clone or download the project
cd "c:\Work\important stuff\B.T. 2.0"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (automatic on first run, but you can pre-download)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 5. Configure config.json (see next section)

# 6. Run the assistant
python main.py
```

---

## Installation & Setup

### Step 1: Install Dependencies

Create a `requirements.txt` with:

```txt
# Speech & Audio
SpeechRecognition==3.10.0
sounddevice==0.4.6
soundfile==0.12.1
pydub==0.25.1
gTTS==2.3.2
piper-tts==1.2.0
pvporcupine==3.0.2

# LLM & Translation
openai==1.3.0
deep-translator==1.11.4

# System & Control
pyautogui==0.9.53
psutil==5.9.6
Pillow==10.0.0
win32-setctime==1.1.0
pyperclip==1.8.2
python-Levenshtein==0.21.1

# Spotify Integration
spotipy==2.22.1

# Audio Control
pycaw==20240128

# GUI & Data
nltk==3.8.1
requests==2.31.0
```

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Voice Models

The project uses **Piper TTS** for voice synthesis. You need:

1. **Download Piper binary** from [rhasspy/piper releases](https://github.com/rhasspy/piper/releases)
   - Extract to `piper/` folder
   - Should have: `piper.exe`, `espeak-ng-data/`, `pkgconfig/`

2. **Download Voice Models**:
   - BT Voice: `en_US-lessac-medium.onnx` (place in `Titanfall2/BT7274/`)
   - Scorch Voice: `en_GB-alan-medium.onnx` (place in `Titanfall2/ScorchAI/`)

3. **espeak-ng** is bundled with Piper (no separate install needed)

### Step 3: Wake-Word Model

The `Hey_Bt.ppn` file is **Picovoice Porcupine's** keyword model. It's provided in the repo—no additional setup needed unless you want to custom-train a new model.
Just change the api key that you'll find in the `config.json` and in  `main.py`for porcupine.

---

## API Configuration

### config.json Setup

Create or edit `config.json` with your API keys and settings:

```json
{
  "paths": {
    "piper_exe": "piper/piper.exe",
    "bt_voice_model": "Titanfall2/BT7274/BT7274.onnx",
    "scorch_voice_model": "Titanfall2/ScorchAI/ScorchAI.onnx",
    "tts_cache_dir": "tts_cache"
  },
  "tts_settings": {
    "piper_timeout": 30,
    "sentences_per_chunk": 2,
    "use_multiprocessing_threshold": 2,
    "max_concurrent_piper": 2
  },
  "audio": {
    "default_input_device_name": "YOUR_MIC_NAME",
    "default_output_device_name": "YOUR_SPEAKERS_NAME",
    "listen_duration": 7
  },
  "llm_service": {
    "api_key": "sk-proj-YOUR_OPENAI_API_KEY",
    "model": "openai/gpt-3.5-turbo",
    "site_url": "http://localhost",
    "site_name": "TitanAssistant"
  },
  "vision_service": {
    "model": "qwen/qwen2.5-vl-32b-instruct:free",
    "default_prompt": "What is in this image?",
    "read_prompt": "Extract all text from this image"
  },
  "spotify": {
    "client_id": "YOUR_SPOTIFY_CLIENT_ID",
    "client_secret": "YOUR_SPOTIFY_CLIENT_SECRET",
    "redirect_uri": "http://localhost:8888/callback"
  },
  "commands": [
    // See Command Reference section
  ],
  "dialogue_pools": {
    "startup": ["BT online. Standing by for mission parameters."],
    "shutdown": ["Good luck, Pilot. See you on the Frontier."],
    "ptt_ack": ["Acknowledged."],
    "confirmation": ["Awaiting confirmation, Pilot."],
    "error": ["Unable to comply."]
  }
}
```

### API Keys Explained

| API | Purpose | Cost | Setup |
|-----|---------|------|-------|
| **OpenAI** | Intent classification & conversational LLM | ~$0.15/1k tokens | Get key at [platform.openai.com](https://platform.openai.com) |
| **Picovoice** | Wake-word detection (Porcupine) | Free tier available | Access key in code (limited but functional) |
| **Spotify** | Music control & now-playing info | Free with account | [spotify-dev.com](https://developer.spotify.com) |
| **Google Speech-to-Text** | Transcription (free via SpeechRecognition lib) | ~$0.006/15min | Built-in, no key needed |

> **Caution**: Store API keys in environment variables in production:
> ```python
> import os
> api_key = os.getenv('OPENAI_API_KEY')
> ```

---

## Pipeline Architecture

### Audio Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAIN LOOP                                │
├─────────────────────────────────────────────────────────────────┤
│  1. Wake-Word Detection (Porcupine)                             │
│     └─→ Listening continuously for "Hey BT"                     │
│         (8kHz, 16-bit PCM)                                      │
│                                                                  │
│  2. PTT Acknowledged                                            │
│     └─→ Play "ptt_ack" sound (cached BT voice)                  │
│                                                                  │
│  3. Listen for Command                                          │
│     └─→ Record up to 7 seconds of audio                         │
│     └─→ Stop on silence detected                                │
│                                                                  │
│  4. Speech-to-Text (Google)                                     │
│     └─→ Convert AudioData → Query string                        │
│                                                                  │
│  5. Command Processing                                          │
│     ├─→ Exact Keyword Match? YES → Execute (Fast)              │
│     ├─→ No Match? → LLM Intent Classification                   │
│     │   └─→ Classify intent → Map to command type               │
│     ├─→ Still No Match? → Conversational LLM fallback           │
│     └─→ Execute Action (speak, open app, control media, etc.)   │
│                                                                  │
│  6. Text-to-Speech (Piper) [PIPELINED]                          │
│     ├─→ Split response into chunks (2 sentences/chunk)          │
│     ├─→ Launch Piper processes (max 2 concurrent)               │
│     ├─→ Play chunks as they complete (don't wait for all)       │
│     └─→ Cache results for reuse                                 │
│                                                                  │
│  Loop Back to Step 1                                            │
└─────────────────────────────────────────────────────────────────┘
```

### TTS Pipelining Strategy

**Problem**: Generating full voice responses takes 2-5 seconds per 30 seconds of audio.

**Solution**: **Multiprocessing + Streaming Playback**

```python
# Example: Response = "First sentence. Second sentence. Third sentence."
# Chunks = ["First sentence.", "Second sentence.", "Third sentence."]

# Traditional (Sequential) - Wait 3s total
chunk1_path = generate(chunks[0])  # 1s
play(chunk1_path)                   # 1s
chunk2_path = generate(chunks[1])  # 1s
play(chunk2_path)                   # 1s
chunk3_path = generate(chunks[2])  # 1s
play(chunk3_path)                   # 1s
# Total: 6 seconds ⚠️

# Pipelined (Parallel + Streaming) - Wait ~2s total
Process(generate, chunks[0])  # Start immediately
Process(generate, chunks[1])  # Start immediately
play(chunk1_path)            # 1s (while chunk2 still generating)
# chunk2 finishes during chunk1 playback
play(chunk2_path)            # 1s (while chunk3 still generating)
play(chunk3_path)            # 1s
# Total: ~2 seconds ✅
```

### Caching System

- **Dialogue pool entries** (startup, shutdown, etc.) cached by mode + phrase hash
- **Generated TTS** stored as `.wav` in `tts_cache/`
- **Reuse on repeat commands** (e.g., "What time is it?" always uses cached response)
- **Auto-cleanup**: Temporary files deleted after playback

---

## Folder Structure

```
B.T. 2.0/
├── main.py                 # Main assistant code
├── config.json            # Configuration (API keys, paths, commands)
├── Hey_Bt.ppn             # Porcupine wake-word model
├── Overlyrics.py          # Lyrics display module (optional)
├── requirements.txt       # Python dependencies
│
├── piper/                 # Piper TTS binary & data
│   ├── piper.exe          # TTS executable
│   ├── espeak-ng-data/    # Phoneme data (60+ languages)
│   └── pkgconfig/         # Library configuration
│
├── Titanfall2/            # Voice models & project data
│   ├── BT7274/
│   │   ├── BT7274.onnx         # BT's voice model (primary)
│   │   ├── BT7274.onnx.json    # Model metadata
│   │   └── voices.json         # Voice configuration
│   ├── ScorchAI/
│   │   ├── ScorchAI.onnx       # Scorch's voice model
│   │   ├── ScorchAI.onnx.json  # Model metadata
│   │   ├── voices.json         # Voice configuration
│   │   └── test.py             # Testing script
│   └── Todo.md            # Development notes
│
├── tts_cache/             # Generated audio cache
│   ├── cache_bt_a1b2c3d4e5f6.wav
│   ├── cache_scorch_f6e5d4c3b2a1.wav
│   └── ...
│
└── __pycache__/           # Python bytecode (auto-generated)
```

---

## Decision Logic

### How Does B.T. Choose What to Do?

```python
# DECISION TREE (in process_command)

query = "play darude sandstorm"  # User says this

# LEVEL 1: Mode Switch Detection (Highest Priority)
if "switch mode" in query or "switch to scorch" in query:
    → Change current_mode
    → Execute immediately
    → Return (skip remaining levels)

# LEVEL 2: Exact Keyword Matching (Fast Path)
for command in CONFIG['commands']:
    for keyword in command['keywords']:
        if query.startswith(keyword):  # Exact match wins!
            → Execute command
            → Return (skip remaining levels)
            
# LEVEL 3: LLM Intent Classification (Smart Path)
# Only if no exact match found AND LLM available
intent = llm_client.classify_intent(query)
# intent = "play_music" (from a predefined set of ~25 intents)

intent_to_command_map = {
    "play_music": {"type": "media.play_music", ...},
    "set_volume": {"type": "system.set_volume", ...},
    "open_website": {"type": "web.open", ...},
    # ... 25+ more mappings
}

command = intent_to_command_map.get(intent)
if command:
    → Execute command with parsed data
    → Return

# LEVEL 4: Conversational Fallback (Weakest)
# If still no match, hand off to conversational LLM
response = llm_client.chat(query, system_prompt=bt_system_prompt)
speak(response)  # "I'm not familiar with that request, Pilot."
```

### Why This Order?

1. **Mode switching** is critical (immediate response needed)
2. **Exact keywords** are fastest (~5ms, no LLM call)
3. **LLM classification** handles variations ("play music" vs "start playing a song")
4. **Conversational fallback** handles questions and casual chat

### Example Decision Traces

| User Query | Decision Path | Result |
|------------|---------------|--------|
| "Switch to Scorch" | Level 1 | Switches mode, speaks confirmation |
| "Play Hacking to the Gate" | Level 2 → Keyword "play" matches | Executes `media.play_music` |
| "Can you play some chill music?" | Level 3 → LLM: "play_music" | Executes `media.play_music` with data |
| "Are you alive?" | Level 4 → Conversational LLM | BT responds philosophically |

---

## Features & Capabilities

### Core Features

| Feature | Status | Notes |
|---------|--------|-------|
| **Wake-Word Detection** | ✅ | "Hey BT" always listening (Porcupine) |
| **Speech Recognition** | ✅ | Google Speech-to-Text (free, accurate) |
| **Voice Synthesis** | ✅ | Dual voices (BT & Scorch), pipelined TTS |
| **Command Execution** | ✅ | 40+ commands (system, media, web, utility) |
| **Intent Classification** | ✅ | LLM-powered, 25+ intent categories |
| **Conversational AI** | ✅ | OpenAI GPT for fallback responses |
| **Spotify Integration** | ✅ | Play/pause, skip, current track, volume |
| **Vision Analysis** | ✅ | Screenshot + OCR (QwEN Vision model) |
| **Multi-Mode** | ✅ | Switch between BT (logical) and Scorch (aggressive) |
| **TTS Caching** | ✅ | Reuse generated audio for fast responses |
| **Audio Device Config** | ✅ | Choose input/output devices from config |

### Command Categories

#### 🎵 **Media Control** (7 commands)
- `play [song]` → Spotify playback
- `pause` / `resume` → Toggle music
- `next` / `skip` → Next track
- `previous` / `back` → Previous track
- `volume [0-100]` → Set volume
- `mute` / `unmute` → Mute audio
- `what's playing` / `now playing` → Current track info

#### 🌐 **Web & App** (8 commands)
- `open [website]` → Launch URL (Chrome, Firefox, Edge)
- `search [query]` → Google search
- `open [app]` → Launch application
- `close [app]` → Terminate application
- `switch to [app]` → Bring window to front

#### 💻 **System Control** (12 commands)
- `set brightness [0-100]` → Display brightness
- `set volume [0-100]` → Master volume
- `wifi on` / `wifi off` → Network control
- `lock computer` → Lock screen
- `shutdown` / `restart` → System reboot
- `list windows` → Show open windows
- `system status` → CPU, RAM, disk usage
- `top processes` → Running applications
- `set input device [name]` → Microphone selection
- `set output device [name]` → Speaker selection

#### 🔧 **Utilities** (8 commands)
- `translate [text] to [language]` → Google Translate + TTS
- `analyze screen` → Vision LLM analysis + OCR
- `open lyrics` / `close lyrics` → Display song lyrics
- `type [text]` → Keyboard input
- `what time is it` → Current time
- `what's the date` → Current date
- `tell me a joke` → Humor (configurable)
- `switch mode` → Toggle BT ↔ Scorch

#### 📋 **Window Control** (5 commands)
- `minimize [window]` → Reduce window
- `maximize [window]` → Expand window
- `close [window]` → Terminate window
- `next tab` → Switch browser tab
- `list windows` → Show all open windows

---

## Command Reference

### Awesome Commands to Try 🎮

Here are the **coolest commands** that showcase B.T. 2.0's power:

#### 1. **Vision Analysis** (Requires Vision API)
```
"Analyze my screen"
"Read what's on the screen"
```
→ Captures screenshot, compresses it, sends to QwEN Vision LLM, and reads the result aloud.
**Use Case**: Quick reading of notifications, website content, or screen info without looking.

#### 2. **Smart Search**
```
"Search for titanfall 3 news"
"Open youtube"
```
→ Detects intent, opens browser, performs search (or just opens site).
**Use Case**: Hands-free web browsing while working.

#### 3. **Music Playback with Intent**
```
"Play darude sandstorm"
"I want to hear some lofi beats"
```
→ LLM classifies as music intent, launches Spotify, searches for song/playlist.
**Use Case**: Voice control without knowing exact song titles.

#### 4. **Dual Personality Switch**
```
"Switch to Scorch"
```
→ Changes voice, system prompt, and personality instantly.
- **BT**: Logical, concise, mission-focused
- **Scorch**: Aggressive, combat-oriented, laconic

**Use Case**: Roleplay, different moods, or entertainment.

#### 5. **Real-Time Translation**
```
"Translate hello world to spanish"
```
→ Translates via Google Translate, plays audio in target language via gTTS.
**Use Case**: Quick language learning or communication.

#### 6. **Screen Content Reading** (OCR)
```
"Read my screen"
```
→ Captures image, compresses, sends to Vision LLM, extracts & speaks all text.
**Use Case**: Accessibility, reading fine print, verifying passwords visually.

#### 7. **Smart Window Management**
```
"Switch to Chrome"
"Minimize Spotify"
"Close this window"
```
→ Fuzzy-matches open windows by name, performs action.
**Use Case**: Window organization without touching mouse.

#### 8. **Dynamic System Status**
```
"How's my system?"
"What are the top processes?"
```
→ Reads CPU, RAM, disk, running processes; speaks in BT's voice.
**Use Case**: Quick diagnostics while gaming or working.

#### 9. **Automatic Lyrics Display**
```
"Show me the lyrics"
"Close the lyrics"
```
→ Launches external lyrics viewer (OVLyrics) or closes it.
**Use Case**: Karaoke-style singing along.

#### 10. **Mode-Aware Caching**
```
(Ask for the time twice rapidly)
```
→ First call: generates voice file
→ Second call: reuses cached file (instant response)
**Use Case**: See the difference in speed for repeated requests.

---

## Debugging & Scaling

### Performance Optimization

#### 1. **TTS Bottleneck** (Most Common)
**Problem**: Voice synthesis slow (2-5s per response)

**Solutions**:
```python
# In config.json
{
  "tts_settings": {
    "sentences_per_chunk": 2,           # ↑ Increase = fewer chunks = faster
    "use_multiprocessing_threshold": 2, # ↓ Lower = use parallel sooner
    "max_concurrent_piper": 2           # ↑ Increase = more parallel processes
  }
}
```

**Benchmark**:
- `sentences_per_chunk: 1` → Many small chunks → Better streaming, slower overall
- `sentences_per_chunk: 3` → Fewer large chunks → Faster overall, chunkier playback
- **Sweet spot**: 2 (balance of both)

#### 2. **LLM Latency** (Intent Classification)
**Problem**: First response slow (LLM API call takes 1-2s)

**Solutions**:
- Use exact keywords instead (5ms vs 1500ms)
- Increase `max_tokens` in config only if needed
- Fallback to conversational if intent classification fails
- Cache common intents in local config

#### 3. **Speech Recognition Timeout**
**Problem**: "Waiting for microphone" stalls input

**Solutions**:
```python
# Adjust in code:
listen_timeout = 5      # Max wait for speech to START
duration = 7            # Max length of one command
```

#### 4. **Memory Leak in TTS Cache**
**Problem**: `tts_cache/` grows unbounded

**Solutions**:
```python
# Add periodic cleanup (monthly)
import glob
for f in glob.glob("tts_cache/*.wav"):
    if time.time() - os.path.getmtime(f) > 30*24*3600:  # 30 days
        os.unlink(f)
```

---

### Scaling to Enterprise

**Challenge**: Running 100+ instances, centralized audio storage

**Architecture**:
```
┌─────────────────────────────────────────────┐
│   B.T. 2.0 Instance (Local PC)              │
│   ├─ Wake-word detection                    │
│   ├─ Command parsing (local LLM/rules)      │
│   └─ UI/Audio output                        │
└──────────────┬──────────────────────────────┘
               │ (API calls only)
               ▼
┌─────────────────────────────────────────────┐
│   Central Server                            │
│   ├─ OpenAI API (shared quota)              │
│   ├─ TTS service (Piper + caching)          │
│   ├─ Shared command config                  │
│   └─ Metrics & logging                      │
└─────────────────────────────────────────────┘
```

**Implementation**:
1. Move TTS to server (GPU-accelerated)
2. Cache results in Redis
3. Distribute config via API endpoint
4. Use async/await for all I/O

---

### Debugging Checklist

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| Wake-word not triggering | Check mic levels: `sd.query_devices()` | Increase `sensitivities` in code (0.6 → 0.8) |
| Transcription says "None" | Check internet, Google SR rate limits | Wait 30s, try again |
| Voice response too fast/slow | Piper output speed | Edit `piper_generate_worker` speech rate |
| LLM response nonsensical | System prompt mismatch | Verify `bt_system_prompt` in `get_llm_response()` |
| Audio device selection failing | Device names mismatch config | Run `sd.query_devices()` to find exact names |
| Cache files not cleaning up | Temp file deletion failing | Check disk permissions on `tts_cache/` |
| "Piper timeout" error | Audio file too long | Lower `SENTENCES_PER_CHUNK` or increase `PIPER_TIMEOUT` |

---

## Troubleshooting

### Common Issues & Fixes

#### 🔊 "No Audio Device Found"
```python
# Diagnose:
import sounddevice as sd
print(sd.query_devices())

# Look for your device, get index
# Update config.json:
{
  "audio": {
    "default_input_device_name": "YOUR_EXACT_MIC_NAME",
    "default_output_device_name": "YOUR_EXACT_SPEAKER_NAME"
  }
}
```

#### 🤫 "Microphone Sensitivity Too High/Low"
```python
# In calibrate_microphone():
MINIMUM_ENERGY_THRESHOLD = 300  # ↑ Increase = less sensitive
                                 # ↓ Decrease = more sensitive
```

#### 📡 "API Key Invalid"
```json
{
  "llm_service": {
    "api_key": "sk-proj-..."  // Must start with "sk-proj-" or "sk-"
  }
}
```
Get from: https://platform.openai.com/api-keys

#### 🎙️ "Wake-Word Not Detecting"
```python
# In main.py, check Porcupine initialization:
porcupine = pvporcupine.create(
    access_key="YOUR_KEY",
    keyword_paths=["Hey_Bt.ppn"],
    sensitivities=[0.6]  # ↑ 0.6 = default, 0.9 = very sensitive, 0.3 = strict
)
```

#### ⏱️ "Commands Timing Out"
```python
# In config.json:
{
  "tts_settings": {
    "piper_timeout": 30  # ↑ Increase for long responses
  }
}
```

#### 🔥 "GPU Memory Issues" (if using GPU TTS)
```bash
# Force CPU-only Piper:
export CUDA_VISIBLE_DEVICES=""
python main.py
```

---

## Advanced Configuration

### Custom Commands

Add to `config.json`:

```json
{
  "commands": [
    {
      "name": "Play Music",
      "type": "media.play_music",
      "keywords": ["play"],
      "ack": "Starting playback."
    },
    {
      "name": "Custom Greeting",
      "type": "custom.greet",
      "keywords": ["hey", "hello"],
      "ack": "What's up, Pilot?"
    }
  ]
}
```

Then handle in `execute_action()`:
```python
elif action_type == "custom.greet":
    speak("Acknowledged. Standing by for orders.")
```

### Custom Dialogue Pools

```json
{
  "dialogue_pools": {
    "startup": [
      "BT online. Standing by for mission parameters.",
      "Vanguard class Titan, ready for deployment."
    ],
    "error": [
      "Unable to comply.",
      "That request exceeds my current parameters."
    ]
  }
}
```

---

## Performance Metrics

On Windows 11 (i7-12700K, 16GB RAM):

| Operation | Time | Notes |
|-----------|------|-------|
| Wake-word detection | < 100ms | Per audio frame |
| Command transcription | 0.5-2s | Network-dependent |
| Exact keyword match | ~5ms | No LLM call |
| LLM intent classification | 1-3s | API latency |
| TTS generation (sequential) | 2-5s | Per response |
| TTS generation (pipelined 2 chunks) | 1-2s | Parallel processing |
| Spotify command latency | 0.5-1.5s | API + playback start |

---

## References & Resources

- **Piper TTS**: https://github.com/rhasspy/piper
- **Picovoice Porcupine**: https://picovoice.ai/
- **OpenAI API**: https://platform.openai.com/
- **Spotipy Docs**: https://spotipy.readthedocs.io/
- **SpeechRecognition**: https://github.com/Uberi/speech_recognition

---

## License & Credits

**B.T. 2.0** is a fan project inspired by Titanfall 2. All Titanfall 2 assets and characters belong to Respawn Entertainment / EA Games.

- **BT-7274**: "Guardian of the Frontier"
- **Scorch**: "Area Denial Specialist"
- **Pilot**: That's you, Titan. Let's move.

---

## Join the Frontier 🎮

> "Stay sharp, Pilot. We've got a lot of work to do." — BT-7274

For issues, feature requests, or Titanfall memes, reach out or open an issue on GitHub.

**Mission Status**: ACTIVE ✅

---

**Last Updated**: January 8, 2026
**Version**: 2.0 (Vanguard Protocol)
