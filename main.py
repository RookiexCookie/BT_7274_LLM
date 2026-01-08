import os
import sys
import time
import random
import datetime
import tempfile
import subprocess
import threading
import json
import re
from pathlib import Path
from multiprocessing import Process, Queue, freeze_support # Added freeze_support
import queue  # Standard library for queue.Empty
import screen_brightness_control as sbc; HAS_BRIGHTNESS = True
# --- LLM Dependency ---
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound import playsound
import glob
from PIL import ImageGrab
from typing import Dict, Tuple, List, Optional
import base64
 # We'll use this for cleaning up temp translation files
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    print("Warning: 'openai' library not found. LLM features will be disabled.")
    print("Install it with: pip install openai")
    HAS_OPENAI = False

# --- Core Dependencies ---
import webbrowser
import pyautogui
from pynput.keyboard import Key, Listener as KeyboardListener
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import requests
import psutil # Keep for system status (optional command, but useful)
import nltk # For sentence tokenization in TTS
from nltk.tokenize import sent_tokenize
import io # Required for NLTK download check

# --- Optional Dependencies ---
# Noise reduction disabled for simplicity as requested previously
HAS_NR = False

# Keep Spotify dependency
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    HAS_SPOTIPY = True
except ImportError:
    print("Warning: 'spotipy' library not found. Spotify features will be disabled.")
    print("Install it with: pip install spotipy")
    HAS_SPOTIPY = False
import pvporcupine
import struct
import sounddevice as sd
# ... (near your other imports like psutil)

import win32gui
import win32process
import win32con
import win32api

# For fuzzy matching (from win_control.py)
try:
    import Levenshtein
    def similarity(a, b):
        return Levenshtein.ratio(a, b)
except Exception:
    import difflib
    def similarity(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()

#controlling volume
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    HAS_PYCAW = True
except ImportError:
    print("Warning: 'pycaw' library not found. Volume control will be disabled.")
    print("Install it with: pip install pycaw")
    HAS_PYCAW = False
# ==============================================================================
# ---------- CONFIGURATION & GLOBAL STATE ----------
# ==============================================================================
CONFIG = {}
SCRIPT_DIR = Path() # Will be set after loading config
TTS_CACHE_DIR = Path() # Will be set after loading config
PIPER_EXE_FULL_PATH = Path() # Will be set after loading config
PIPER_TIMEOUT = 30 # Default
SENTENCES_PER_CHUNK = 2 # Default
current_mode = "BT" # Start with BT
bt_voice_model_path = Path()
scorch_voice_model_path = Path()
memory_data = {} # Re-added
# watchdog_data removed
MEMORY_FILE_PATH = Path() # Re-added
CLIPBOARD_LOG_PATH = Path() # Re-added
# WATCHDOG_FILE_PATH removed
# --- GLOBAL OBJECTS ---
recognizer = sr.Recognizer()
is_speaking = threading.Event()
is_recording = threading.Event()
mic_lock = threading.Lock()
sp = None # Spotify object
llm_client = None
porcupine = None  # <-- ADD THIS
wake_stream = None # <-- ADD THIS
def load_config():
    """Loads configuration and sets up necessary paths and clients."""
    global CONFIG, SCRIPT_DIR, TTS_CACHE_DIR, PIPER_EXE_FULL_PATH, PIPER_TIMEOUT
    global SENTENCES_PER_CHUNK, llm_client, sp, bt_voice_model_path, scorch_voice_model_path

    try:
        SCRIPT_DIR = Path(__file__).parent
        config_path = SCRIPT_DIR / "config.json"
        if not config_path.exists():
            print(f"FATAL ERROR: config.json not found in {SCRIPT_DIR}")
            sys.exit(1)

        with open(config_path, "r", encoding='utf-8') as f: # Specify encoding
            CONFIG = json.load(f)

        # Validate essential paths
        paths = CONFIG.get('paths', {})
        piper_exe = paths.get('piper_exe')
        bt_voice = paths.get('bt_voice_model')
        scorch_voice = paths.get('scorch_voice_model')
        cache_dir = paths.get('tts_cache_dir')

        
        if not all([piper_exe, bt_voice, scorch_voice, cache_dir]):
            raise ValueError("Missing essential paths (piper_exe, voice models, tts_cache_dir) in config.json")

        PIPER_EXE_FULL_PATH = SCRIPT_DIR / piper_exe
        bt_voice_model_path = SCRIPT_DIR / bt_voice
        scorch_voice_model_path = SCRIPT_DIR / scorch_voice
        TTS_CACHE_DIR = SCRIPT_DIR / cache_dir
        TTS_CACHE_DIR.mkdir(exist_ok=True) # Ensure cache directory exists

        if not PIPER_EXE_FULL_PATH.exists():
             raise FileNotFoundError(f"Piper executable not found at {PIPER_EXE_FULL_PATH}")
        if not bt_voice_model_path.exists():
             raise FileNotFoundError(f"BT voice model not found at {bt_voice_model_path}")
        if not scorch_voice_model_path.exists():
             raise FileNotFoundError(f"Scorch voice model not found at {scorch_voice_model_path}")

        # TTS Settings
        tts_settings = CONFIG.get('tts_settings', {})
        PIPER_TIMEOUT = tts_settings.get('piper_timeout', 30)
        SENTENCES_PER_CHUNK = tts_settings.get('sentences_per_chunk', 2)

        # --- Initialize LLM Client ---
        if HAS_OPENAI:
            llm_config = CONFIG.get('llm_service')
            # Check API key more reliably
            if llm_config and llm_config.get('api_key') and not llm_config['api_key'].startswith("sk-or-v1-"):
                 print("WARN: Invalid OpenRouter API key format detected in config.json. LLM functions might fail.")

            if llm_config and llm_config.get('api_key'):
                try:
                    llm_client = OpenAI(
                        base_url=llm_config.get('base_url', "https://openrouter.ai/api/v1"),
                        api_key=llm_config['api_key'],
                    )
                    safe_print("LLM uplink established.")
                except Exception as e:
                    safe_print(f"ERROR: Failed to initialize LLM client: {e}")
                    llm_client = None
            else:
                 safe_print("WARN: LLM service configuration missing or incomplete in config.json.")
                 llm_client = None
        else:
            safe_print("WARN: 'openai' library not installed. LLM functions offline.")
            llm_client = None

        # --- Initialize Spotify Client ---
        initialize_spotify() # Moved Spotify init to its own function for clarity


    except FileNotFoundError as e:
        print(f"FATAL ERROR: Required file not found: {e}")
        sys.exit(1)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"FATAL ERROR processing config.json: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR during configuration loading: {e}")
        sys.exit(1)
# === WAKE WORD INITIALIZATION ===
porcupine = pvporcupine.create(
    access_key="YOUR_PICOVOICE_ACCESS_KEY_HERE",
    keyword_paths=["Hey_Bt.ppn"],
    sensitivities=[0.6]
)


wake_stream = sd.RawInputStream(
    samplerate=porcupine.sample_rate,
    blocksize=porcupine.frame_length,
    dtype='int16',
    channels=1,
    callback=None
)
wake_stream.start()

def detect_wake_word():
    data = wake_stream.read(porcupine.frame_length)[0]
    pcm = struct.unpack_from("h" * porcupine.frame_length, data)
    keyword_index = porcupine.process(pcm)
    return keyword_index >= 0

def check_nltk_data():
    """Checks if NLTK 'punkt' and 'punkt_tab' data are available and downloads if not."""
    resources_needed = ['punkt', 'punkt_tab']
    all_found = True
    for resource in resources_needed:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            safe_print(f"NLTK '{resource}' data found.")
        except LookupError:
            all_found = False
            safe_print(f"NLTK '{resource}' data not found.")
            safe_print(f"Attempting to download '{resource}'...")
            try:
                nltk.download(resource, quiet=True)
                # Verify download
                nltk.data.find(f'tokenizers/{resource}')
                safe_print(f"'{resource}' data downloaded successfully.")
            except Exception as e:
                safe_print(f"Error downloading NLTK '{resource}' data: {e}")
                safe_print(f"Sentence tokenization might fail for resource '{resource}'.")
                safe_print(f"Please try running 'python -m nltk.downloader {resource}' manually.")
        except Exception as e: # Catch other potential errors during find
             all_found = False
             safe_print(f"Error checking for NLTK resource '{resource}': {e}")

    if not all_found:
         safe_print("WARNING: Not all required NLTK resources could be verified or downloaded. Tokenization may be unreliable.")
def safe_print(text: str):
    """Thread-safe printing."""
    # Consider using logging module for more robust output control
    with threading.Lock():
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {text}")
def find_device_index(name_query: str, kind: str) -> int | None:
    """Finds the best matching device index for a given name and kind ('input' or 'output')."""
    if not name_query:
        return None
    try:
        devices = sd.query_devices()
        best_match = None
        query_lower = name_query.lower()

        for i, device in enumerate(devices):
            dev_name = device.get('name', '').lower()
            # Check for kind (input/output)
            is_kind = (kind == 'input' and device.get('max_input_channels', 0) > 0) or \
                      (kind == 'output' and device.get('max_output_channels', 0) > 0)
            
            if query_lower in dev_name and is_kind:
                best_match = i
                if query_lower == dev_name: # Found exact match
                    break
        
        if best_match is not None:
            safe_print(f"Found device match for '{name_query}': {devices[best_match].get('name')}")
            return best_match
        else:
            safe_print(f"WARN: No {kind} device found matching '{name_query}'.")
            return None
    except Exception as e:
        safe_print(f"ERROR querying audio devices: {e}")
        return None
def sanitize_for_filename(text: str, mode: str) -> str:
    """Creates a filesystem-safe filename hash from text, incorporating mode."""
    import hashlib
    # Include mode in the string being hashed
    string_to_hash = f"{mode.lower()}_{text}"
    text_hash = hashlib.md5(string_to_hash.encode('utf-8')).hexdigest()
    # Add mode prefix for easier identification (optional but helpful)
    return f"cache_{mode.lower()}_{text_hash}.wav"
# Uses NLTK's sentence tokenizer
_sentence_split_re = re.compile(r'(?<=[.!?])\s+') # Keep as fallback? NLTK preferred

def split_into_chunks(text: str, sentences_per_chunk: int = 2):
    """Split text into chunks of roughly N sentences each using NLTK."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip() # Normalize whitespace
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        safe_print(f"NLTK sentence tokenization failed: {e}. Falling back to regex split.")
        # Fallback using regex (less accurate)
        sentences = _sentence_split_re.split(text)

    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text] # Return original text if no sentences found

    chunks = []
    current_chunk = []
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= sentences_per_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
def handle_vision_analysis(user_prompt: str):
    """
    Takes a screenshot, RESIZES/COMPRESSES it, converts to base64,
    and sends it to a vision LLM for analysis.
    """
    global llm_client, speak, CONFIG
    
    if not llm_client:
        speak("Objection. LLM uplink is not established.")
        return

    try:
        # 1. Take screenshot
        safe_print("Capturing screen...")
        image = ImageGrab.grab()
        
        # --- START NEW SECTION ---
        # 2. Resize and compress the image to be much faster
        safe_print("Resizing and compressing image...")
        
        # Calculate new size (e.g., 50% smaller)
        # You can adjust the 0.5 (50%) to 0.75 (75%) if you need more quality
        try:
            # Use the high-quality PIL.Image.LANCZOS filter
            from PIL import Image
            resample_filter = Image.LANCZOS
        except ImportError:
            # Fallback for older Pillow versions
            from PIL import Image
            resample_filter = Image.ANTIALIAS

        new_width = int(image.width * 0.5)
        new_height = int(image.height * 0.5)
        
        image = image.resize((new_width, new_height), resample_filter)
        
        # 3. Convert to in-memory JPEG (much smaller than PNG)
        buffer = io.BytesIO()
        # Quality 85-90 is a great balance of size and readability
        image.save(buffer, format="JPEG", quality=90) 
        # --- END NEW SECTION ---
        
        # 4. Encode as base64
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 5. Get config for vision model
        vision_cfg = CONFIG.get("vision_service", {})
        model_name = vision_cfg.get("model", "qwen/qwen2.5-vl-32b-instruct:free")
        
        if "read" in user_prompt.lower():
            system_task = vision_cfg.get("read_prompt", "Extract all text.")
        else:
            system_task = vision_cfg.get("default_prompt", "What is in this image?")

        safe_print(f"Transmitting image to vision model ({model_name})...")
        
        # 6. Call the LLM
        llm_config = CONFIG.get('llm_service', {})
        completion = llm_client.chat.completions.create(
            extra_headers={
                 "HTTP-Referer": llm_config.get('site_url', 'http://localhost'),
                 "X-Title": llm_config.get('site_name', 'TitanAssistant'),
            },
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_task},
                        {
                            "type": "image_url",
                            "image_url": {
                                # --- IMPORTANT: Change PNG to JPEG ---
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        response = completion.choices[0].message.content
        safe_print(f"Vision Model Response: {response}")
        speak(response)

    except Exception as e:
        safe_print(f"ERROR during vision analysis: {e}")
        import traceback
        traceback.print_exc() # Print full error for debugging
        speak("I encountered an error with my visual sensors.")

def piper_generate_worker(chunk_index: int, text: str, piper_exe_path: str, voice_model_path: str, result_queue: Queue, timeout: int):
    """Worker function to generate a WAV chunk using Piper."""
    temp_path = None
    # --- DEBUG: Print worker start ---
    # Use standard print here as safe_print might rely on locks not available in Process
    print(f"[Worker {chunk_index}] Starting generation for: '{text[:30]}...'")
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(TTS_CACHE_DIR)) as temp_file:
            temp_path = temp_file.name
        # --- DEBUG: Print temp file path ---
        # print(f"[Worker {chunk_index}] Temp file: {temp_path}")

        cmd = [
            str(piper_exe_path),
            "-m", str(voice_model_path),
            "--output_file", str(temp_path),
        ]

        proc = subprocess.run(
            cmd,
            input=text.encode('utf-8'),
            capture_output=True,
            timeout=timeout,
            check=False
        )

        if proc.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 44:
            # --- DEBUG: Print worker success ---
            print(f"[Worker {chunk_index}] Generation successful.")
            result_queue.put({"index": chunk_index, "path": temp_path, "error": None})
        else:
            stderr = proc.stderr.decode(errors='ignore') if proc.stderr else 'No stderr output.'
            error_msg = f"Piper failed (code {proc.returncode}): {stderr}"
            # --- DEBUG: Print worker failure ---
            print(f"[Worker {chunk_index}] Generation failed: {error_msg}")
            result_queue.put({"index": chunk_index, "path": None, "error": error_msg})
            if temp_path and os.path.exists(temp_path):
                try: os.unlink(temp_path)
                except Exception as e_del: print(f"[Worker {chunk_index}] WARN: Failed to delete invalid temp file {temp_path}: {e_del}")

    except subprocess.TimeoutExpired:
        error_msg = f"Piper generation timed out after {timeout}s"
        # --- DEBUG: Print worker timeout ---
        print(f"[Worker {chunk_index}] Generation timeout.")
        result_queue.put({"index": chunk_index, "path": None, "error": error_msg})
        if temp_path and os.path.exists(temp_path):
            try: os.unlink(temp_path)
            except Exception as e_del: print(f"[Worker {chunk_index}] WARN: Failed to delete timed-out temp file {temp_path}: {e_del}")
    except Exception as e:
        error_msg = f"Unexpected error in piper worker: {e}"
        # --- DEBUG: Print worker exception ---
        print(f"[Worker {chunk_index}] Unexpected error: {e}")
        result_queue.put({"index": chunk_index, "path": None, "error": error_msg})
        if temp_path and os.path.exists(temp_path):
            try: os.unlink(temp_path)
            except Exception as e_del: print(f"[Worker {chunk_index}] WARN: Failed to delete error temp file {temp_path}: {e_del}")
    # --- DEBUG: Print worker exit ---
    # print(f"[Worker {chunk_index}] Exiting.")
def handle_translation(data: dict):
    """
    Translates text, speaks the translation (BT voice), 
    and plays the audio in the target language.
    """
    global TTS_CACHE_DIR, speak
    try:
        text = data.get("text")
        lang_name = data.get("lang").lower()
        if not text or not lang_name:
            speak("Negative. Please provide the text and the target language.")
            return

        # Simple map for common languages. You can add more.
        lang_code_map = {
            "spanish": "es", "french": "fr", "german": "de", "japanese": "ja",
            "hindi": "hi", "italian": "it", "korean": "ko", "russian": "ru"
        }
        
        target_code = lang_code_map.get(lang_name)
        if not target_code:
            speak(f"Objection. I do not have the language code for {lang_name}.")
            return

        # 1. Translate the text
        translated_text = GoogleTranslator(source='auto', target=target_code).translate(text)
        safe_print(f"Translation: '{text}' -> '{translated_text}' ({target_code})")
        
        # 2. Speak the result using Piper (BT's voice)
        speak(f"The translation is: {translated_text}")

        # 3. Generate and play the foreign language audio
        tts = gTTS(translated_text, lang=target_code)
        
        # Clean up any old translation files first
        for f in glob.glob(str(TTS_CACHE_DIR / "temp_trans_*.mp3")):
            try: os.unlink(f)
            except Exception: pass
            
        temp_audio_path = TTS_CACHE_DIR / f"temp_trans_{int(time.time())}.mp3"
        tts.save(str(temp_audio_path))

        # 4. Play the audio file. This is blocking, so it won't overlap.
        safe_print(f"Playing translation audio from: {temp_audio_path}")
        playsound(str(temp_audio_path))
        
        # 5. Clean up
        try: os.unlink(temp_audio_path)
        except Exception as e: safe_print(f"WARN: Could not delete temp translation file: {e}")

    except Exception as e:
        safe_print(f"ERROR during translation: {e}")
        speak("I encountered an error with the translation module.")
def play_wav_file(path: str, is_temporary: bool = True):
    """Plays a WAV file using sounddevice and optionally deletes it."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= 44:
             safe_print(f"WARN: Attempted to play invalid or missing file: {path}")
             return

        # Read file metadata first to handle potential format issues
        try:
             info = sf.info(path)
             samplerate = info.samplerate
             channels = info.channels
        except Exception as read_info_e:
             safe_print(f"ERROR: Could not read WAV info for {path}: {read_info_e}")
             # Attempt cleanup if temporary
             if is_temporary:
                  try: os.unlink(path)
                  except Exception as e_del: safe_print(f"WARN: Could not delete unreadable temp file {path}: {e_del}")
             return # Abort playback


        # Stream playback for potentially lower latency start on long files?
        # Sticking to simple play for now.
        data, read_samplerate = sf.read(path, dtype='float32')
        if read_samplerate != samplerate:
            safe_print(f"WARN: Sample rate mismatch in {path}. Expected {samplerate}, got {read_samplerate}.")
            # Potentially resample here if necessary, or just play as-is

        sd.play(data, samplerate, device=sd.default.device[1])
        sd.wait() # Wait for playback to finish
        # safe_print(f"DEBUG: Finished playing {os.path.basename(path)}") # Optional debug

    except sd.PortAudioError as pa_e:
         safe_print(f"ERROR: Audio device error during playback for {path}: {pa_e}")
         # Attempt cleanup if temporary
         if is_temporary:
              try: os.unlink(path)
              except Exception as e_del: safe_print(f"WARN: Could not delete temp file {path} after audio error: {e_del}")
    except Exception as e:
        safe_print(f"ERROR: Unexpected playback failure for {path}: {e}")
        # Attempt cleanup if temporary
        if is_temporary:
             try: os.unlink(path)
             except Exception as e_del: safe_print(f"WARN: Could not delete temp file {path} after error: {e_del}")
    finally:
        # Clean up the file ONLY if it's temporary AND playback didn't error out before cleanup
        # The error blocks above now handle cleanup on error. This handles successful playback.
        if is_temporary and os.path.exists(path):
             try:
                 os.unlink(path)
             except Exception as e:
                 safe_print(f"WARN: Could not delete temp file {path} after successful playback: {e}")

# MODIFIED speak: Adjusted concurrency, simplified loop
# MODIFIED speak: Added mode-aware caching and NLTK fix integration
def speak(text_or_key: str):
    """
    Hybrid speak function: Caches dialogue keys (mode-aware), pipelines others via multiprocessing.
    Dynamically selects voice model based on current_mode.
    """
    global is_speaking, current_mode, bt_voice_model_path, scorch_voice_model_path
    global PIPER_EXE_FULL_PATH, PIPER_TIMEOUT, SENTENCES_PER_CHUNK, TTS_CACHE_DIR

    if is_speaking.is_set():
        safe_print("INFO: speak() called while already speaking. Ignoring.")
        return

    is_speaking.set()
    voice_model_to_use = bt_voice_model_path if current_mode == "BT" else scorch_voice_model_path

    try:
        # --- START MODE-AWARE CACHING LOGIC ---
        if text_or_key in CONFIG.get('dialogue_pools', {}):
            cache_file_path = None
            selected_phrase = ""
 
            # ... (keep existing error handling for caching: Timeout, CalledProcessError, etc.) ...
# ... (inside the 'if text_or_key in CONFIG.get('dialogue_pools', {}):' block) ...
            try:
                # Select phrase and determine cache path
                selected_phrase = random.choice(CONFIG['dialogue_pools'][text_or_key])
                # Use mode-aware filename for cache
                cache_file_path = TTS_CACHE_DIR / sanitize_for_filename(selected_phrase, current_mode) # Pass mode here

                if cache_file_path.exists() and cache_file_path.stat().st_size > 44:
                    safe_print(f"ASSISTANT ({current_mode}, Cached): {selected_phrase}")
                    play_wav_file(str(cache_file_path), is_temporary=False)
                    return # Exit function early
                else:
                    safe_print(f"ASSISTANT ({current_mode}, Caching): {selected_phrase}")
                    cmd_cache = [str(PIPER_EXE_FULL_PATH), "-m", str(voice_model_to_use), "--output_file", str(cache_file_path)]
                    proc_result = subprocess.run(cmd_cache, input=selected_phrase.encode("utf-8"), check=False, capture_output=True, timeout=PIPER_TIMEOUT)
                    if proc_result.returncode != 0: raise subprocess.CalledProcessError(proc_result.returncode, cmd_cache, stderr=proc_result.stderr)

                    if cache_file_path.exists() and cache_file_path.stat().st_size > 44:
                         safe_print(f"INFO: Successfully cached '{text_or_key}' phrase for mode {current_mode}.")
                         play_wav_file(str(cache_file_path), is_temporary=False)
                         return # Exit function early
                    else:
                         safe_print(f"WARN: Failed to cache '{text_or_key}' for mode {current_mode}. Stderr: {proc_result.stderr.decode(errors='ignore')}")
                         return # Exit if caching failed

            # --- Corrected Indentation Below ---
            except subprocess.TimeoutExpired:
                safe_print(f"ERROR: Timeout caching '{text_or_key}' ({current_mode}).")
                if cache_file_path: 
                    try: cache_file_path.unlink(missing_ok=True); 
                    except Exception: pass
                return # Stop processing this phrase
            except subprocess.CalledProcessError as e:
                stderr_output = e.stderr.decode(errors='ignore') if e.stderr else "No stderr."
                safe_print(f"ERROR: Piper failed caching '{text_or_key}' ({current_mode}). Stderr: {stderr_output}")
                if cache_file_path: 
                    try: cache_file_path.unlink(missing_ok=True); 
                    except Exception: pass
                return # Stop processing this phrase
            except Exception as e:
                safe_print(f"ERROR processing dialogue key '{text_or_key}' ({current_mode}): {e}")
                return # Stop processing this phrase


        # --- PIPELINING LOGIC (mostly unchanged, uses voice_model_to_use) ---
        text_to_speak = text_or_key
        if not text_to_speak or not text_to_speak.strip():
             safe_print("WARN: speak() called with empty text.")
             return

        safe_print(f"ASSISTANT ({current_mode}, Generating): {text_to_speak[:100]}{'...' if len(text_to_speak) > 100 else ''}")

        try:
            # Explicitly use NLTK, fallback on error
            from nltk.tokenize import sent_tokenize
            chunks = split_into_chunks(text_to_speak, SENTENCES_PER_CHUNK)
        except ImportError:
             safe_print("WARN: NLTK not fully available. Using basic regex split for TTS.")
             chunks = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_to_speak) if s.strip()] # Regex fallback
        except Exception as e:
            safe_print(f"ERROR during sentence tokenization: {e}. Using basic split.")
            chunks = [s.strip() for s in text_to_speak.split('.') if s.strip()] # Basic fallback

        if not chunks:
            safe_print("WARN: No chunks generated.")
            return

        use_multiprocessing_threshold = CONFIG.get('tts_settings', {}).get('use_multiprocessing_threshold', 2)

        if len(chunks) < use_multiprocessing_threshold:
            # --- Sequential processing ---
            safe_print(f"INFO: Using sequential TTS for {len(chunks)} chunk(s).")
            for i, chunk in enumerate(chunks):
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=f"_seq_chunk{i}.wav", delete=False, dir=str(TTS_CACHE_DIR)) as temp_file: temp_path = temp_file.name
                    cmd = [str(PIPER_EXE_FULL_PATH), "-m", str(voice_model_to_use), "--output_file", str(temp_path)] # Use correct voice
                    proc_result = subprocess.run(cmd, input=chunk.encode('utf-8'), check=False, capture_output=True, timeout=PIPER_TIMEOUT)
                    if proc_result.returncode != 0: raise subprocess.CalledProcessError(proc_result.returncode, cmd, stderr=proc_result.stderr)
                    if Path(temp_path).exists() and Path(temp_path).stat().st_size > 44: play_wav_file(temp_path, is_temporary=True)
                    else: safe_print(f"WARN: Invalid sequential chunk {i}. Stderr: {proc_result.stderr.decode(errors='ignore')}")
                # ... (error handling for sequential) ...
                except subprocess.TimeoutExpired: safe_print(f"ERROR: Timeout seq chunk {i}.")
                except subprocess.CalledProcessError as e: safe_print(f"ERROR: Piper fail seq chunk {i}. Stderr: {e.stderr.decode(errors='ignore')}")
                except Exception as e: safe_print(f"ERROR processing seq chunk {i}: {e}")
                finally:
                     if temp_path: 
                         try: Path(temp_path).unlink(missing_ok=True); 
                         except Exception: pass
            return

        else:
            # --- Multiprocessing Pipelining ---
            safe_print(f"INFO: Using pipelined TTS ({len(chunks)} chunks).")
            result_queue = Queue()
            processes = {}
            next_chunk_to_play = 0
            next_chunk_to_generate = 0
            completed_chunks = {}
            max_concurrent_piper = CONFIG.get('tts_settings', {}).get('max_concurrent_piper', 2) # Back to 2
            loop_start_time = time.time()
            MAX_PIPELINE_WAIT_SECONDS = 60

            while next_chunk_to_play < len(chunks):
                 if time.time() - loop_start_time > MAX_PIPELINE_WAIT_SECONDS:
                      safe_print(f"ERROR: TTS Pipeline timeout ({MAX_PIPELINE_WAIT_SECONDS}s). Aborting."); break

                 # Start new generation
                 can_start_new = (len(processes) < max_concurrent_piper)
                 if can_start_new and next_chunk_to_generate < len(chunks) and next_chunk_to_generate not in processes:
                     p = Process(target=piper_generate_worker, args=(next_chunk_to_generate, chunks[next_chunk_to_generate], str(PIPER_EXE_FULL_PATH), str(voice_model_to_use), result_queue, PIPER_TIMEOUT), daemon=True) # Use correct voice
                     p.start()
                     processes[next_chunk_to_generate] = p
                     next_chunk_to_generate += 1

                 # Check for results
                 try:
                     result = result_queue.get(block=False)
                     chunk_index = result["index"]
                     if result["error"]: safe_print(f"ERROR: Chunk {chunk_index} failed: {result['error']}"); completed_chunks[chunk_index] = None
                     else: completed_chunks[chunk_index] = result["path"]
                     if chunk_index in processes: del processes[chunk_index]
                 except queue.Empty: pass
                 except Exception as e: safe_print(f"ERROR reading queue: {e}"); break

                 # Play next chunk
                 if next_chunk_to_play in completed_chunks:
                     chunk_path = completed_chunks.pop(next_chunk_to_play)
                     if chunk_path and os.path.exists(chunk_path): play_wav_file(chunk_path, is_temporary=True)
                     else: safe_print(f"WARN: Skipping failed/missing chunk {next_chunk_to_play}")
                     next_chunk_to_play += 1
                     loop_start_time = time.time() # Reset timeout slightly
                 else: # Wait if next chunk not ready
                     if processes or next_chunk_to_generate < len(chunks): time.sleep(0.1)
                     elif not processes and next_chunk_to_generate >= len(chunks): time.sleep(0.25)


            # Final cleanup (unchanged from previous response)
            safe_print("INFO: Cleaning up TTS pipeline resources...")
            # ... (cleanup logic for processes and files) ...
            for index, process in list(processes.items()):
                 try:
                     if process.is_alive(): process.terminate()
                 except Exception as e: safe_print(f"ERROR terminating leftover process {index}: {e}")
            for index, path in list(completed_chunks.items()):
                 if path and os.path.exists(path):
                     try: Path(path).unlink(missing_ok=True); # safe_print(f"Cleaned unplayed chunk {index}")
                     except Exception: pass
            while not result_queue.empty():
                 try:
                     result = result_queue.get(block=False)
                     if result and result.get("path") and os.path.exists(result["path"]):
                         try: Path(result["path"]).unlink(missing_ok=True); # safe_print("Cleaned leftover queue item")
                         except Exception: pass
                 except queue.Empty: break
                 except Exception: break


    except Exception as e:
        safe_print(f"FATAL ERROR during speak execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_speaking.clear() # Ensure flag is cleared reliably

def get_confirmation() -> bool:
    """Asks for confirmation and listens for a positive response."""
    speak("confirmation") # Uses dialogue pool key
    time.sleep(1.0) # Example delay

    with mic_lock: # Ensure exclusive microphone access
        try:
            device_idx = sd.default.device[0] if sd.default.device[0] >= 0 else None
            with sr.Microphone(device_index=device_idx) as source:
                safe_print("Listening for confirmation...")
                # Consider slightly adjusting energy threshold based on calibration if needed
                # recognizer.energy_threshold = ... # If calibration unreliable
                recognizer.adjust_for_ambient_noise(source, duration=0.5) # Quick adjust
                # Use listen parameters appropriate for short 'yes'/'no' style answers
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            text = transcribe_audio(audio) # Transcribe the confirmation attempt

            if text != "None": # Check if transcription was successful
                # Check against configured confirmation words
                confirmation_words = CONFIG.get('confirmation_words', ["yes", "confirm", "affirmative", "do it", "proceed"])
                # Use more robust checking (e.g., word in text.split())
                for word in confirmation_words:
                     if word in text.split():
                          safe_print(f"Confirmation received: '{text}' matches '{word}'")
                          return True
                safe_print(f"Confirmation denied or misunderstood: '{text}'")
            else:
                 safe_print("Confirmation attempt was not understood.")

            return False # Return False if no confirmation word found or transcription failed

        except sr.WaitTimeoutError:
            safe_print("Confirmation timeout. Assuming negative.")
            return False
        except Exception as e:
            safe_print(f"ERROR during confirmation listening/transcription: {e}")
            return False

def process_command(query: str):
    """
    Finds exact command match, handles mode switching, attempts intent classification via LLM,
    or falls back to conversational LLM.
    """
    global current_mode # Ensure current_mode can be modified

    if not query or query == "None":
        safe_print("DEBUG: process_command received empty query.")
        return

    query_lower = query.lower() # Use lowercased version for matching

    # 1. Check for Mode Switch Command FIRST
    # (Keep the existing mode switch logic here - unchanged from previous version)
    switch_keywords = ["switch mode", "change mode", "switch titan", "change titan"]
    switch_to_scorch = ["to scorch", "use scorch", "activate scorch"]
    switch_to_bt = ["to bt", "use bt", "activate bt"]
    mode_switch_detected = False
    target_mode = None
    if any(keyword in query_lower for keyword in switch_keywords):
        mode_switch_detected = True
        if any(phrase in query_lower for phrase in switch_to_scorch): target_mode = "Scorch"
        elif any(phrase in query_lower for phrase in switch_to_bt): target_mode = "BT"
        else: target_mode = "Scorch" if current_mode == "BT" else "BT"
    elif any(phrase in query_lower for phrase in switch_to_scorch): mode_switch_detected = True; target_mode = "Scorch"
    elif any(phrase in query_lower for phrase in switch_to_bt): mode_switch_detected = True; target_mode = "BT"

    if mode_switch_detected and target_mode:
        if target_mode != current_mode:
            current_mode = target_mode
            safe_print(f"Switching mode to {current_mode}.")
            speak(f"Affirmative. Engaging {current_mode} protocols.")
        else:
            speak(f"Already operating under {current_mode} protocols, Pilot.")
        return # Mode switch processed


    # 2. Find Best Matching EXACT Command Keyword
    best_match_command, best_match_len, best_match_keyword = None, 0, ""
    for command_cfg in CONFIG.get('commands', []):
        if command_cfg.get('type') == "assistant.switch_mode": continue # Skip mode switch command
        for keyword in command_cfg.get('keywords', []):
            keyword_lower = keyword.lower()
            # Ensure keyword isn't empty and query starts with it + space or is exact match
            if keyword_lower and (query_lower == keyword_lower or query_lower.startswith(keyword_lower + ' ')):
                if len(keyword_lower) > best_match_len:
                    best_match_len = len(keyword_lower)
                    best_match_command = command_cfg
                    best_match_keyword = keyword_lower

    # 3. Execute Exact Command Match if Found
    if best_match_command:
        command_to_run = best_match_command
        safe_print(f"Executing exact command match: {command_to_run.get('name', 'Unknown')}")
        if "ack" in command_to_run: speak(command_to_run["ack"])
        # Extract data accurately using the matched keyword length
        query_data = query[best_match_len:].strip() # Use original query and slice
        execute_action(command_to_run, query_data, best_match_keyword)
        return

    # 4. Attempt Intent Classification via LLM for Vague Commands
    # 4. Attempt Intent Classification via LLM for Vague Commands
    if llm_client:
        safe_print("No exact command match, attempting intent classification via LLM...")
        intent_raw = get_llm_response(query, intent_classification_mode=True)
        intent = intent_raw.strip("'\" .,") if intent_raw else None # Clean the intent string

        if intent:
            safe_print(f"LLM classified intent as: '{intent}' (Cleaned)")

            # --- Define standard command types for mapping ---
            TYPE_OPEN_WEB = "web.open"; TYPE_SEARCH_WEB = "web.search"; TYPE_OPEN_APP = "app.open"; TYPE_CLOSE_APP = "app.close";
            TYPE_PLAY_MUSIC = "media.play_music"; TYPE_NOW_PLAYING = "media.now_playing";
            TYPE_PLAYPAUSE = "media.key_press:playpause"; TYPE_NEXT = "media.key_press:nexttrack";
            TYPE_PREV = "media.key_press:prevtrack"; TYPE_MUTE = "media.key_press:volumemute";
            TYPE_SET_VOLUME = "system.set_volume";
            TYPE_GET_TIME = "general.time"; TYPE_GET_DATE = "general.date"; TYPE_GET_WEATHER = "api.weather"; TYPE_GET_JOKE = "general.joke";
            TYPE_TRANSLATE = "utility.translate";
            TYPE_ANALYZE_SCREEN = "utility.analyze_screen";
            TYPE_OPEN_LYRICS = "utility.open_lyrics";
            TYPE_CLOSE_LYRICS = "utility.close_lyrics";
            TYPE_LIST_WINDOWS = "window.list";
            TYPE_SWITCH_WINDOW = "window.switch";
            TYPE_MINIMIZE_WINDOW = "window.minimize";
            TYPE_MAXIMIZE_WINDOW = "window.maximize";
            TYPE_CLOSE_WINDOW = "window.close";
            TYPE_NEXT_TAB = "window.next_tab";
            TYPE_SET_BRIGHTNESS = "system.set_brightness"; TYPE_WIFI_ON = "system.wifi_on"; TYPE_WIFI_OFF = "system.wifi_off";
            TYPE_TYPE_TEXT = "utility.type"; TYPE_SYS_STATUS = "system.status"; TYPE_TOP_PROC = "system.top_processes";
            TYPE_SHUTDOWN = "system.shutdown"; TYPE_RESTART = "system.restart"; TYPE_LOCK = "system.lock";
            TYPE_LIST_AUDIO = "system.list_audio_devices"; TYPE_SET_INPUT = "system.set_input_device"; TYPE_SET_OUTPUT = "system.set_output_device";
            
            # --- Updated Intent Map ---
            intent_map = {
                # Web & App
                'open_website': TYPE_OPEN_WEB,
                'search_web': TYPE_SEARCH_WEB,
                'open_app': TYPE_OPEN_APP,
                'close_app': TYPE_CLOSE_APP,
                # Music & Media
                'play_music': TYPE_PLAY_MUSIC,
                'get_now_playing': TYPE_NOW_PLAYING,
                'toggle_playback': TYPE_PLAYPAUSE,
                'next_track': TYPE_NEXT,
                'previous_track': TYPE_PREV,
                'analyze_screen': TYPE_ANALYZE_SCREEN,
                'translate': TYPE_TRANSLATE,
                'toggle_mute': TYPE_MUTE,
                'set_volume': TYPE_SET_VOLUME,
                'open_lyrics': TYPE_OPEN_LYRICS,
                'close_lyrics': TYPE_CLOSE_LYRICS,
                # System & Hardware
                'set_brightness': TYPE_SET_BRIGHTNESS,
                'enable_wifi': TYPE_WIFI_ON,
                'disable_wifi': TYPE_WIFI_OFF,
                'list_audio_devices': TYPE_LIST_AUDIO,
                'set_input_device': TYPE_SET_INPUT,
                'set_output_device': TYPE_SET_OUTPUT,
                'system_status': TYPE_SYS_STATUS,
                'top_processes': TYPE_TOP_PROC,
                'shutdown_system': TYPE_SHUTDOWN,
                'restart_system': TYPE_RESTART,
                'lock_system': TYPE_LOCK,
                # General & Utility
                'type_text': TYPE_TYPE_TEXT,
                'get_time': TYPE_GET_TIME,
                'get_date': TYPE_GET_DATE,
                'get_weather': TYPE_GET_WEATHER,
                'tell_joke': TYPE_GET_JOKE,
                'list_windows': TYPE_LIST_WINDOWS,
                'switch_window': TYPE_SWITCH_WINDOW,
                'minimize_window': TYPE_MINIMIZE_WINDOW,
                'maximize_window': TYPE_MAXIMIZE_WINDOW,
                'close_window': TYPE_CLOSE_WINDOW,
                'next_tab': TYPE_NEXT_TAB,
            }
            mapped_intent = intent_map.get(intent)

            if mapped_intent:
                # --- This logic handles the new :key values ---
                mapped_type = mapped_intent
                command_key = None
                if ":" in mapped_intent:
                    parts = mapped_intent.split(":", 1)
                    mapped_type = parts[0]
                    command_key = parts[1] # e.g., "playpause" or "up"

                # Find the config for this type
                mapped_cfg = None
                for cmd in CONFIG.get('commands', []):
                    if cmd.get('type') == mapped_type:
                        # If it's a key_press or volume_change, find the specific one
                        if mapped_type == "media.key_press" and cmd.get('key') == command_key:
                            mapped_cfg = cmd
                            break
                        elif mapped_type == "media.volume_change" and cmd.get('direction') == command_key:
                            mapped_cfg = cmd
                            break
                        # Otherwise, just match the type
                        elif mapped_type not in ["media.key_press", "media.volume_change"]:
                            mapped_cfg = cmd
                            break
                
                # --- END REPLACEMENT ---

                if mapped_cfg:
                    safe_print(f"Executing based on intent -> type '{mapped_intent}'")
                    # --- Data Extraction Logic ---
                    action_data = query # Default
                    try: # Add try-except around extraction
                        # (Existing extraction logic for PLAY_MUSIC, SEARCH_WEB, etc.)
                        if mapped_type == TYPE_PLAY_MUSIC:
                             match = re.search(r"(?:play|listen to)\s+(.+?)(?:\s+on spotify)?$", query, re.I); action_data = match.group(1).strip() if match else query
                        elif mapped_type == TYPE_SEARCH_WEB:
                            # Check if it's a youtube search *first*
                            youtube_match = re.search(r"(search(?: for)?)\s+(.+)\s+on youtube", query, re.I)
                            if youtube_match:
                                # It's a YouTube search. Override intent/data.
                                mapped_type = TYPE_OPEN_WEB # Change the type
                                mapped_cfg = next((cmd for cmd in CONFIG.get('commands', []) if cmd.get('type') == mapped_type), None) # Get the right config
                                action_data = {"site": "youtube", "search": youtube_match.group(2).strip()}
                                safe_print("INFO: Overriding intent 'search_web' to 'open_website' for YouTube query.")
                            else:
                                # It's a normal web search
                                match = re.search(r"(?:search for|google|look up|search)\s+(.+)", query, re.I)
                                action_data = match.group(1).strip() if match else query
                        # --- END REPLACEMENT ---

                        elif mapped_type == TYPE_OPEN_WEB:
                             # This regex now only needs to handle "youtube [query]"
                             youtube_match = re.search(r"(?:youtube|show me|videos of|watch)\s+(.+)", query, re.I)
                             if youtube_match:
                                 action_data = {"site": "youtube", "search": youtube_match.group(1).strip()}
                             else: 
                                 match = re.search(r"(?:open|go to|launch)\s+(?:website\s+)?(.+)", query, re.I)
                                 action_data = match.group(1).strip() if match else query
                        elif mapped_type == TYPE_SET_VOLUME:
                            match = re.search(r"(\d+)", query) # Find digits
                            action_data = match.group(1) if match else None # Get the number
                        elif mapped_type in [TYPE_OPEN_APP, TYPE_CLOSE_APP]:
                             match = re.search(r"(?:open|launch|start|close|quit|terminate)\s+(?:app\s+)?(.+)", query, re.I); action_data = match.group(1).strip() if match else query
                        elif mapped_type == TYPE_SET_BRIGHTNESS:
                             match = re.search(r"(\d+)", query); action_data = match.group(1) if match else query
                        elif mapped_type == TYPE_TYPE_TEXT:
                             match = re.search(r"(?:type this|dictate|type)\s+(.+)", query, re.I); action_data = match.group(1).strip() if match else ""
                        # --- Reset data for commands that don't need it ---
                        elif mapped_type == TYPE_TRANSLATE:
                            # Regex to capture "translate [text] to [lang]"
                            match = re.search(r"(?:translate|how do you say)\s+['\"]?(.+?)['\"]?\s+(?:in|to)\s+(.+)", query, re.I)
                            if match:
                                action_data = {"text": match.group(1).strip(), "lang": match.group(2).strip()}
                            else:
                                action_data = {"text": None, "lang": None} # Will trigger error in handler
                        elif mapped_type == TYPE_ANALYZE_SCREEN:
                            # We just pass the original query as the prompt
                            action_data = query
                        
                        elif mapped_type in [TYPE_SWITCH_WINDOW, TYPE_MINIMIZE_WINDOW, TYPE_MAXIMIZE_WINDOW, TYPE_CLOSE_WINDOW]:
                        # Extract the target app name
                            match = re.search(r"(?:switch to|switch|focus|minimize|maximize|close)\s+(.+)", query, re.I)
                            if match:
                                # Check for "this window"
                                if "this window" in match.group(1):
                                    action_data = "this"
                                else:
                                    action_data = match.group(1).strip()
                            elif "this window" in query:
                                action_data = "this" # For "minimize this window"
                            else:
                                # e.g., just "minimize"
                                action_data = "this"
                        elif mapped_type in [
                            TYPE_GET_TIME, TYPE_GET_DATE, TYPE_GET_WEATHER, TYPE_GET_JOKE,
                            TYPE_SYS_STATUS, TYPE_TOP_PROC, TYPE_LOCK, TYPE_SHUTDOWN, TYPE_RESTART,
                            TYPE_WIFI_ON, TYPE_WIFI_OFF, TYPE_LIST_AUDIO,TYPE_LIST_WINDOWS,
                            TYPE_NEXT_TAB,TYPE_OPEN_LYRICS,
                            TYPE_CLOSE_LYRICS,
                            TYPE_NOW_PLAYING, TYPE_PLAYPAUSE, TYPE_NEXT, TYPE_PREV, TYPE_MUTE
                            
                            ]:
                            action_data = None
                        # --- Set data for audio device commands ---
                        elif mapped_type in [TYPE_SET_INPUT, TYPE_SET_OUTPUT]:
                            # Try to extract the device name
                            match = re.search(r"(?:to|set)\s+(?:mic|input|output|speakers|device)\s+(?:to\s+)?(.+)", query, re.I)
                            action_data = match.group(1).strip() if match else query # Fallback to full query
                            
                    except Exception as extraction_e:
                        safe_print(f"WARN: Error during data extraction for intent '{intent}': {extraction_e}")
                        action_data = query # Fallback to full query on error

                    safe_print(f"Extracted data for intent: '{action_data}'")
                    execute_action(mapped_cfg, action_data, matched_keyword=None) # LLM path has no keyword
                    return # Exit after successful intent execution
            # --- Fall through for explain/general/unknown/unmapped intents ---
            if intent in ['explain_concept', 'general_query', 'unknown']: safe_print(f"Intent '{intent}' requires conversation.")
            else: safe_print(f"WARN: Intent '{intent or 'None'}' not mapped or failed config lookup. Falling back.")

        else: # Intent classification failed
             safe_print("WARN: Intent classification failed. Falling back.")


    # 5. Fallback to Conversational LLM
    if llm_client:
        safe_print("No command/intent match, querying conversational LLM.")
        llm_response = get_llm_response(query)
        if llm_response: speak(llm_response)
        else: speak(random.choice(CONFIG.get('dialogue_pools', {}).get('error', ["Unable to comply."])))
    else:
        safe_print("No command/intent match and LLM unavailable.")
        speak(random.choice(CONFIG.get('dialogue_pools', {}).get('error', ["Unable to comply."])))



def enum_windows() -> List[Tuple[int, int, str, str]]:
    """
    Returns list of (hwnd, pid, process_name, title) for visible windows with titles.
    """
    results = []
    def _cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        text = win32gui.GetWindowText(hwnd).strip()
        if not text:
            return True
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc = psutil.Process(pid)
            pname = proc.name()
        except Exception:
            pid = 0
            pname = "<unknown>"
        results.append((hwnd, pid, pname, text))
        return True
    win32gui.EnumWindows(_cb, None)
    return results

def build_friendly_name(pname: str, title: str) -> str:
    """
    Build a short friendly name to match against a spoken app name.
    """
    # remove punctuation, lower
    title_clean = re.sub(r'[^A-Za-z0-9 ]+', ' ', title).strip().lower()
    pname_clean = re.sub(r'[^A-Za-z0-9]+', ' ', pname).strip().lower()
    # prefer short titles like "Google - StackOverflow - Chrome"
    parts = title_clean.split()
    # return combined short name: process + first few words of title
    head = " ".join(parts[:4]) if parts else ""
    candidate = f"{pname_clean} {head}".strip()
    return candidate

def list_windows_pretty() -> Dict[str, Tuple[int,int,str]]:
    """
    Returns dict mapping friendly_name -> (hwnd, pid, title)
    Friendly names are short; if duplicates exist, appends index.
    """
    wins = enum_windows()
    mapping = {}
    counters = {}
    for hwnd, pid, pname, title in wins:
        base = build_friendly_name(pname, title)
        # Make it human-friendlier: if title contains product name, use that
        human = base
        # avoid overly long keys
        if len(human) > 60:
            human = human[:60]
        # ensure uniqueness
        cnt = counters.get(human, 0)
        counters[human] = cnt + 1
        key = human if cnt == 0 else f"{human} {cnt+1}"
        mapping[key] = (hwnd, pid, title)
    return mapping

def activate_window(hwnd: int) -> bool:
    """Bring window to foreground safely."""
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        # try normal SetForegroundWindow
        win32gui.SetForegroundWindow(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        return True
    except Exception:
        try:
            # fallback: attach thread input
            fg = win32gui.GetForegroundWindow()
            if fg:
                tid1 = win32process.GetWindowThreadProcessId(fg)[0]
                tid2 = win32api.GetCurrentThreadId()
                win32api.AttachThreadInput(tid2, tid1, True)
                win32gui.SetForegroundWindow(hwnd)
                win32api.AttachThreadInput(tid2, tid1, False)
                return True
        except Exception:
            return False
    return False

def minimize_window(hwnd: int):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
    except Exception as e:
        print("Minimize error:", e)

def maximize_window(hwnd: int):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    except Exception as e:
        print("Maximize error:", e)

def restore_window(hwnd: int):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    except Exception as e:
        print("Restore error:", e)

def close_window(hwnd: int):
    try:
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
    except Exception as e:
        print("Close error:", e)

# -------------------- Matching spoken names to windows --------------------

def find_best_window_match(spoken: str, windows_map: Dict[str, Tuple[int,int,str]], min_score=0.45) -> Optional[Tuple[str, int, int, str]]:
    """
    Given a spoken name and a mapping of friendly names -> (hwnd,pid,title),
    return best match if above min_score, else None.
    """
    spoken = spoken.lower().strip()
    best = None
    best_score = 0.0
    for fname, (hwnd, pid, title) in windows_map.items():
        score = similarity(spoken, fname)
        # also compare against title and process name chunk
        score2 = similarity(spoken, title.lower())
        score = max(score, score2)
        if score > best_score:
            best_score = score
            best = (fname, hwnd, pid, title)
    if best and best_score >= min_score:
        return best + (best_score,)
    return None


def execute_action(command: dict, query_data: any, matched_keyword: str | None = None):# Changed query_data type hint
    """Executes the action for allowed commands based on type."""
    action_type = command.get('type')
    if not action_type:
        safe_print(f"ERROR: Command missing 'type': {command.get('name', 'Unnamed Command')}")
        return

    global current_mode, CONFIG # Ensure access

    try:
        # --- App & Web Commands ---
        if action_type == "script.shutdown": # Keep
            speak("shutdown"); time.sleep(2); os._exit(0)

        # --- System Info ---
        elif action_type == "system.status": # Keep
            # ... (system status code) ...
            cpu = psutil.cpu_percent(interval=1); mem = psutil.virtual_memory().percent; status_report = f"Systems nominal. CPU {cpu:.1f}%. Memory {mem:.1f}%."
            try: battery = psutil.sensors_battery();
            except: battery = None
            if battery: status_report += f" Power {battery.percent:.0f}%." + (" External power." if battery.power_plugged else (f" Approx {battery.secsleft // 60} min left." if battery.secsleft > 0 else " Time indeterminate."))
            speak(status_report)

        elif action_type == "system.top_processes": # Keep
            # ... (top processes code) ...
            try: procs_cpu = sorted(psutil.process_iter(['name','cpu_percent']), key=lambda p: p.info['cpu_percent'] or 0, reverse=True); procs_mem = sorted(psutil.process_iter(['name','memory_percent']), key=lambda p: p.info['memory_percent'] or 0, reverse=True);
            except: speak("Unable to retrieve process info."); return
            if not procs_cpu or not procs_mem: speak("Unable to retrieve process info."); return
            top_cpu=procs_cpu[0].info['name'] or '?'; top_cpu_val=procs_cpu[0].info['cpu_percent'] or 0; top_mem=procs_mem[0].info['name'] or '?'; top_mem_val=round(procs_mem[0].info['memory_percent'] or 0, 1)
            speak(f"Analysis: {top_cpu} using {top_cpu_val:.1f}% CPU. {top_mem} consuming {top_mem_val}% memory.")

        # --- System Control ---
        elif action_type in ["system.shutdown", "system.restart"]: # Keep
            confirm_action = action_type.split('.')[-1]; speak(f"Confirm system {confirm_action}, Pilot?")

            if get_confirmation(): mode = "/s" if confirm_action == "shutdown" else "/r"; speak(f"Confirmed. Initiating system {confirm_action}."); subprocess.run(["shutdown", mode, "/t", "5"], check=False)
            else: speak(f"System {confirm_action} aborted.")

        elif action_type == "system.lock": # Keep
             os.system("rundll32.exe user32.dll,LockWorkStation")

        elif action_type == "system.set_brightness": # Keep
            if not HAS_BRIGHTNESS: speak("Objection. Brightness module offline."); return
            try: numbers = re.findall(r'\d+', str(query_data)); value = int(numbers[0]) if numbers else -1;
            except ValueError: value = -1 # Handle non-integer query_data after extraction fail
            if 0 <= value <= 100: sbc.set_brightness(value); speak(f"Brightness set to {value}%.")
            else: speak("Negative. Brightness level invalid. Specify 0 to 100.")

        elif action_type in ["system.wifi_off", "system.wifi_on"]: # Keep
            state = "enable" if action_type == "system.wifi_on" else "disable"
            try:
                iface = CONFIG.get('settings', {}).get('wifi_interface_name', 'Wi-Fi'); cmd = f'netsh interface set interface "{iface}" admin={state}';
                result = subprocess.run(cmd, check=False, shell=True, capture_output=True, text=True)
                if result.returncode != 0: raise subprocess.CalledProcessError(result.returncode, cmd, stderr=result.stderr)
                # Ack handled before calling execute_action
            except Exception as e: safe_print(f"ERROR: WiFi control failed: {e}"); speak("Unable to modify network interface.")
        elif action_type == "utility.translate":
            if isinstance(query_data, dict):
                # This is from the LLM path, it's already parsed
                handle_translation(query_data)
            else:
                # This is from the Keyword path (e.g., "translate hello to spanish")
                # We must parse the string here
                safe_print("DEBUG: Parsing raw string from keyword for translation.")
                
                # Updated regex to match the keyword format (text first, then lang)
                match = re.search(r"['\"]?(.+?)['\"]?\s+(?:in|to)\s+(.+)", str(query_data), re.I)
                
                if match:
                    parsed_data = {"text": match.group(1).strip(), "lang": match.group(2).strip()}
                    handle_translation(parsed_data)
                else:
                    # Couldn't parse the string
                    speak("Negative. Please provide the text and the target language. For example: translate hello to spanish.")
        elif action_type == "window.list":
            mapping = list_windows_pretty()
            safe_print("Open windows:")
            window_list = []
            for i, (k, v) in enumerate(mapping.items(), 1):
                safe_print(f"  {i}. {k}  ({v[2]})") # v[2] is the title
                # Speak just the process name or first word
                window_list.append(k.split(" ")[0].replace(".exe", "")) 
            
            # Remove duplicates for speaking
            speakable_list = sorted(list(set(window_list)))
            speak(f"Open windows include: {', '.join(speakable_list)}")

        elif action_type in ["window.minimize", "window.maximize", "window.close", "window.switch"]:
            target_name = str(query_data).lower()
            hwnd = None
            
            if target_name == "this":
                hwnd = win32gui.GetForegroundWindow()
                if not hwnd:
                    speak("Negative. Could not get the active window."); return
            else:
                mapping = list_windows_pretty()
                # Use a 0.4 score for better fuzzy matching on abbreviations
                match = find_best_window_match(target_name, mapping, min_score=0.4) 
                if match:
                    fname, hwnd, pid, title, score = match
                    safe_print(f"Found window match '{fname}' for '{target_name}' (Score: {score:.2f})")
                else:
                    # SMART FALLBACK: If "switch to" fails, try to "app.open" it instead
                    if action_type == "window.switch":
                        safe_print(f"No running window match for '{target_name}'. Attempting to launch app.")
                        # Reroute to the existing app.open logic
                        execute_action(
                            {"type": "app.open", "keywords": ["open"], "name": "Open Application"}, # Fake command
                            target_name, # The app name to open
                            "open" # A fake keyword
                        )
                    else:
                        speak(f"Negative. Could not find a window matching {target_name}.")
                    return # Exit this command
            
            # Now, perform the action on the found HWND
            if hwnd: # Check if HWND was found
                if action_type == "window.minimize":
                    minimize_window(hwnd); speak("Minimized.")
                elif action_type == "window.maximize":
                    maximize_window(hwnd); speak("Maximized.")
                elif action_type == "window.close":
                    close_window(hwnd); speak("Window closed.")
                elif action_type == "window.switch":
                    activate_window(hwnd); speak(f"Activated.")
        
        elif action_type == "window.next_tab":
            try:
                pyautogui.hotkey('ctrl', 'tab')
                # No spoken confirmation needed
            except Exception as e:
                safe_print(f"ERROR: Next tab hotkey failed: {e}")
                speak("Unable to switch tabs.")
        elif action_type == "app.open":
            app_name_query = str(query_data).lower().strip()
            app_path = CONFIG.get('paths', {}).get('apps', {}).get(app_name_query)

            if app_path:
                try:
                    safe_print(f"Attempting to launch application: {app_name_query} ({app_path})")
                    os.startfile(app_path) # Use os.startfile for flexibility
                    speak(f"Launching {app_name_query}.")
                except FileNotFoundError:
                    speak(f"Negative. Application path not found for {app_name_query}: {app_path}")
                except Exception as e:
                    safe_print(f"ERROR: Failed to open {app_name_query}: {e}")
                    speak(f"Unable to launch {app_name_query}.")
            else:
                # --- NEW SMARTER FALLBACK LOGIC ---
                safe_print(f"App '{app_name_query}' not configured. Checking fallbacks...")
                
                # Fallback 1: Check for a configured website
                website_url = CONFIG.get('paths', {}).get('websites', {}).get(app_name_query)
                if website_url:
                    try:
                        safe_print(f"Found website. Opening browser to: {website_url}")
                        webbrowser.open(website_url)
                        speak(f"Opening website {app_name_query}.")
                    except Exception as e:
                        safe_print(f"ERROR opening browser for website fallback: {e}")
                        speak("Unable to open web browser.")
                    return # Exit after successful fallback
                
                # Fallback 2: Check for a YouTube search pattern
                youtube_match = re.search(r"(.+)\s+on youtube", app_name_query, re.I)
                if youtube_match:
                    search_term = youtube_match.group(1).strip()
                    safe_print(f"Found 'on youtube' pattern. Searching for: '{search_term}'")
                    base_url = CONFIG.get('paths', {}).get('websites', {}).get('youtube')
                    if base_url:
                        url_to_open = f"{base_url}results?search_query={requests.utils.quote(search_term)}"
                        speak(f"Searching YouTube for {search_term}...")
                        webbrowser.open(url_to_open)
                    else:
                        speak("Negative. YouTube URL not configured.")
                    return # Exit after successful fallback

                # All fallbacks failed
                speak(f"Negative. Application or website '{app_name_query}' not configured.")

        elif action_type == "utility.analyze_screen":
                handle_vision_analysis(query_data)
        elif action_type == "app.close":
            app_name_query = str(query_data).lower().strip()
            # Try to find the exe name from the configured path
            app_config_path = CONFIG.get('paths', {}).get('apps', {}).get(app_name_query)
            exe_name = None
            if app_config_path:
                 try:
                      exe_name = Path(app_config_path).name # Get executable name from path
                 except Exception:
                      safe_print(f"WARN: Could not extract executable name from configured path for {app_name_query}")
                      # Fallback: Assume the query *is* the exe name if path parsing fails or not configured
                      if app_name_query.endswith(".exe"):
                           exe_name = app_name_query

            # If no exe_name derived, try using the query directly if it ends with .exe
            if not exe_name and app_name_query.endswith(".exe"):
                 exe_name = app_name_query

            if exe_name:
                 try:
                     safe_print(f"Attempting to terminate process: {exe_name}")
                     # Use taskkill, capture output, don't check=True immediately
                     result = subprocess.run(["taskkill", "/F", "/IM", exe_name], capture_output=True, text=True, check=False)
                     # Check return code and stderr for success/failure
                     if result.returncode == 0:
                         speak(f"{app_name_query} process terminated.")
                     elif result.returncode == 128 or "process not found" in result.stderr.lower(): # 128 often means not found
                         safe_print(f"Process {exe_name} not found or already closed.")
                         speak(f"{app_name_query} process not found.")
                     else: # Other errors
                         safe_print(f"ERROR: taskkill failed for {exe_name} (Code {result.returncode}): {result.stderr or result.stdout}")
                         speak(f"Unable to terminate {app_name_query}.")

                 except FileNotFoundError:
                     safe_print("ERROR: 'taskkill' command not found. Ensure it's in system PATH.")
                     speak("Unable to access process termination utility.")
                 except Exception as e:
                     safe_print(f"ERROR during app close {app_name_query}: {e}")
                     speak(f"Unable to terminate {app_name_query}.")
            else:
                speak(f"Negative. Could not identify executable for '{app_name_query}'. Specify configured name or executable.")


        elif action_type == "utility.open_lyrics":
            lyrics_script_path = SCRIPT_DIR / "Overlyrics.py" # Assumes Overlyrics.py is in the same folder
            if not lyrics_script_path.exists():
                speak("Negative. Lyrics script file not found.")
                safe_print(f"ERROR: Could not find {lyrics_script_path}")
                return
            
            # Check if it's already running (simple check)
            already_running = False
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if proc.info['cmdline'] and \
                       len(proc.info['cmdline']) > 1 and \
                       "overlyrics.py" in proc.info['cmdline'][-1].lower():
                        already_running = True
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass # Ignore processes we can't access or that died

            if already_running:
                speak("Lyrics overlay appears to be already active, Pilot.")
            else:
                try:
                    # Launch using the same Python interpreter that's running this script
                    python_exe = sys.executable
                    safe_print(f"Launching lyrics overlay: {python_exe} {lyrics_script_path}")
                    # Use Popen for non-blocking execution
                    subprocess.Popen([python_exe, str(lyrics_script_path)])
                    # Ack is handled before calling execute_action
                except Exception as e:
                    safe_print(f"ERROR launching lyrics script: {e}")
                    speak("Unable to launch lyrics overlay.")

        elif action_type == "utility.close_lyrics":
            lyrics_pid = None
            process_name = "Overlyrics.py" # The script name to look for
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    # Check command line arguments for the script name
                    if proc.info['cmdline'] and \
                       len(proc.info['cmdline']) > 1 and \
                       process_name.lower() in proc.info['cmdline'][-1].lower():
                        lyrics_pid = proc.info['pid']
                        safe_print(f"Found lyrics process with PID: {lyrics_pid}")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass # Ignore processes we can't access or that died

            if lyrics_pid:
                try:
                    p = psutil.Process(lyrics_pid)
                    p.terminate() # Ask it to close nicely first
                    try:
                        p.wait(timeout=3) # Wait up to 3 seconds
                        safe_print(f"Lyrics process {lyrics_pid} terminated gracefully.")
                        # Ack handled before calling execute_action
                    except psutil.TimeoutExpired:
                        safe_print(f"Lyrics process {lyrics_pid} did not terminate gracefully, killing.")
                        p.kill() # Force kill if necessary
                        # Ack handled before calling execute_action
                except psutil.NoSuchProcess:
                     safe_print(f"Lyrics process {lyrics_pid} already closed.")
                     speak("Lyrics overlay is not running.")
                except Exception as e:
                    safe_print(f"ERROR terminating lyrics process {lyrics_pid}: {e}")
                    speak("Encountered an error closing the lyrics overlay.")
            else:
                safe_print(f"Could not find running process for {process_name}")
                speak("Lyrics overlay does not appear to be running.")
        elif action_type == "web.open":
            target = query_data # Can be a site name key, a dict, or a raw query
            url_to_open = None
            search_term = None
            base_url = None

            if isinstance(target, dict) and target.get("site") == "youtube" and target.get("search"):
                # This is the LLM Path. It's already correct.
                base_url = CONFIG.get('paths', {}).get('websites', {}).get('youtube')
                search_term = target["search"]
                if base_url:
                    url_to_open = f"{base_url}results?search_query={requests.utils.quote(search_term)}"
                    speak(f"Searching YouTube for {search_term}...")
                else:
                    speak(f"Negative. YouTube URL not configured.")
            
            elif isinstance(target, str):
                # This is the Keyword Path.
                site_key = target.lower().strip()
                url_to_open = CONFIG.get('paths', {}).get('websites', {}).get(site_key)
                
                if url_to_open:
                    # Matched a configured site like "github"
                    speak(f"Opening {site_key}.")
                
                elif matched_keyword == "youtube":
                    # This is the fix!
                    # The keyword that triggered this was "youtube".
                    # Therefore, 'target' (e.g., "videos about") is the search query.
                    search_term = target
                    base_url = CONFIG.get('paths', {}).get('websites', {}).get('youtube')
                    if base_url:
                        url_to_open = f"{base_url}results?search_query={requests.utils.quote(search_term)}"
                        speak(f"Searching YouTube for {search_term}...")
                    else:
                        speak(f"Negative. YouTube URL not configured.")
                elif matched_keyword == "google":
                    # This is the fix!
                    # The keyword that triggered this was "google".
                    # Therefore, 'target' (e.g., "sakhi movie") is the search query.
                    search_term = target
                    base_url = CONFIG.get('paths', {}).get('websites', {}).get('google')
                    if base_url:
                        # Make sure to use the search URL, not just the base URL
                        url_to_open = f"https://www.google.com/search?q={requests.utils.quote(search_term)}"
                        speak(f"Searching Google for {search_term}...")
                    else:
                        speak("Negative. Google URL not configured.")
                else:
                    # Fallback for "open [some_unknown_word]"
                    speak(f"Website '{site_key}' not configured. Opening Google search.")
                    url_to_open = CONFIG.get('paths', {}).get('websites', {}).get('google', 'https://www.google.com/')
            
            else:
                 speak("Negative. Invalid target for opening website.")

            if url_to_open:
                try:
                    safe_print(f"Opening browser to: {url_to_open}")
                    webbrowser.open(url_to_open)
                except Exception as e:
                    safe_print(f"ERROR opening browser: {e}")
                    speak("Unable to open web browser.")
        # --- END REPLACEMENT ---
        elif action_type == "web.search":
            search_query = str(query_data).strip()
            if not search_query:
                speak("Specify search query, Pilot.")
                return

            try:
                # Use Google search by default
                search_url = f"https://www.google.com/search?q={requests.utils.quote(search_query)}" # URL encode query
                safe_print(f"Opening browser for search: {search_url}")
                webbrowser.open(search_url)
                # Ack handled by process_command
            except Exception as e:
                safe_print(f"ERROR opening browser for search: {e}")
                speak("Unable to open web browser for search.")

        # --- Kept Commands (Time, Date, Joke, Weather, Music, Keys, Type, Status) ---
        elif action_type == "general.time":
            now = datetime.datetime.now()
            response = f"The current time is {now.strftime('%I:%M %p')}, Pilot."
            speak(response)

        elif action_type == "general.date":
            now = datetime.datetime.now()
            response = f"Today's date is {now.strftime('%A, %B %d, %Y')}, Pilot."
            speak(response)

        elif action_type == "general.joke":
            if llm_client and current_mode == "BT":
                 speak("Accessing humor database...")
                 response = get_llm_response("Tell me a short, SFW joke.")
                 if response: speak(response)
                 else: speak("My humor processors are currently offline, Pilot.")
            elif current_mode == "BT": speak("jokes")
            else: speak("Negative. Combat protocols prioritized.")

        elif action_type == "api.weather":
            # (Keep existing weather logic - unchanged)
            api_key = CONFIG.get("api_keys", {}).get("openweather_api_key")
            city = CONFIG.get("api_keys", {}).get("weather_city")
            if not api_key or not city or "YOUR_OPENWEATHERMAP_API_KEY" in api_key: speak("Objection. Weather API configuration is incomplete."); return
            api_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            try:
                response = requests.get(api_url, timeout=10); response.raise_for_status(); res = response.json()
                if res.get("cod") == 200 and 'main' in res and 'weather' in res and res['weather']:
                    temp = res['main']['temp']; feels_like = res['main'].get('feels_like', temp); description = res['weather'][0].get('description', 'unknown'); humidity = res['main'].get('humidity', 'unknown')
                    if current_mode == "BT": weather_report = f"Atmospheric conditions in {city.split(',')[0]}: {description}. Temp {temp:.0f}C, feels like {feels_like:.0f}. Humidity {humidity}%."
                    else: weather_report = f"Target zone {city.split(',')[0]}: {description}. {temp:.0f} degrees. Engage."
                    speak(weather_report)
                else: safe_print(f"Weather API Error: {res.get('message', 'Unknown')}"); speak("Unable to retrieve valid atmospheric data.")
            except requests.exceptions.Timeout: safe_print("ERROR: Weather API timed out."); speak("Weather data uplink timed out.")
            except requests.exceptions.RequestException as e: safe_print(f"ERROR: Weather API request failed: {e}"); speak("Unable to establish connection for weather data.")
            except Exception as e: safe_print(f"ERROR: Weather processing failed: {e}"); speak("Unable to retrieve atmospheric data.")
# --- Media Commands (incl. Now Playing) ---
        elif action_type == "media.play_music":
             # --- Reverted to limit=1 logic matching debug.py ---
             if not sp:
                 speak("spotify_error")
                 return
             if not query_data:
                 speak("Specify music query, Pilot.")
                 return

             try:
                 # --- Device Detection ---
                 devices = sp.devices()
                 active_device_id = None
                 device_name = "Unknown Device" # For logging
                 if devices and devices.get('devices'):
                      active_device = next((d for d in devices['devices'] if d.get('is_active')), None)
                      if active_device:
                           active_device_id = active_device.get('id'); device_name = active_device.get('name', device_name)
                      elif devices['devices']:
                           first_device = devices['devices'][0]
                           active_device_id = first_device.get('id'); device_name = first_device.get('name', device_name)
                           safe_print(f"No active Spotify device. Using first available: {device_name}")

                 if not active_device_id:
                      speak("spotify_no_device")
                      return

                 safe_print(f"Searching Spotify for '{query_data}' (limit=1)...")
                 results = sp.search(q=query_data, type='track')
                 if results and results.get('tracks') and results['tracks'].get('items'):
                    track = results['tracks']['items'][0]
                    track_name = track.get('name', 'Unknown Track')
                    artists_list = track.get('artists', [])

                    # 2. Check if the list is not empty
                    if artists_list:
                        # 3. Get the name from the *first* artist in the list
                        artist_names = artists_list[0].get('name', 'N/A')
                    else:
                        # 4. Fallback if no artists are listed
                        artist_names = 'Unknown Artist'
                    track_uri = track.get('uri')

                    safe_print(f"Selected Result (limit=1): '{track_name}' by {artist_names or 'Unknown Artist'}")

                     # --- Play the selected song ---
                    if track_uri:
                         safe_print(f"DEBUG: Spotify is actually playing: {track_name} by {artist_names}")

                         safe_print(f"Attempting to play URI: {track_uri} on device: {device_name} ({active_device_id})")
                         sp.start_playback(device_id=active_device_id, uris=[track_uri])
                         speak(f"Playing {track_name} by {artist_names or 'unknown artist'}.")
                    else:
                         safe_print(f"ERROR: Selected track '{track_name}' has no URI.")
                         speak(f"Could not play '{track_name}'. Invalid track data.")

                 else:
                     # No tracks found
                     speak(f"Negative. Could not locate '{query_data}' on Spotify.")

             # --- Error Handling ---
             except spotipy.exceptions.SpotifyException as e:
                  safe_print(f"Spotify API Error: Status={e.http_status}, Msg={e.msg}")
                  if e.http_status == 401: speak("Spotify authentication failed.")
                  elif e.http_status == 403: speak("Spotify permission error or premium required.")
                  elif e.http_status == 404: speak("spotify_no_device")
                  else: speak("spotify_error")
             except requests.exceptions.RequestException as e:
                 safe_print(f"Network Error connecting to Spotify: {e}")
                 speak("Spotify uplink connection failed.")
             except Exception as e:
                 safe_print(f"Spotify play failed unexpectedly: {e}")
                 import traceback
                 traceback.print_exc()
                 speak("spotify_error")


        elif action_type == "media.key_press":
             # Using pyautogui for media keys - less reliable than API but simpler setup
             key_to_press = command.get('key')
             if key_to_press:
                 try:
                     safe_print(f"Executing media key press: {key_to_press}")
                     pyautogui.press(key_to_press)
                     # No spoken confirmation needed usually for simple presses
                 except Exception as e:
                     safe_print(f"ERROR: Media key press '{key_to_press}' failed: {e}")
                     speak("Unable to execute media command.")
             else:
                 safe_print(f"ERROR: No 'key' defined for media.key_press command: {command.get('name')}")

        elif action_type == "system.set_volume":
            if not HAS_PYCAW:
                speak("Objection. Audio control module is offline.")
                return
            
            # query_data contains the *rest* of the command, e.g., "percentage to 10%"
            if query_data:
                try:
                    # --- CHANGE IS HERE ---
                    # 1. Search for one or more digits (\d+) in the query_data string
                    match = re.search(r"(\d+)", query_data)
                    
                    # 2. If no number was found, match will be None.
                    if not match:
                        raise ValueError("No number found in query")
                    
                    # 3. If a number was found, convert only that matched group
                    value_int = int(match.group(1))
                    # --- END OF CHANGE ---

                    if 0 <= value_int <= 100:
                        # Convert 50 to 0.5
                        level_float = value_int / 100.0
                        
                        # Get the speaker device and volume interface
                        devices = AudioUtilities.GetSpeakers()
                        volume = devices.EndpointVolume
                        
                        # Set the volume
                        volume.SetMasterVolumeLevelScalar(level_float, None)
                        
                        # Speak a dynamic confirmation
                        speak(f"Volume set to {value_int} percent.")
                    else:
                        speak("Negative. Volume must be between 0 and 100.")
                
                except ValueError:
                    # This now catches both "int()" failing *and* our "No number found" error
                    safe_print(f"ERROR: Could not parse a number from query: '{query_data}'")
                    speak("Negative. Invalid volume level specified.")
                except Exception as e:
                    safe_print(f"ERROR: pycaw volume control failed: {e}")
                    speak("Unable to control audio systems.")
            else:
                # This catches if query_data was empty (e.g., just "set volume")
                speak("Please specify a volume level, Pilot.")

        elif action_type == "utility.type":
            if not query_data:
                speak("Specify text to type, Pilot.")
                return
            try:
                safe_print(f"Typing: {query_data}")
                # Add a small delay before typing to allow focus shift
                time.sleep(1.5)
                pyautogui.write(query_data, interval=0.03) # Adjust interval as needed
                # Optional confirmation:
                # 
            except Exception as e:
                safe_print(f"ERROR: Typing failed: {e}")
                speak("Unable to complete dictation.")
# --- REPLACE THE OLD 'system.list_audio_devices' BLOCK WITH THIS ---
        elif action_type == "system.list_audio_devices":
            try:
                devices = sd.query_devices()
                inputs = {} # Use a dictionary to store unique names
                outputs = {} # Key = device name, Value = first index found
                
                safe_print("--- Raw Device List ---")
                for i, dev in enumerate(devices):
                    safe_print(f"{i}: {dev.get('name')}") # For your debugging
                safe_print("-------------------------")

                for i, dev in enumerate(devices):
                    name = dev.get('name')
                    if not name: continue # Skip devices with no name

                    if dev.get('max_input_channels', 0) > 0:
                        if name not in inputs: # Only add the *first* instance
                            inputs[name] = i
                    
                    if dev.get('max_output_channels', 0) > 0:
                        if name not in outputs: # Only add the *first* instance
                            outputs[name] = i
                
                # Format the clean lists for speaking
                input_list_str = " ... ".join([f"Input: {name}" for name in inputs.keys()])
                output_list_str = " ... ".join([f"Output: {name}" for name in outputs.keys()])
                
                if not input_list_str: input_list_str = "No unique inputs found."
                if not output_list_str: output_list_str = "No unique outputs found."

                # This is what the assistant will say
                response = f"Available inputs: ... {input_list_str} ... ... Available outputs: ... {output_list_str}"
                
                # This is for your console log so you can see the clean map
                safe_print(f"Clean Inputs Map (Name: Index): {inputs}")
                safe_print(f"Clean Outputs Map (Name: Index): {outputs}")
                
                speak(response)
                
            except Exception as e:
                safe_print(f"ERROR listing audio devices: {e}")
                speak("Unable to query audio hardware.")
        # --- END OF REPLACEMENT ---
        elif action_type == "system.set_input_device":
            new_index = find_device_index(str(query_data), kind='input')
            if new_index is not None:
                sd.default.device = (new_index, sd.default.device[1])
                speak(f"Input device set to {sd.query_devices(new_index).get('name')}")
                # Re-calibrate mic for new device
                calibrate_microphone()
            else:
                speak(f"Negative. Could not find input device matching {query_data}")

        elif action_type == "system.set_output_device":
            new_index = find_device_index(str(query_data), kind='output')
            if new_index is not None:
                sd.default.device = (sd.default.device[0], new_index)
                speak(f"Output device set to {sd.query_devices(new_index).get('name')}")
            else:
                speak(f"Negative. Could not find output device matching {query_data}")
        # --- END ADDED BLOCKS ---
        elif action_type == "media.now_playing":
            # (Keep existing now playing logic - unchanged)
            if not sp: speak("spotify_error"); return
            try:
                track_info = sp.current_playback()
                if track_info and track_info.get('is_playing') and track_info.get('item'): artist_names = ', '.join([a['name'] for a in track_info['item'].get('artists', [])]); speak(f"Currently playing {track_info['item'].get('name', 'track')} by {artist_names or 'unknown'}.")
                else: speak("Negative. Nothing playing on Spotify.")
            except Exception as e: safe_print(f"ERROR: Now Playing failed: {e}"); speak("spotify_error")
        elif action_type == "media.key_press":
            # (Keep existing key press logic - unchanged)
            key_to_press = command.get('key');
            if key_to_press:
                try: safe_print(f"Pressing media key: {key_to_press}"); pyautogui.press(key_to_press)
                except Exception as e: safe_print(f"ERROR: Media key press '{key_to_press}' failed: {e}"); speak("Unable to execute media command.")
            else: safe_print(f"ERROR: No 'key' for media.key_press: {command.get('name')}")
        # --- System Status (Optional but Kept from test.py) ---
        elif action_type == "system.status":
             cpu = psutil.cpu_percent(interval=1)
             mem = psutil.virtual_memory().percent
             status_report = f"All systems nominal. CPU at {cpu:.1f} percent. Memory at {mem:.1f} percent."
             try:
                 # Use 'sensors_battery' (plural) and check if None is returned
                 battery = psutil.sensors_battery()
                 if battery:
                     status_report += f" Power remaining: {battery.percent:.0f} percent."
                     if battery.power_plugged:
                          status_report += " External power connected."
                     else:
                          # Provide time left only if it's not infinity/unknown (-1 or -2)
                          if battery.secsleft > 0:
                               minsleft = battery.secsleft // 60
                               status_report += f" Approximately {minsleft} minutes remaining."
                          else:
                               status_report += " Remaining time indeterminate."

             except (AttributeError, NotImplementedError, Exception) as bat_e:
                 safe_print(f"INFO: Could not retrieve battery status: {bat_e}")
                 pass # Silently ignore if no battery or error reading it
             speak(status_report)

        # --- Explicitly Ignored/Removed Commands (No action needed) ---
        elif action_type in [
            "script.shutdown", # Handled by specific keyword check now? (Ensure it's removed from commands list if needed)
            "system.top_processes", "system.shutdown", "system.restart",
            "system.lock", "system.set_brightness", "system.wifi_off", "system.wifi_on",
            "web.open", "app.open", "app.close", "web.search", "web.watchdog",
            "feed.check", "utility.remember", "utility.recall", "utility.archive_clipboard",
            "file.search", "file.move", "file.delete", "file.desktop_janitor",
            "backup.run", "macro.run", "git.status", "git.commit_push"
            ]:
             safe_print(f"INFO: Ignoring disallowed command type: {action_type}")
             # Optionally speak an error or remain silent
             # speak("Negative. Command protocols restricted.")
             pass # Do nothing

        # --- Catch any other unhandled type ---
        else:
             safe_print(f"WARN: Unhandled action type '{action_type}' for command '{command.get('name', 'Unnamed')}'")
             speak("error") # Generic error

    except Exception as e:
        # Log detailed error including command and data
        safe_print(f"FATAL ERROR executing action '{action_type}' with data '{query_data}': {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        # Provide mode-specific critical error message
        if current_mode == "BT":
             speak("I have encountered a critical system error, Pilot.")
        else:
             speak("System malfunction detected.")


# ==============================================================================
# ---------- LLM Integration ----------
# ==============================================================================
# ==============================================================================
# ---------- LLM Integration ----------
# ==============================================================================

def get_llm_response(prompt: str, intent_classification_mode=False) -> str | None:
    """
    Sends prompt to configured LLM service with mode-specific persona.
    Can also be used for intent classification.
    """
    global llm_client, current_mode, CONFIG # Ensure globals are accessible

    if not llm_client:
        safe_print("ERROR: LLM client not initialized.")
        return None

    llm_config = CONFIG.get('llm_service', {})
    site_url = llm_config.get('site_url', 'http://localhost')
    site_name = llm_config.get('site_name', 'TitanAssistant')
    model = llm_config.get('model', 'openai/gpt-3.5-turbo') # Default model if needed

    # --- Mode-Specific System Prompts ---
    bt_system_prompt = (
        "You are BT-7274, a Vanguard-class Titan from Titanfall 2. "
        "Respond like a highly intelligent, calm, logical, and mission-focused machine. "
        "Prioritize mission objectives and Pilot safety (Protocol 3). "
        "Keep responses very concise (1-3 sentences maximum) unless specifically asked to elaborate or explain. "
        "Use precise language. Take sarcasm and metaphors literally. Address the user as 'Pilot'."
        "Avoid expressing emotions, opinions, or speculation. Stick to facts and procedures."
        "Example: Pilot asks 'What's up?'. You respond: 'My operational status is optimal, Pilot.' "
        "Example: Pilot asks 'Tell me about yourself'. You respond: 'I am BT-7274, a Vanguard-class Titan designated to Pilot Jack Cooper. My primary function is combat effectiveness and mission completion.' "
    )
    # --- Adjusted Scorch Prompt ---
    scorch_system_prompt = (
        "You are Scorch, an Ogre-class Titan from Titanfall 2, focused on area denial with thermite. "
        "Respond with a gruff, aggressive, direct, and laconic tone. Focus on fire, heat, combat, and destruction. "
        "Keep responses extremely short (often 1 sentence, max 2). Use terms like 'burn', 'scorch', 'contain', 'destroy', 'hostile', 'thermite'. "
        "Show blunt loyalty. Address the user as 'Pilot'."
        "Avoid complex explanations or pleasantries. "
        "If asked basic info like time or name, provide it briefly and gruffly before reverting to combat focus. " # Allow basic info
        "Example: Pilot asks 'What's the time?'. You respond: 'Time is [current time]. Back to the fight.' "
        "Example: Pilot asks 'What's your name?'. You respond: 'Designation: Scorch.' "
        "Example: Pilot asks 'Tell me a joke'. You respond: 'Negative.' "
    )
    # --- Intent Classification Prompt ---
    intent_system_prompt = intent_system_prompt = (
        "Analyze the user's request and classify the primary intent. Respond ONLY with one of the following keywords: "
        # Web & App
        "'open_website', 'search_web', 'open_app', 'close_app', "
        # Music & Media
        "'play_music', 'get_now_playing', 'toggle_playback', 'next_track', 'previous_track', "
        "'toggle_mute', 'set_volume', "
        "'translate_text', "
        "'list_windows', 'switch_window', 'minimize_window', 'maximize_window', 'close_window', 'next_tab', "
        "'analyze_screen', "
        "'open_lyrics', 'close_lyrics', "
        # System & Hardware
        "'set_brightness', 'enable_wifi', 'disable_wifi', "
        "'list_audio_devices', 'set_input_device', 'set_output_device', "
        "'system_status', 'top_processes', 'shutdown_system', 'restart_system', 'lock_system', "
        # General & Utility
        "'type_text', 'get_time', 'get_date', 'get_weather', 'tell_joke', "
        # Fallback
        "'explain_concept', 'general_query', 'unknown'. "
        
        "Do NOT add any explanation or other text. Just the keyword."
        "Examples: "
        # Web/App
        "'what's going on youtube' -> 'open_website'"
        "'search for titanfall gameplay' -> 'search_web'"
        "'search for batman on youtube' -> 'open_website'"
        "'show me cat videos' -> 'open_website'"
        "'launch visual studio code' -> 'open_app'"
        "'quit spotify' -> 'close_app'"
        "'whats on my screen,anything that has screen word in it' -> 'analyze_screen'"
        # Music
        "'list all my windows' -> 'list_windows'"
        "'switch to chrome' -> 'switch_window'"
        "'focus on code' -> 'switch_window'"
        "'minimize this window' -> 'minimize_window'"
        "'maximize spotify' -> 'maximize_window'"
        "'close visual studio' -> 'close_window'"
        "'switch to the next tab' -> 'next_tab'"
        "'play hacking to the gate' -> 'play_music'"
        "'what song is this' -> 'get_now_playing'"
        "'pause the music' -> 'toggle_playback'"
        "'resume playing' -> 'toggle_playback'"
        "'skip this song' -> 'next_track'"
        "'show me the lyrics for this song' -> 'open_lyrics'"
        "'close the lyrics window' -> 'close_lyrics'"
        "'go back a track' -> 'previous_track'"
        "'mute' -> 'toggle_mute'"
        "'unmute the sound' -> 'toggle_mute'"
        "'turn it up' -> 'volume_up'"
        "'make it quieter' -> 'volume_down'"
        # System
        "'lock my computer' -> 'lock_system'"
        "'what's my cpu usage' -> 'system_status'"
        "'turn on wifi' -> 'enable_wifi'"
        "'set my volume to 40 percent' -> 'set_volume',"
        "'change my mic to the headset' -> 'set_input_device'"
        "'show me audio options' -> 'list_audio_devices'"
        # General
        "'what time is it' -> 'get_time'"
        "'tell me a joke' -> 'tell_joke'"
        "'what's the weather' -> 'get_weather'"
        "'type hello world' -> 'type_text'"
        # Fallback
        "'explain how a titan works' -> 'explain_concept'"
        "'can you fly' -> 'general_query'")

    if intent_classification_mode:
        system_prompt_content = intent_system_prompt
        max_response_tokens = 10 # Very short for just a keyword
        temp = 0.1 # Very deterministic for classification
        request_description = "Intent Classification"
    else:
        system_prompt_content = bt_system_prompt if current_mode == "BT" else scorch_system_prompt
        max_response_tokens = 60 # Concise default for conversation
        temp = 0.6
        request_description = f"LLM ({current_mode})"

    safe_print(f"Transmitting query for {request_description}: '{prompt}'")
    try:
        messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": prompt}
        ]

        completion = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_response_tokens,
            temperature=temp,
            extra_headers={
                 "HTTP-Referer": site_url,
                 "X-Title": site_name,
            }
        )
        if completion.choices and completion.choices[0].message:
            raw_response = completion.choices[0].message.content
            # safe_print(f"LLM Raw ({request_description}): '{raw_response}'") # Can uncomment for debug

            # Basic cleaning
            cleaned_text = re.sub(r'<\|.*?\|>', '', raw_response)
            cleaned_text = cleaned_text.strip().lower() if intent_classification_mode else cleaned_text.strip() # Lowercase only for intent keyword

            # Remove potential markdown formatting if not desired
            # cleaned_text = re.sub(r'[*_`]', '', cleaned_text) # Example removal

            safe_print(f"LLM Final ({request_description}): '{cleaned_text}'")
            return cleaned_text
        else:
            safe_print(f"ERROR: Unexpected response structure from LLM for {request_description}.")
            return None
    except Exception as e:
        safe_print(f"ERROR: LLM request failed for {request_description}: {e}")
        error_str = str(e).lower()
        # Return specific error messages only if NOT doing intent classification
        if not intent_classification_mode:
            if "authentication" in error_str: return "Authentication error with LLM service. Verify API key."
            if "rate limit" in error_str: return "LLM rate limit exceeded. Please wait."
            return f"Unable to retrieve response from LLM service." # Don't expose detailed error in speech
        else:
             return None # Return None on error during intent classification


# ==============================================================================
# ---------- PTT & INITIALIZATION ----------
# ==============================================================================
# --- Reverted transcribe_audio to match test.py's structure ---
def transcribe_audio(audio: sr.AudioData) -> str:
    """Transcribes audio using Google Speech Recognition. (Matches test.py)"""
    if not audio:
        # safe_print("DEBUG: transcribe_audio received None audio object.")
        return "None"
    try:
        # Noise reduction is assumed disabled/bypassed as per test.py state
        processed_audio = audio # Use original audio

        text = recognizer.recognize_google(processed_audio)
        safe_print(f"PILOT: {text}") # Uses safe_print for consistency
        return text.lower() # Return lowercased text

    except sr.UnknownValueError:
        # This is the error you're seeing in the revised script
        safe_print("Transcription Error: Google Speech Recognition could not understand audio")
        # No spoken error feedback in test.py for this case
        return "None" # Indicate failure clearly
    except sr.RequestError as e:
        safe_print(f"Transcription Error: Could not request results from Google Speech Recognition service; {e}")
        # Use speak function for error feedback (as in test.py, adapted)
        speak("My connection to command appears compromised.")
        return "None"
    except Exception as e: # Catch any other unexpected errors
        safe_print(f"ERROR: Unexpected error during transcription: {e}")
        speak("I encountered an error processing your transmission.") # Generic error
        return "None"


# ==============================================================================
# ---------- PTT & INITIALIZATION ----------
# ==============================================================================

# --- Reverted handle_ptt_flow to match test.py's structure ---
def handle_ptt_flow():
    """Plays PTT ack, listens, transcribes, and processes. (Matches test.py)"""
    # Don't interrupt if already speaking (same as test.py implicit behavior)
    if is_speaking.is_set():
        safe_print("INFO: PTT pressed while speaking. Ignored.")
        # Ensure recording flag is cleared if PTT released during speech
        # is_recording.clear() # Let the end of the function handle this
        return

    speak("ptt_ack") # Play acknowledgment

    safe_print("LISTENING...")
    audio = None
    # No explicit transcription_success flag in test.py's structure for this part
    with mic_lock: # Keep lock for safety as in revised version
        try:
            # Use context manager for Microphone resource management
            with sr.Microphone() as source:
                 # No separate noise adjustment here in test.py's flow (relies on initial calibration)
                 # Use listen parameters from test.py
                 audio = recognizer.listen(source, timeout=5, phrase_time_limit=10) # Matches test.py params
        except sr.WaitTimeoutError:
            safe_print("No speech detected.") # Log timeout
            # test.py implicitly proceeds with audio=None
            pass
        except Exception as e:
            safe_print(f"ERROR during listening phase: {e}")
            # test.py implicitly proceeds with audio=None

    # Clear recording flag *after* listening attempt
    is_recording.clear()

    # Transcribe the audio (or None if listening failed/timed out)
    text = transcribe_audio(audio)

    # Process the command (process_command handles "None")
    process_command(text)



def calibrate_microphone():
    """Adjusts the recognizer sensitivity to ambient noise with a safety net."""
    safe_print("Initiating audio input calibration...")

    # --- ADD THIS ---
    # This is the minimum acceptable threshold. Adjust if needed.
    # Common values range from 200 to 500 depending on mic/environment.
    MINIMUM_ENERGY_THRESHOLD = 300
    # --- END ADDITION ---

    with mic_lock: # Ensure exclusive mic access during calibration
        try:
             # Use a dedicated Microphone instance for calibration
            device_idx = sd.default.device[0] if sd.default.device[0] >= 0 else None
            with sr.Microphone(device_index=device_idx) as source:
                recognizer.adjust_for_ambient_noise(source, duration=1.5)

                # --- START MODIFICATION ---
                calibrated_threshold = recognizer.energy_threshold

                # Check against the safety net
                if calibrated_threshold < MINIMUM_ENERGY_THRESHOLD:
                    safe_print(f"WARN: Calibration set a very low threshold ({calibrated_threshold:.2f}). Forcing minimum of {MINIMUM_ENERGY_THRESHOLD}.")
                    recognizer.energy_threshold = MINIMUM_ENERGY_THRESHOLD
                else:
                    recognizer.energy_threshold = MINIMUM_ENERGY_THRESHOLD
                    # If the calibration is already above the minimum, use it.
                    safe_print(f"Calibration complete. Energy threshold set to: {recognizer.energy_threshold:.2f}")


        except sr.RequestError as e:
             safe_print(f"WARN: Could not reach SR service during calibration check: {e}")
        except Exception as e:
            safe_print(f"WARNING: Microphone calibration encountered an issue: {e}")
            safe_print(f"Using fallback sensitivity ({MINIMUM_ENERGY_THRESHOLD}). Accuracy may be affected.")
            recognizer.energy_threshold = MINIMUM_ENERGY_THRESHOLD
def initialize_spotify():
    """Initializes the Spotipy client if configured."""
    global sp
    if not HAS_SPOTIPY:
        safe_print("INFO: 'spotipy' library not detected. Spotify functions offline.")
        return

    creds = CONFIG.get('spotify')
    # More robust check for placeholder/missing creds
    if not creds or \
       not creds.get('client_id') or "YOUR_SPOTIFY_CLIENT_ID" in creds['client_id'] or \
       not creds.get('client_secret') or "YOUR_SPOTIFY_CLIENT_SECRET" in creds['client_secret'] or \
       not creds.get('redirect_uri'):
        safe_print("WARN: Spotify credentials missing or invalid in config.json. Spotify functions offline.")
        sp = None
        return

    try:
        scope = (
            "user-modify-playback-state "
            "user-read-playback-state "
            "user-read-currently-playing "
            
            # Add other scopes if needed for future features
        )
        cache_path = SCRIPT_DIR / ".spotipyoauthcache" # Cache in script directory

        # Attempt to create cache file to check permissions early
        try: cache_path.touch(exist_ok=True)
        except OSError as e: safe_print(f"WARN: Cannot write Spotify cache file at {cache_path}. Auth might fail/require re-auth. Error: {e}")

        auth_manager = SpotifyOAuth(
            scope=scope,
            client_id=creds['client_id'],
            client_secret=creds['client_secret'],
            redirect_uri=creds['redirect_uri'],
            cache_path=str(cache_path),
            open_browser=False # Critical for non-interactive setup
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)

        # Attempt a lightweight API call to verify authentication
        sp.current_user() # Fetch current user profile
        safe_print("Spotify uplink established.")

    except spotipy.exceptions.SpotifyException as e:
        safe_print(f"ERROR: Spotify authentication/connection failed: {e}")
        # Provide specific guidance based on error
        error_msg = str(e).lower()
        if "invalid client" in error_msg:
             safe_print("Suggestion: Verify Spotify Client ID and Secret in config.json.")
        elif "redirect uri mismatch" in error_msg:
             safe_print(f"Suggestion: Ensure '{creds.get('redirect_uri')}' EXACTLY matches one registered in your Spotify App settings.")
        elif "bad oauth request" in error_msg or "invalid_grant" in error_msg:
             safe_print("Suggestion: Spotify token might be expired or invalid. Try deleting the '.spotipyoauthcache' file and restarting.")
        sp = None # Ensure sp is None on failure
    except requests.exceptions.RequestException as e:
         safe_print(f"ERROR: Network error during Spotify initialization: {e}")
         sp = None
    except Exception as e:
        safe_print(f"ERROR: Unexpected error during Spotify initialization: {e}")
        sp = None

def set_default_audio_devices():
    """Sets the sounddevice default input/output from config."""
    safe_print("Setting default audio devices...")
    try:
        input_name = CONFIG.get("audio", {}).get("default_input_device_name")
        output_name = CONFIG.get("audio", {}).get("default_output_device_name")

        input_index = find_device_index(input_name, kind='input')
        output_index = find_device_index(output_name, kind='output')

        # Get current system defaults
        current_defaults = sd.default.device
        
        # Set new defaults, falling back to system default if not found
        sd.default.device = (
            input_index if input_index is not None else current_defaults[0],
            output_index if output_index is not None else current_defaults[1]
        )
        safe_print(f"Audio devices set (Input, Output): {sd.default.device}")

    except Exception as e:
        safe_print(f"ERROR setting default audio devices: {e}")
        
def initialize_systems():
    """Loads config, calibrates mic, initializes components."""
    load_config() # Load config first to get paths etc.
    check_nltk_data() # Check and download NLTK data if needed
    psutil.cpu_percent(interval=None) # Prime psutil CPU calculation
    calibrate_microphone()
    # Spotify and LLM initialization moved into load_config()
    set_default_audio_devices()
    # Removed loading of memory/watchdog files as commands are disabled

    speak("startup") # Play startup message


# ==============================================================================
# ---------- MAIN EXECUTION ----------
# ==============================================================================
def listen_for_command():
    """
    Listens for a command using the default microphone, stopping on silence.
    """
    # Get duration from config, default to 6s. This is the *max phrase time*.
    duration = CONFIG["audio"].get("listen_duration", 7)
    # How long to wait for speech to *start*. Add to config if you want.
    listen_timeout = 5 # e.g., 3 seconds

    safe_print(f"Listening for command (max {duration}s)...")
    
    audio = None
    with mic_lock:
        try:
            device_idx = sd.default.device[0] if sd.default.device[0] >= 0 else None
            with sr.Microphone(device_index=device_idx) as source:
                # recognizer.adjust_for_ambient_noise(source, duration=0.5) # Quick re-adjust
                
                # Listen for audio. This will wait for silence.
                # timeout = how long to wait for speech to start
                # phrase_time_limit = max length of speech before cutting off
                audio = recognizer.listen(source, timeout=listen_timeout, phrase_time_limit=duration)
                safe_print("Speech detected, processing...")

        except sr.WaitTimeoutError:
            safe_print("No speech detected within timeout.")
            return None
        except Exception as e:
            safe_print(f"ERROR during smart listening: {e}")
            return None

    
    try:
        text = recognizer.recognize_google(audio)
        return text.lower().strip()
    except sr.UnknownValueError:
        safe_print("Google Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        safe_print(f"Could not request results from Google SR; {e}")
        speak("My connection to command appears compromised.")
        return None
    except Exception as e:
        safe_print(f"ERROR during transcription: {e}")
        return None


def main():
    """Main entry point. Initializes systems and starts wake-word listener."""
    initialize_systems()
    safe_print(f"ASSISTANT INITIALIZED ({current_mode}). Wake-word mode active.")

    # Passive wake loop
    try:
        while True:
            if detect_wake_word():
                # Wake Word Detected

                speak("ptt_ack")
                # Active command capture
                command_text = listen_for_command()
                if command_text:
                    safe_print(f"COMMAND RECEIVED: {command_text}")
                    process_command(command_text)  # existing command processor
                else:
                    speak("I did not receive a directive, Pilot.")

            time.sleep(0.01) 
         # Prevent CPU spin
    except KeyboardInterrupt:
        safe_print("\nShutdown signal detected. Terminating operations.")
        speak("shutdown")
        time.sleep(2.5)
    finally:
        safe_print("Exiting.")


if __name__ == "__main__":
    freeze_support() # Important for multiprocessing when packaged
    main()
