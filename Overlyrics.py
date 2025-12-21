# -*- coding: utf-8 -*-
"""
Overlyrics v3.1 (PySide6 Redesign - Corrected)

Redesigned with:
- A two-bar (current, next) UI.
- Qt6 (PySide6) for hardware acceleration and full transparency.
- Smooth fade-in/fade-out animations for lyric transitions.
- High-performance, multi-threaded signal/slot architecture.
- Corrected thread management and cleanup.
"""

import sys
import os
import re
import time
import webbrowser
from datetime import datetime
import syncedlyrics
import spotipy
from spotipy.oauth2 import SpotifyPKCE
from tkinter import messagebox, Tk
import tkinter as tk # Keep for auth window

# --- Import PySide6 (Qt) ---
from PySide6.QtCore import (
    Qt, QThread, QObject, Signal, Slot, QTimer, QPoint, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup
)
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QMenu, QGraphicsOpacityEffect
)

# --- Constants ---
VERBOSE_MODE = False
# Polls Spotify API (slow)
TRACK_INFO_POLL_RATE = 1.0  # (seconds) How often to ask Spotify what's playing
# Updates lyrics UI (fast)
LYRIC_UPDATE_POLL_RATE = 0.05 # (seconds) Update every 50ms for smoother sync

# --- IMPORTANT: Paste your own Client ID here ---
# 1. Go to https://developer.spotify.com/dashboard/
# 2. Create an app
# 3. Paste the Client ID here
CLIENT_ID = "026011d0727c4c2db8c9b77405efa6f4" 
# 4. In your app's settings, add this as a Redirect URI
REDIRECT_URI = "https://cezargab.github.io/Overlyrics"
SCOPE = "user-read-playback-state"

# --- Helper for Auth Window (using tkinter) ---
def show_auth_code_window():
    """Uses a simple Tkinter window to get the auth code."""
    auth_code = None
    
    def on_finish():
        nonlocal auth_code
        auth_code = code_entry.get()
        auth_window.destroy()

    auth_window = Tk()
    auth_window.title("Overlyrics: Authentication")
    auth_window.geometry("500x200")
    
    label_text = (
        "Please proceed with authentication in your browser.\n"
        "Then, paste the code from the URL here."
    )
    label = tk.Label(auth_window, text=label_text, padx=10, pady=10)
    label.pack()
    
    code_entry = tk.Entry(auth_window, width=50)
    code_entry.pack(pady=10, padx=20, fill='x')
    
    finish_button = tk.Button(auth_window, text="Finish Authentication", command=on_finish)
    finish_button.pack(pady=10)

    auth_window.attributes("-topmost", True)
    # Center the window
    auth_window.update_idletasks()
    width = auth_window.winfo_width()
    height = auth_window.winfo_height()
    x = (auth_window.winfo_screenwidth() // 2) - (width // 2)
    y = (auth_window.winfo_screenheight() // 2) - (height // 2)
    auth_window.geometry(f'{width}x{height}+{x}+{y}')
    auth_window.mainloop()
    return auth_code

# --- Worker for Spotify API (slow polling) ---
class SpotifyAPIWorker(QObject):
    """
    Runs in a separate thread.
    Polls Spotify API at a slow, safe rate.
    """
    trackInfoReady = Signal(dict)
    noMusic = Signal()
    apiError = Signal(str)
    finished = Signal() # <-- FIX: Added this signal
    
    def __init__(self, auth_manager):
        super().__init__()
        self.auth_manager = auth_manager
        self.sp = None
        self._is_running = True
        self._is_paused = False # Internal state for polling rate

    @Slot()
    def run(self):
        """Authenticates and starts the polling timer."""
        try:
            cached_token = self.auth_manager.get_cached_token()
            if cached_token is None:
                raise Exception("No cached token")
            
            if self.auth_manager.is_token_expired(cached_token):
                print("[INFO] Token expired, refreshing...")
                cached_token = self.auth_manager.refresh_access_token(cached_token['refresh_token'])
            
            self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
            self.sp.current_user() # Test API call
            print("[INFO] Authenticated using cached token.")
            
        except Exception as e:
            print(f"[INFO] Cache auth failed ({e}). Proceeding with manual auth.")
            auth_url = self.auth_manager.get_authorize_url()
            webbrowser.open_new_tab(auth_url)
            
            auth_code = show_auth_code_window()
            if not auth_code:
                self.apiError.emit("Authentication cancelled by user.")
                self.finished.emit() # <-- FIX: Emit finished on auth fail
                return

            try:
                access_token = self.auth_manager.get_access_token(code=auth_code, check_cache=False)
                self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
            except Exception as auth_e:
                self.apiError.emit(f"Authentication failed: {auth_e}")
                self.finished.emit() # <-- FIX: Emit finished on auth fail
                return
        
        # Start the polling loop
        print("[INFO] SpotifyAPIWorker started.")
        while self._is_running:
            self.poll_spotify()
            
            poll_rate = TRACK_INFO_POLL_RATE
            if not self.sp or self._is_paused:
                poll_rate *= 3 # Poll slower if paused or no music
                
            # Sleep in small chunks to be responsive to stop()
            for _ in range(int(poll_rate / 0.1)):
                if not self._is_running:
                    break
                time.sleep(0.1)
        
        self.finished.emit() # <-- FIX: Emit finished when loop ends
        print("[INFO] SpotifyAPIWorker finishing.")

    @Slot()
    def poll_spotify(self):
        if not self.sp:
            return

        try:
            track_info = self.sp.current_user_playing_track()
            
            if track_info is None or track_info['item'] is None:
                self.noMusic.emit()
                self._is_paused = True
                return
            
            self._is_paused = not track_info['is_playing']
            self.trackInfoReady.emit(track_info)
            
        except spotipy.exceptions.SpotifyException as e:
            if "invalid access token" in str(e).lower():
                self.apiError.emit("Auth token expired. Please restart.")
                self.stop()
            else:
                print(f"[ERROR] Spotify API error: {e}")
        except Exception as e:
            print(f"[ERROR] Error in track_info_updater: {e}")

    @Slot()
    def stop(self):
        self._is_running = False
        print("[INFO] SpotifyAPIWorker stopping...")

# --- Worker for Lyric Syncing (fast polling) ---
class LyricSyncWorker(QObject):
    """
    Runs in a separate thread.
    Parses lyrics and runs a fast timer to find the current line.
    """
    lyricsReady = Signal(str, str) # main, next
    statusUpdate = Signal(str, str) # main, next
    finished = Signal() # <-- FIX: Added this signal
    
    def __init__(self):
        super().__init__()
        self.parsed_lyrics = []
        self.track_name = ""
        self.artist_name = ""
        self.current_progress_sec = 0.0
        self.is_paused = True
        self.last_api_call_time = 0.0
        self.last_main_verse = ""
        self._is_running = True
        self._timer_running = False

    @Slot(dict)
    def on_track_info_ready(self, track_info):
        """Receives new track info from the API worker."""
        new_track_name = track_info['item']['name']
        self.is_paused = not track_info['is_playing']
        self.current_progress_sec = track_info['progress_ms'] / 1000.0
        self.last_api_call_time = time.time()
        
        if new_track_name != self.track_name:
            print(f"[INFO] New track detected: {new_track_name}")
            self.track_name = new_track_name
            self.artist_name = track_info['item']['artists'][0]['name']
            self.parsed_lyrics = [] # Clear old lyrics
            self.last_main_verse = ""
            self.statusUpdate.emit("Loading lyrics...", "")
            self.search_for_lyrics() # Run in this thread
        
        if not self._timer_running:
            self._timer_running = True
            # Use QTimer.singleShot for a non-blocking start
            QTimer.singleShot(0, self.start_fast_poll)

    @Slot()
    def on_no_music(self):
        self.track_name = ""
        self.is_paused = True
        self.parsed_lyrics = []
        self.statusUpdate.emit("No music playing on Spotify.", "")
        
    def search_for_lyrics(self):
        """Searches for and parses lyrics."""
        try:
            search_term = f"{self.track_name} {self.artist_name}"
            lyrics = syncedlyrics.search(search_term)

            if lyrics is None or lyrics.isspace():
                self.statusUpdate.emit("No lyrics found.", "")
                self.parsed_lyrics = []
            else:
                self.parsed_lyrics = self.get_parsed_lyrics(lyrics)
                if not self.parsed_lyrics:
                    self.statusUpdate.emit("Could not parse lyrics.", "")
                    
        except Exception as e:
            print(f"[ERROR] Error in lyrics_parser: {e}")
            self.statusUpdate.emit("Error finding lyrics.", "")

    def get_parsed_lyrics(self, lyrics):
        """Parses LRC string into [(timestamp, text)]."""
        parsed_list = []
        pattern = r'\[(\d{2}:\d{2}\.\d{2})\](.+)'
        for line in lyrics.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                time_str, verse_text = match.group(1), match.group(2).strip()
                if not verse_text:
                    verse_text = "..."
                try:
                    time_obj = datetime.strptime(time_str, "%M:%S.%f")
                    seconds = (time_obj.minute * 60) + time_obj.second + (time_obj.microsecond / 1000000)
                    parsed_list.append((seconds, verse_text))
                except ValueError:
                    continue
        parsed_list.sort(key=lambda x: x[0])
        return parsed_list

    @Slot()
    def start_fast_poll(self):
        """Starts the fast QTimer loop."""
        # <-- FIX: Check for stop signal at the beginning
        if not self._is_running: 
            self.finished.emit()
            print("[INFO] LyricSyncWorker finishing.")
            return
            
        self.update_lyric_line()
        # Schedule the next call
        QTimer.singleShot(int(LYRIC_UPDATE_POLL_RATE * 1000), self.start_fast_poll)
        
    def update_lyric_line(self):
        """Finds the current lyric line based on precise time."""
        if self.is_paused or not self.parsed_lyrics or not self.track_name:
            return # Don't update if paused or no lyrics

        # Calculate precise progress
        time_since_last_poll = time.time() - self.last_api_call_time
        precise_progress = self.current_progress_sec + time_since_last_poll

        # Find the current lyric index
        current_index = -1
        for i, (timestamp, _) in enumerate(self.parsed_lyrics):
            if timestamp > precise_progress:
                break
            current_index = i
        
        # Get lyrics
        main_lyric = "..."
        next_lyric = ""

        if current_index == -1:
            main_lyric = "..."
            if self.parsed_lyrics:
                next_lyric = self.parsed_lyrics[0][1]
        else:
            main_lyric = self.parsed_lyrics[current_index][1]
            if current_index + 1 < len(self.parsed_lyrics):
                next_lyric = self.parsed_lyrics[current_index + 1][1]

        # Only emit a signal if the main lyric has changed
        if main_lyric != self.last_main_verse:
            self.last_main_verse = main_lyric
            self.lyricsReady.emit(main_lyric, next_lyric)

    @Slot()
    def stop(self):
        self._is_running = False
        self._timer_running = False
        print("[INFO] LyricSyncWorker stopping...")

# --- Main UI Window ---
class OverlyricsWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # --- Font Configuration ---
        self.main_font_size = 12
        self.next_font_size = 7   
        self.main_font_family = "Public Sans"
        self.main_font_color = "#FFFFFF" # White
        self.next_font_color = "#AAAAAA" # Gray
        
        # Check if font exists
        try:
            # Use tkinter to check font
            root = Tk()
            root.withdraw()
            tk.font.Font(family="Public Sans")
            root.destroy()
        except Exception:
            print("[WARN] 'Public Sans' not found. Falling back to Arial.")
            self.main_font_family = "Arial"

        # --- Window Flags ---
        self.setWindowFlags(
            Qt.FramelessWindowHint |       # No title bar, no borders
            Qt.WindowStaysOnTopHint    # Always on top
                  # Don't show in taskbar alt-tab
        )
        # --- Full Transparency ---
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        
        self.drag_start_position = None
        
        # --- Create Layout ---
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # --- Create Labels ---
        self.next_label = QLabel("...")
        self.next_label.setAlignment(Qt.AlignCenter)

        self.main_label = QLabel("Starting Overlyrics...")
        self.main_label.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.next_label)
        self.layout.addWidget(self.main_label)

        # Hidden alt main label for fade animation
        self.main_label_hidden = QLabel("")
        self.main_label_hidden.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.main_label_hidden)
        self.main_label_hidden.setHidden(True)

        # Make full layout centered
        self.layout.setAlignment(Qt.AlignCenter)
        # --- Animation Setup ---
        # We need two main labels to fade between
        self.main_label_hidden = QLabel("")
        self.main_label_hidden.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.main_label_hidden)
        self.main_label_hidden.setHidden(True)
        
        self.main_opacity_effect = QGraphicsOpacityEffect(self.main_label)
        self.main_label.setGraphicsEffect(self.main_opacity_effect)
        
        self.hidden_opacity_effect = QGraphicsOpacityEffect(self.main_label_hidden)
        self.main_label_hidden.setGraphicsEffect(self.hidden_opacity_effect)
        
        self.active_main_label = self.main_label
        self.inactive_main_label = self.main_label_hidden
        
        self.animation_group = QParallelAnimationGroup(self)

        self.update_font_styles()
        self.setup_workers()

    def setup_workers(self):
        """Creates and starts the background threads and workers."""
        try:
            self.auth_manager = SpotifyPKCE(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                scope=SCOPE,
                cache_handler=spotipy.CacheFileHandler(".cache_sp"),
                open_browser=False
            )
        except Exception as e:
            self.show_error(f"Failed to init auth: {e}")
            return

        # Create API Worker Thread
        self.api_thread = QThread()
        self.api_worker = SpotifyAPIWorker(self.auth_manager)
        self.api_worker.moveToThread(self.api_thread)
        self.api_thread.started.connect(self.api_worker.run)
        self.api_worker.apiError.connect(self.show_error)

        # Create Lyric Sync Worker Thread
        self.lyric_thread = QThread()
        self.lyric_worker = LyricSyncWorker()
        self.lyric_worker.moveToThread(self.lyric_thread)
        # self.lyric_thread.started.connect(self.lyric_worker.start_fast_poll) # <-- This is started by on_track_info_ready

        # Connect signals
        self.api_worker.trackInfoReady.connect(self.lyric_worker.on_track_info_ready)
        self.api_worker.noMusic.connect(self.lyric_worker.on_no_music)
        
        self.lyric_worker.lyricsReady.connect(self.on_new_lyrics)
        self.lyric_worker.statusUpdate.connect(self.on_status_update)
        
        # --- FIX: Correct Thread Cleanup ---
        # When worker emits finished, tell its thread to quit
        self.api_worker.finished.connect(self.api_thread.quit)
        self.lyric_worker.finished.connect(self.lyric_thread.quit)

        # When thread has *actually* finished, schedule worker and thread for deletion
        self.api_thread.finished.connect(self.api_worker.deleteLater)
        self.lyric_thread.finished.connect(self.lyric_worker.deleteLater)
        self.api_thread.finished.connect(self.api_thread.deleteLater)
        self.lyric_thread.finished.connect(self.lyric_thread.deleteLater)
        # ------------------------------------

        # Start threads
        self.lyric_thread.start()
        self.api_thread.start()

    @Slot(str, str)
    def on_new_lyrics(self, main_text, next_text):
        """Slot to receive new lyrics and trigger animation."""
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.raise_()
        self.show()
        self.next_label.setText(next_text)
        
        # Stop any ongoing animation
        self.animation_group.stop()
        
        # Prepare the inactive label
        self.inactive_main_label.setText(main_text)
        self.hidden_opacity_effect.setOpacity(0.0)
        self.inactive_main_label.setHidden(False)
        self.inactive_main_label.setStyleSheet(self.main_label_style) # Ensure style
        
        # Create fade-out animation for the current label
        anim_out = QPropertyAnimation(self.main_opacity_effect, b"opacity")
        anim_out.setDuration(250) # 250ms fade
        anim_out.setStartValue(1.0)
        anim_out.setEndValue(0.0)
        anim_out.setEasingCurve(QEasingCurve.InQuad)
        
        # Create fade-in animation for the new label
        anim_in = QPropertyAnimation(self.hidden_opacity_effect, b"opacity")
        anim_in.setDuration(250)
        anim_in.setStartValue(0.0)
        anim_in.setEndValue(1.0)
        anim_in.setEasingCurve(QEasingCurve.OutQuad)
        
        self.animation_group = QParallelAnimationGroup(self)
        self.animation_group.addAnimation(anim_out)
        self.animation_group.addAnimation(anim_in)
        
        # When animation finishes, swap the labels
        self.animation_group.finished.connect(self.swap_active_label)
        self.animation_group.start()

    def swap_active_label(self):
        """Swaps the active and inactive labels post-animation."""
        # Check if animation group is still running (prevents rare race condition)
        if self.animation_group.state() == QPropertyAnimation.State.Running:
            return
            
        self.active_main_label.setHidden(True)
        self.main_opacity_effect.setOpacity(1.0) # Reset for next time
        
        # Swap roles
        self.active_main_label, self.inactive_main_label = \
            self.inactive_main_label, self.active_main_label
            
        # Swap opacity effects
        self.main_opacity_effect, self.hidden_opacity_effect = \
            self.hidden_opacity_effect, self.main_opacity_effect

    @Slot(str, str)
    def on_status_update(self, main_text, next_text):
        """Slot for non-animated updates (like 'No lyrics found')."""
        self.animation_group.stop() # Stop any running animations
        self.active_main_label.setText(main_text)
        self.inactive_main_label.setHidden(True)
        self.main_opacity_effect.setOpacity(1.0)
        self.next_label.setText(next_text)
        
    @Slot(str)
    def show_error(self, message):
        """Shows a critical error and prepares to quit."""
        print(f"[CRITICAL] {message}")
        # Use Tkinter for the error box as Qt may not be fully init
        root = Tk()
        root.withdraw()
        messagebox.showerror("Overlyrics: Critical Error", message)
        root.destroy()
        self.close()

    def update_font_styles(self):
        """Applies current font sizes and colors to labels."""
        self.main_label_style = f"""
            background-color: transparent;
            color: {self.main_font_color};
            font-family: '{self.main_font_family}';
            font-size: {self.main_font_size}pt;
            font-weight: 600;
        """
        self.next_label_style = f"""
            background-color: transparent;
            color: {self.next_font_color};
            font-family: '{self.main_font_family}';
            font-size: {self.next_font_size}pt;
            font-weight: 400;
        """
        self.next_label.setStyleSheet(self.next_label_style)
        self.active_main_label.setStyleSheet(self.main_label_style)
        self.inactive_main_label.setStyleSheet(self.main_label_style)
        
        self.adjustSize() # Fit window to new text size
        self.update()

    def resize_font(self, delta):
        self.main_font_size = max(8, self.main_font_size + delta)
        self.next_font_size = max(6, int(self.main_font_size * 0.7))
        self.update_font_styles()

    # --- Window Controls (Right-click menu, Dragging) ---

    def contextMenuEvent(self, event):
        """Creates the right-click context menu."""
        menu = QMenu(self)
        
        increase_font_action = QAction("Increase Font Size", self)
        increase_font_action.triggered.connect(lambda: self.resize_font(2))
        
        decrease_font_action = QAction("Decrease Font Size", self)
        decrease_font_action.triggered.connect(lambda: self.resize_font(-2))
        
        quit_action = QAction("Quit Overlyrics", self)
        quit_action.triggered.connect(self.close)
        
        menu.addAction(increase_font_action)
        menu.addAction(decrease_font_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        
        menu.exec(event.globalPos())

    def mousePressEvent(self, event):
        """Captures mouse press for dragging."""
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """Handles window dragging."""
        if event.buttons() == Qt.LeftButton and self.drag_start_position:

            self.move(event.globalPosition().toPoint() - self.drag_start_position)
            event.accept()

    def closeEvent(self, event):
        """Handles application quit."""
        print("[INFO] Quitting app...")
        if hasattr(self, 'api_worker'):
            self.api_worker.stop()
        if hasattr(self, 'lyric_worker'):
            self.lyric_worker.stop()
        

        time.sleep(0.2)
        
        if hasattr(self, 'api_thread'):
            self.api_thread.quit()
        if hasattr(self, 'lyric_thread'):
            self.lyric_thread.quit()
            
        print("[INFO] Cleanup complete. Exiting.")
        QApplication.quit()


if __name__ == "__main__":
    

    os.environ["QT_HIGH_DPI_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"
    
    app = QApplication(sys.argv)
    
    # Check for client ID
    if CLIENT_ID == "YOUR_OWN_CLIENT_ID_GOES_HERE":
        root = Tk()
        root.withdraw()
        messagebox.showerror(
            "Overlyrics: Configuration Error",
            "Please add your own Spotify Client ID to the python script (line 53) to continue."
        )
        root.destroy()
        sys.exit()

    window = OverlyricsWindow()
    window.adjustSize()  
    window.repaint()
    window.raise_()
    screen = app.primaryScreen()
    geo = screen.geometry()
    avail = screen.availableGeometry()

    taskbar_height = geo.height() - avail.height()

    window_width = window.width()
    window_height = window.height()

    x = (geo.width() - window_width) // 2
    y = geo.height() - window_height - (taskbar_height // 2)

    print(f"[INFO] Positioning window at ({x}, {y})")
    window.move(x, y)

    window.show()
    sys.exit(app.exec())