# audio/asr_streaming.py
# Continuous streaming ASR using two background threads.
# Thread 1 (_listen_loop): records audio chunks via pyaudio (or custom record_fn for NAO mic)
# Thread 2 (_process_loop): transcribes chunks via local faster-whisper
# Speech detection uses Silero VAD (neural network) instead of RMS energy.

import os
import struct
import threading
import time
from collections import deque

import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper import WhisperModel

CHUNK_DURATION_SECONDS = 2
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16 = 2 bytes

# Load models once at module level
print("[STREAMING_ASR] Loading Silero VAD model...")
_vad_model = load_silero_vad()
print("[STREAMING_ASR] Loading faster-whisper model (small.en)...")
_whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")
print("[STREAMING_ASR] Models loaded.")


class StreamingTranscriber:
    """
    Records audio continuously in background threads and transcribes locally.

    Uses Silero VAD for speech detection and faster-whisper for transcription.
    No API calls needed — everything runs locally.

    Usage:
        transcriber = StreamingTranscriber()     # PC mic (pyaudio)
        transcriber = StreamingTranscriber(record_fn=robot.record_from_nao)  # NAO mic

        transcriber.start()
        # ... later ...
        text = transcriber.get_recent_text(last_n_chunks=2)   # last ~6s
        full = transcriber.get_session_text()
        transcriber.pause()     # while robot speaks
        robot.speak(reply)
        transcriber.resume()
        transcriber.clear_session()   # reset after interruption
        transcriber.stop()
    """

    def __init__(self, chunk_duration=CHUNK_DURATION_SECONDS,
                 sample_rate=SAMPLE_RATE, record_fn=None):
        """
        Args:
            chunk_duration: seconds per audio recording chunk (default 2)
            sample_rate: sample rate in Hz (default 16000)
            record_fn: optional callable(duration_sec: float, sample_rate: int) -> bytes
                       If None, uses pyaudio on default system mic.
                       If provided (e.g. NAO mic), must return raw int16 PCM bytes.
        """
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self._record_fn = record_fn

        # Queue of raw PCM byte strings waiting to be transcribed
        self._audio_queue = deque()
        self._queue_lock = threading.Lock()
        self._queue_event = threading.Event()

        # Transcription results — 30 chunks × 2s = 60s rolling window
        self.recent_transcripts = deque(maxlen=30)
        self._transcripts_lock = threading.Lock()

        self.current_session_text = ""
        self._session_lock = threading.Lock()

        # Silence tracking for turn-end detection
        self._last_chunk_silent = True
        self._silence_lock = threading.Lock()
        self._consecutive_silent_chunks = 0

        # Self-hearing filter: stores the robot's last spoken text
        self._last_robot_text = ""
        self._robot_text_lock = threading.Lock()

        # Pause/stop control
        self._paused = False
        self._paused_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._listen_thread = None
        self._process_thread = None

        # Optional callback fired on each new non-empty transcript chunk
        self._on_chunk_callback = None

        # Fallback tracking: switch to PC mic if custom record_fn keeps failing
        self._consecutive_empty = 0
        self._using_fallback = False
        self._FALLBACK_THRESHOLD = 3  # switch after this many consecutive empty returns

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def start(self):
        """Launch both background threads."""
        self._stop_event.clear()
        self._listen_thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="asr-listen"
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="asr-process"
        )
        self._listen_thread.start()
        self._process_thread.start()
        print("[STREAMING_ASR] Started. chunk={}s, rate={}Hz, custom_mic={}".format(
            self.chunk_duration, self.sample_rate, self._record_fn is not None
        ))

    def pause(self):
        """Freeze recording (call before robot speaks to avoid self-feedback)."""
        with self._paused_lock:
            self._paused = True

    def resume(self):
        """Resume recording after robot finishes speaking.
        Drains any audio queued during the pause to prevent self-hearing.
        Only flushes the queue when actually transitioning from paused state."""
        with self._paused_lock:
            was_paused = self._paused
            self._paused = False
        if was_paused:
            with self._queue_lock:
                self._audio_queue.clear()

    def stop(self):
        """Signal threads to stop and wait for them."""
        self._stop_event.set()
        self._queue_event.set()   # unblock process loop if waiting
        if self._listen_thread:
            self._listen_thread.join(timeout=2)
        if self._process_thread:
            self._process_thread.join(timeout=8)
        print("[STREAMING_ASR] Stopped.")

    # ------------------------------------------------------------------
    # Public data access
    # ------------------------------------------------------------------

    def get_recent_text(self, last_n_chunks=2):
        """Return the joined text of the last N transcribed chunks (~6s for n=2)."""
        with self._transcripts_lock:
            chunks = list(self.recent_transcripts)
        tail = chunks[-last_n_chunks:] if len(chunks) >= last_n_chunks else chunks
        return " ".join(tail).strip()

    def get_session_text(self):
        """Return all accumulated transcription text since last clear_session()."""
        with self._session_lock:
            return self.current_session_text.strip()

    def get_rolling_window(self, max_seconds=45):
        """Return the last N seconds of speaker transcript as a single string.
        Unlike get_session_text(), this is NOT cleared after interrupts."""
        with self._transcripts_lock:
            chunks = list(self.recent_transcripts)
        n_chunks = max(1, max_seconds // self.chunk_duration)
        tail = chunks[-n_chunks:] if len(chunks) >= n_chunks else chunks
        return " ".join(tail).strip()

    def flush_audio_only(self):
        """Drain audio queue without clearing transcript history.
        Call after robot finishes speaking to discard self-hearing audio
        while preserving the rolling window for the evaluation loop."""
        with self._queue_lock:
            self._audio_queue.clear()

    def set_on_chunk_callback(self, fn):
        """Set a callback fired when a new non-empty transcript chunk arrives.
        Signature: fn(text: str) -> None"""
        self._on_chunk_callback = fn

    def is_using_fallback(self):
        """Return True if PC mic fallback is active (NAO mic failed)."""
        return self._using_fallback

    def clear_session(self):
        """Reset accumulated transcript and flush pending audio. Call after each robot turn."""
        with self._queue_lock:
            self._audio_queue.clear()
        with self._session_lock:
            self.current_session_text = ""
        with self._transcripts_lock:
            self.recent_transcripts.clear()
        with self._silence_lock:
            self._last_chunk_silent = True
            self._consecutive_silent_chunks = 0

    def get_last_chunk_was_silent(self):
        """Return True if the most recently processed audio chunk was silent.
        Use this to detect end-of-turn: user spoke then went quiet."""
        with self._silence_lock:
            return self._last_chunk_silent

    def get_consecutive_silent_chunks(self):
        """Return the number of consecutive silent chunks since last speech."""
        with self._silence_lock:
            return self._consecutive_silent_chunks

    def set_last_robot_text(self, text):
        """Store the robot's last spoken text so we can filter self-hearing."""
        with self._robot_text_lock:
            self._last_robot_text = text.lower().strip() if text else ""

    def _is_self_hearing(self, transcript):
        """Check if a transcript chunk is just the robot's own voice being picked up."""
        with self._robot_text_lock:
            robot_text = self._last_robot_text
        if not robot_text or not transcript:
            return False
        chunk = transcript.lower().strip()
        # Check if most of the chunk's words appear in the robot's last utterance
        chunk_words = chunk.split()
        if not chunk_words:
            return False
        robot_words = set(robot_text.split())
        matches = sum(1 for w in chunk_words if w in robot_words)
        ratio = matches / len(chunk_words)
        if ratio >= 0.6:
            print("[STREAMING_ASR] Dropped self-hearing ({:.0%} match): {}".format(
                ratio, transcript[:80]))
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_paused(self):
        with self._paused_lock:
            return self._paused

    # ------------------------------------------------------------------
    # Thread 1: listen loop
    # ------------------------------------------------------------------

    def _listen_loop(self):
        """
        Continuously records audio chunks. Skips recording while paused.
        Appends raw PCM bytes to _audio_queue for the process thread.

        If a custom record_fn (NAO mic) returns empty bytes repeatedly,
        automatically falls back to pyaudio (PC mic).
        """
        pa = None  # lazy-init pyaudio only when needed

        try:
            while not self._stop_event.is_set():
                if self._is_paused():
                    time.sleep(0.1)
                    continue

                try:
                    if self._record_fn is not None and not self._using_fallback:
                        # Custom recording function (e.g., NAO mic via SSH/SFTP)
                        pcm_bytes = self._record_fn(self.chunk_duration, self.sample_rate)

                        if not pcm_bytes:
                            self._consecutive_empty += 1
                            if self._consecutive_empty >= self._FALLBACK_THRESHOLD:
                                print("[STREAMING_ASR] NAO mic failed {} times, "
                                      "falling back to PC mic.".format(
                                          self._consecutive_empty))
                                self._using_fallback = True
                                self._consecutive_empty = 0
                            continue
                        else:
                            self._consecutive_empty = 0  # reset on success
                    else:
                        # PC mic (pyaudio) — either default or fallback
                        if pa is None:
                            try:
                                import pyaudio
                                pa = pyaudio.PyAudio()
                                print("[STREAMING_ASR] PC microphone initialized.")
                            except ImportError:
                                print("[STREAMING_ASR] pyaudio not installed. "
                                      "Cannot record audio.")
                                return
                        pcm_bytes = self._record_pyaudio(pa)

                    if pcm_bytes and not self._is_paused():
                        # Run VAD here (fast, <50ms) for instant silence detection
                        has_speech = self._contains_speech(pcm_bytes)
                        if has_speech:
                            with self._silence_lock:
                                self._last_chunk_silent = False
                                self._consecutive_silent_chunks = 0
                            # Only queue speech chunks for transcription
                            with self._queue_lock:
                                self._audio_queue.append(pcm_bytes)
                            self._queue_event.set()
                        else:
                            with self._silence_lock:
                                self._last_chunk_silent = True
                                self._consecutive_silent_chunks += 1

                except Exception as e:
                    print("[STREAMING_ASR] Listen error:", e)
                    time.sleep(0.5)
        finally:
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass

    def _record_pyaudio(self, pa):
        """Record one chunk using pyaudio. Returns raw int16 PCM bytes."""
        import pyaudio
        frames_per_buffer = 1024
        total_frames_needed = int(self.sample_rate * self.chunk_duration)

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=frames_per_buffer,
        )
        frames = []
        collected = 0
        try:
            while collected < total_frames_needed:
                if self._stop_event.is_set() or self._is_paused():
                    break
                chunk = stream.read(frames_per_buffer, exception_on_overflow=False)
                frames.append(chunk)
                collected += frames_per_buffer
        finally:
            stream.stop_stream()
            stream.close()

        return b"".join(frames)

    # ------------------------------------------------------------------
    # Thread 2: process loop
    # ------------------------------------------------------------------

    def _process_loop(self):
        """
        Pulls speech-only PCM chunks from the queue (VAD already done in listen loop)
        and transcribes with local faster-whisper.
        """
        while not self._stop_event.is_set():
            # Block until audio is available (or stop signal)
            self._queue_event.wait(timeout=2.0)
            self._queue_event.clear()

            while True:
                with self._queue_lock:
                    if not self._audio_queue:
                        break
                    pcm_bytes = self._audio_queue.popleft()

                text = self._transcribe_pcm(pcm_bytes)
                if text:
                    with self._transcripts_lock:
                        self.recent_transcripts.append(text)
                    with self._session_lock:
                        self.current_session_text += " " + text
                    # Notify listeners that new transcript content arrived
                    if self._on_chunk_callback:
                        try:
                            self._on_chunk_callback(text)
                        except Exception:
                            pass

    # Known Whisper hallucinations on silent/noisy audio
    _HALLUCINATIONS = {
        "thank you for watching",
        "thanks for watching",
        "thank you",
        "you",
        "www.youtube.com",
        "subscribe",
        "like and subscribe",
        "see you in the next video",
        "see you next time",
        "i'll see you in the next video",
        "bye",
        "bye bye",
        "goodbye",
        ".",
        "..",
        "...",
        "....",
        ".....",
        "ha",
        "haha",
        "hahaha",
        "grr",
        "grrr",
        "grrrr",
        "hmm",
        "hm",
        "uh",
        "um",
    }

    def _contains_speech(self, pcm_bytes):
        """Return True if Silero VAD detects human speech in the audio chunk."""
        if len(pcm_bytes) < 2:
            return False
        num_samples = len(pcm_bytes) // 2
        samples = struct.unpack("<" + "h" * num_samples, pcm_bytes[:num_samples * 2])
        # Convert int16 samples to float32 tensor normalized to [-1, 1]
        tensor = torch.FloatTensor(samples) / 32768.0
        timestamps = get_speech_timestamps(tensor, _vad_model, sampling_rate=self.sample_rate)
        has_speech = len(timestamps) > 0
        print("[STREAMING_ASR] VAD: {}".format("SPEECH" if has_speech else "silent"))
        return has_speech

    def _transcribe_pcm(self, pcm_bytes):
        """
        Transcribes a speech chunk with faster-whisper (VAD already done in listen loop).
        Returns transcript string, or "" on error.
        """
        if not pcm_bytes:
            return ""

        # Convert PCM bytes to float32 numpy array for faster-whisper
        num_samples = len(pcm_bytes) // 2
        samples = struct.unpack("<" + "h" * num_samples, pcm_bytes[:num_samples * 2])
        audio_array = np.array(samples, dtype=np.float32) / 32768.0

        try:
            segments, info = _whisper_model.transcribe(
                audio_array,
                language="en",
                beam_size=1,
                vad_filter=False,
            )
            text = " ".join(seg.text for seg in segments).strip()

            # Drop known Whisper hallucinations
            normalized = text.lower().strip(".,!? ")
            if normalized in self._HALLUCINATIONS:
                print("[STREAMING_ASR] Dropped hallucination:", text[:80])
                return ""
            # Filter out robot's own voice being picked up by the mic
            if text and self._is_self_hearing(text):
                return ""
            if text:
                print("[STREAMING_ASR] Chunk:", text[:80])
            return text
        except Exception as e:
            print("[STREAMING_ASR] Whisper error:", e)
            return ""
