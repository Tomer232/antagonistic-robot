# session_logger.py — Non-blocking, queue-based session logging system.
# Creates a timestamped directory per session with structured log files.
# Logging never blocks the audio or interrupt pipeline.

import os
import json
import time
import queue
import threading
import traceback
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_LOG_FILE = os.path.join(BASE_DIR, "conversation_log.jsonl")
LOGS_DIR = os.path.join(BASE_DIR, "logs")


class SessionLogger:
    """Non-blocking session logger with dedicated worker thread.

    Produces four log files per session inside logs/session_YYYYMMDD_HHMMSS/:
        session.log      — human-readable chronological log
        timing.jsonl     — machine-readable latency checkpoints
        transcript.jsonl — full transcript with role and timestamps
        errors.log       — exceptions with full stack traces
    """

    def __init__(self, session_id=None):
        self.session_id = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(LOGS_DIR, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)

        self._session_log_path = os.path.join(self.session_dir, "session.log")
        self._timing_path = os.path.join(self.session_dir, "timing.jsonl")
        self._transcript_path = os.path.join(self.session_dir, "transcript.jsonl")
        self._errors_path = os.path.join(self.session_dir, "errors.log")

        # Open file handles
        self._session_fh = open(self._session_log_path, "a", encoding="utf-8")
        self._timing_fh = open(self._timing_path, "a", encoding="utf-8")
        self._transcript_fh = open(self._transcript_path, "a", encoding="utf-8")
        self._errors_fh = open(self._errors_path, "a", encoding="utf-8")

        # Non-blocking queue — drop entries rather than block the pipeline
        self._queue = queue.Queue(maxsize=1000)
        self._stop_event = threading.Event()

        # Tracks last checkpoint timestamp per interrupt_id for elapsed_ms calculation
        self._interrupt_timings = {}
        self._timings_lock = threading.Lock()

        # Worker thread
        self._worker = threading.Thread(target=self._flush_loop, daemon=True,
                                        name="session-logger")
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API — all methods are non-blocking
    # ------------------------------------------------------------------

    def log(self, message):
        """Log a human-readable message to session.log."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = "[{}] {}\n".format(ts, message)
        self._enqueue(("session", line))

    def log_timing(self, interrupt_id, checkpoint):
        """Log a latency checkpoint to timing.jsonl.

        Returns the current timestamp so callers can chain checkpoints.
        """
        t = time.time()

        with self._timings_lock:
            prev_t = self._interrupt_timings.get(interrupt_id)
            self._interrupt_timings[interrupt_id] = t

        elapsed_ms = round((t - prev_t) * 1000, 1) if prev_t else 0.0

        entry = {
            "session_id": self.session_id,
            "interrupt_id": interrupt_id,
            "checkpoint": checkpoint,
            "t": round(t, 3),
            "elapsed_ms": elapsed_ms,
        }
        self._enqueue(("timing", json.dumps(entry, ensure_ascii=False) + "\n"))

        # Also log to session.log for human readability
        self.log("[TIMING] {} | {} | elapsed={:.1f}ms".format(
            interrupt_id, checkpoint, elapsed_ms))

        return t

    def log_transcript(self, role, text, member_name=None):
        """Log a transcript entry to transcript.jsonl and legacy conversation_log.jsonl."""
        entry = {
            "role": role,
            "text": text,
            "ts": round(time.time(), 6),
        }
        if member_name:
            entry["member"] = member_name

        self._enqueue(("transcript", json.dumps(entry, ensure_ascii=False) + "\n"))

        # Backward compat: also append to conversation_log.jsonl (control_server reads this)
        legacy_entry = {
            "speaker": "user" if role == "speaker" else "robot",
            "text": text,
            "ts": entry["ts"],
        }
        if member_name:
            legacy_entry["member"] = member_name
        self._enqueue(("legacy", json.dumps(legacy_entry, ensure_ascii=False) + "\n"))

    def log_error(self, exception, context=""):
        """Log an exception with full traceback to errors.log."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        tb = traceback.format_exc()
        block = "[{}] {} | {}\n{}\n\n".format(ts, context, str(exception), tb)
        self._enqueue(("error", block))

        # Also to session.log
        self.log("[ERROR] {} | {}".format(context, str(exception)))

    def log_eval_decision(self, interrupt_id, transcript_window, grok_response,
                          decision):
        """Log an interrupt evaluation decision (including 'no interrupt')."""
        did_interrupt = decision.get("interrupt", False)
        speech = decision.get("speech", "")
        tag = "INTERRUPT" if did_interrupt else "NO_INTERRUPT"

        self.log("[EVAL] {} | {} | transcript_len={} | grok_raw={} | speech={}".format(
            interrupt_id, tag, len(transcript_window),
            repr(grok_response)[:200], repr(speech)[:120]))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Flush remaining queue entries and close all file handles."""
        self._stop_event.set()
        self._worker.join(timeout=5)

        # Drain anything left
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                self._write_item(item)
            except queue.Empty:
                break

        for fh in (self._session_fh, self._timing_fh, self._transcript_fh,
                    self._errors_fh):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enqueue(self, item):
        """Put item on queue. Drop silently if full (never block)."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass

    def _flush_loop(self):
        """Worker thread: drain queue and write to disk."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
                self._write_item(item)
            except queue.Empty:
                continue
            except Exception:
                pass  # logging must never crash

    def _write_item(self, item):
        """Write a single queued item to the appropriate file."""
        log_type, data = item
        try:
            if log_type == "session":
                self._session_fh.write(data)
                self._session_fh.flush()
            elif log_type == "timing":
                self._timing_fh.write(data)
                self._timing_fh.flush()
            elif log_type == "transcript":
                self._transcript_fh.write(data)
                self._transcript_fh.flush()
            elif log_type == "error":
                self._errors_fh.write(data)
                self._errors_fh.flush()
            elif log_type == "legacy":
                with open(LEGACY_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(data)
        except Exception:
            pass  # logging must never crash
