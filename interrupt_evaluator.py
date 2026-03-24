# interrupt_evaluator.py
# Daemon thread that periodically evaluates the rolling transcript via Grok
# and fires content-driven interrupts when warranted.

import threading
import time


SILENCE_THRESHOLD_SECONDS = 15  # prompt speaker after this many seconds of silence
MIN_TRANSCRIPT_WORDS = 10       # don't evaluate unless transcript has enough content


class InterruptEvaluator(threading.Thread):
    """Periodic Grok evaluation loop running as a daemon thread.

    Every `eval_interval` seconds (while not paused), sends the rolling
    transcript window to Grok and asks whether the active personality
    should interrupt. If yes, fires `on_interrupt_callback`.

    Also handles silence prompting: if the speaker is silent for 15+ seconds,
    asks Grok whether to nudge them.
    """

    def __init__(self, transcriber, audience_manager, settings, logger,
                 on_interrupt_callback):
        """
        Args:
            transcriber: StreamingTranscriber instance
            audience_manager: AudienceManager instance
            settings: dict — session settings
            logger: SessionLogger instance
            on_interrupt_callback: fn(speech_text: str, interrupt_id: str)
                called when Grok decides to interrupt
        """
        super().__init__(daemon=True, name="interrupt-evaluator")
        self.transcriber = transcriber
        self.audience_manager = audience_manager
        self.settings = settings
        self.logger = logger
        self.on_interrupt_callback = on_interrupt_callback

        self.eval_interval = settings.get("eval_interval_seconds", 5)

        self._paused = False
        self._paused_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._interrupt_counter = 0
        self._counter_lock = threading.Lock()

        # Silence tracking — updated via chunk callback
        self._last_speech_time = time.time()
        self._speech_time_lock = threading.Lock()
        self._silence_prompt_sent = False

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def pause(self):
        """Pause evaluation (call while robot is speaking)."""
        with self._paused_lock:
            self._paused = True

    def resume(self):
        """Resume evaluation after robot finishes speaking."""
        with self._paused_lock:
            self._paused = False

    def stop(self):
        """Signal the evaluation loop to stop."""
        self._stop_event.set()

    def notify_speech(self, text):
        """Called when a new non-empty transcript chunk arrives.
        Updates the last-speech timestamp for silence detection."""
        with self._speech_time_lock:
            self._last_speech_time = time.time()

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run(self):
        self.logger.log("[EVALUATOR] Started. interval={}s".format(self.eval_interval))

        while not self._stop_event.is_set():
            # Wait for the evaluation interval (or stop signal)
            self._stop_event.wait(self.eval_interval)
            if self._stop_event.is_set():
                break

            # Skip while paused (robot is speaking)
            with self._paused_lock:
                if self._paused:
                    continue

            # Check if active member is on cooldown
            if not self.audience_manager.active_member.can_speak():
                continue

            # --- Silence prompting ---
            with self._speech_time_lock:
                seconds_silent = time.time() - self._last_speech_time

            if seconds_silent > SILENCE_THRESHOLD_SECONDS:
                if not self._silence_prompt_sent:
                    self._handle_silence_prompt()
                    self._silence_prompt_sent = True
                continue  # don't run normal evaluation on silence
            else:
                self._silence_prompt_sent = False

            # --- Normal content evaluation ---
            transcript = self.transcriber.get_rolling_window(45)
            word_count = len(transcript.split()) if transcript else 0
            if word_count < MIN_TRANSCRIPT_WORDS:
                continue

            self._run_evaluation(transcript)

        self.logger.log("[EVALUATOR] Stopped.")

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _next_interrupt_id(self):
        with self._counter_lock:
            self._interrupt_counter += 1
            return "int_{:03d}".format(self._interrupt_counter)

    def _run_evaluation(self, transcript):
        """Run a single content-driven evaluation cycle."""
        interrupt_id = self._next_interrupt_id()

        # Checkpoint: eval_start
        self.logger.log_timing(interrupt_id, "eval_start")

        # Checkpoint: grok_request_sent
        self.logger.log_timing(interrupt_id, "grok_request_sent")

        result = self.audience_manager.evaluate_for_interrupt(
            transcript_window=transcript,
            settings=self.settings,
            logger=self.logger,
        )

        # Checkpoint: grok_response_received
        self.logger.log_timing(interrupt_id, "grok_response_received")

        # Determine raw response for logging
        grok_raw = "interrupt={}".format(result.get("interrupt", False))
        if result.get("speech"):
            grok_raw += " speech={}".format(repr(result["speech"])[:100])

        self.logger.log_eval_decision(
            interrupt_id=interrupt_id,
            transcript_window=transcript,
            grok_response=grok_raw,
            decision=result,
        )

        if result.get("interrupt") and result.get("speech"):
            self.logger.log("[EVALUATOR] Interrupt triggered: {} | {}".format(
                interrupt_id, result["speech"][:80]))
            self.audience_manager.active_member.mark_spoke()
            self.on_interrupt_callback(result["speech"], interrupt_id)

    def _handle_silence_prompt(self):
        """Handle the case where the speaker has been silent for too long."""
        interrupt_id = self._next_interrupt_id()

        self.logger.log("[EVALUATOR] Speaker silent for >{}s, checking silence prompt...".format(
            SILENCE_THRESHOLD_SECONDS))
        self.logger.log_timing(interrupt_id, "eval_start")
        self.logger.log_timing(interrupt_id, "grok_request_sent")

        result = self.audience_manager.evaluate_silence_prompt(
            settings=self.settings,
            logger=self.logger,
        )

        self.logger.log_timing(interrupt_id, "grok_response_received")

        grok_raw = "silence_prompt interrupt={}".format(result.get("interrupt", False))
        self.logger.log_eval_decision(
            interrupt_id=interrupt_id,
            transcript_window="(speaker silent)",
            grok_response=grok_raw,
            decision=result,
        )

        if result.get("interrupt") and result.get("speech"):
            self.logger.log("[EVALUATOR] Silence prompt fired: {} | {}".format(
                interrupt_id, result["speech"][:80]))
            self.audience_manager.active_member.mark_spoke()
            self.on_interrupt_callback(result["speech"], interrupt_id)
