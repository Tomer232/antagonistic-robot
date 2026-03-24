# conversation_loop.py — Turn-based conversation loop
# User speaks their full turn. Once silence is detected for long enough,
# the robot generates and delivers a response. Repeat until session ends.

import os
import time
import json

from dotenv import load_dotenv

from robot_state import set_state
from audience_manager import AudienceManager
from session_logger import SessionLogger

# ---------------------------------------------------------
# Paths / IPC files
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROL_FILE = os.path.join(BASE_DIR, "control_flags.txt")
LOG_FILE = os.path.join(BASE_DIR, "conversation_log.jsonl")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

load_dotenv()

# ---------------------------------------------------------
# Default settings
# ---------------------------------------------------------
DEFAULT_SETTINGS = {
    "mode": "simulation",
    "nao_ip": "192.168.1.100",
    "robot_ip": "192.168.1.100",
    "robot_port": 9559,
    "robot_password": "nao",

    "intensity": 5,
    "attack_vector": "logic",
    "response_style": "confrontational",
    "recording_duration_seconds": 2,

    "total_session_minutes": 0,
    "warmup_seconds": 0,
    "tts_voice": "onyx",

    "silence_chunks_to_end_turn": 2,
}

# Polling interval for the main event loop
TICK_INTERVAL = 0.1   # seconds


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_settings():
    s = DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                s.update(data)
    except Exception:
        pass
    return s


def read_control_flag():
    try:
        with open(CONTROL_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() or "running"
    except IOError:
        return "running"


def write_control_flag(value):
    try:
        with open(CONTROL_FILE, "w", encoding="utf-8") as f:
            f.write(value.strip())
    except IOError:
        pass


# ---------------------------------------------------------
# Turn detection
# ---------------------------------------------------------

def _tick_listening(transcriber, silence_chunks_to_end_turn, logger):
    """Called every 0.1s tick while in LISTENING state.

    Reads the consecutive silent chunk counter directly from the transcriber
    (incremented per-chunk in the process loop) to decide when the user's
    turn is over.

    Returns:
        should_end_turn (bool)
    """
    silent_chunks = transcriber.get_consecutive_silent_chunks()
    session_text = transcriber.get_session_text()
    has_content = bool(session_text.strip())
    end_turn = silent_chunks >= silence_chunks_to_end_turn and has_content
    return end_turn


# ---------------------------------------------------------
# State handlers
# ---------------------------------------------------------

def _do_think_and_speak(robot, transcriber, audience, conversation_history,
                        settings, logger):
    """Capture user turn, stream LLM response sentence-by-sentence to TTS.

    Each sentence starts playing as soon as the LLM generates it,
    instead of waiting for the full response.
    """
    transcriber.pause()
    set_state("thinking")

    user_text = transcriber.get_session_text().strip()
    logger.log("[MAIN] User turn captured ({} words): {}".format(
        len(user_text.split()), user_text[:80]))
    logger.log_transcript("speaker", user_text)

    # Stream sentences from LLM → TTS
    full_response = ""
    for sentence in audience.generate_turn_response_streaming(
            user_text, conversation_history, settings, logger):
        set_state("speaking")
        logger.log("[MAIN] Speaking sentence: {}".format(sentence[:60]))
        transcriber.set_last_robot_text(sentence)
        robot.speak(sentence)
        full_response += (" " + sentence) if full_response else sentence

    logger.log("[MAIN] Full response: {}".format(full_response[:80]))
    logger.log_transcript("robot", full_response)
    audience.add_to_history(full_response)

    # Append both turns to conversation history for multi-turn context
    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": full_response})

    transcriber.flush_audio_only()
    transcriber.clear_session()
    set_state("listening")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    write_control_flag("running")

    settings = load_settings()
    mode = settings.get("mode", "simulation")
    robot_ip = settings.get("robot_ip", settings.get("nao_ip", "192.168.1.100"))
    robot_password = settings.get("robot_password", "nao")
    robot_port = settings.get("robot_port", 9559)
    recording_duration = settings.get("recording_duration_seconds", 2)
    silence_chunks_to_end_turn = settings.get("silence_chunks_to_end_turn", 2)

    # ---------------------------------------------------------------
    # Session logger
    # ---------------------------------------------------------------
    logger = SessionLogger()
    logger.log("[MAIN] Session starting. intensity={} attack={} style={} mode={}".format(
        settings.get("intensity", 5), settings.get("attack_vector", "logic"),
        settings.get("response_style", "confrontational"), mode))

    # ---------------------------------------------------------------
    # Backend selection
    # ---------------------------------------------------------------
    if mode == "real":
        from backends.real_backend import RealNao
        robot = RealNao(
            robot_ip=robot_ip,
            tcp_port=9600,
            robot_port=robot_port,
            robot_password=robot_password,
        )
        print("[MAIN] Mode: REAL ROBOT at", robot_ip)
        logger.log("[MAIN] Real robot backend at {}".format(robot_ip))

        def record_fn(duration_sec, sample_rate):
            return robot.record_from_nao(duration_sec, sample_rate)
    else:
        from backends.simulation_backend import SimulationNao
        robot = SimulationNao()
        print("[MAIN] Mode: SIMULATION")
        logger.log("[MAIN] Simulation backend")
        record_fn = None

    # ---------------------------------------------------------------
    # Streaming ASR
    # ---------------------------------------------------------------
    from audio.asr_streaming import StreamingTranscriber
    transcriber = StreamingTranscriber(
        chunk_duration=recording_duration,
        record_fn=record_fn,
    )
    transcriber.start()

    # ---------------------------------------------------------------
    # Audience manager
    # ---------------------------------------------------------------
    audience = AudienceManager(logger=logger)

    # ---------------------------------------------------------------
    # Session state
    # ---------------------------------------------------------------
    start_time = time.time()
    total_session_minutes = settings.get("total_session_minutes", 0)
    fallback_logged = False

    # Turn-based state machine
    conversation_history = []
    state = "LISTENING"

    # ---------------------------------------------------------------
    # Greeting
    # ---------------------------------------------------------------
    set_state("speaking")
    transcriber.pause()
    greeting = "Hello, I am ready to listen. You can start speaking whenever you are ready."
    transcriber.set_last_robot_text(greeting)
    speak_ok = robot.speak(greeting)
    if not speak_ok:
        logger.log("[MAIN] WARNING: Greeting failed to reach robot.")
        print("[MAIN] WARNING: Robot unreachable. ASR will still run.")
    logger.log_transcript("robot", greeting)
    transcriber.flush_audio_only()
    transcriber.clear_session()
    transcriber.resume()
    set_state("listening")

    print("[MAIN] Session started (turn-based). intensity={} chunk={}s silence_chunks={}".format(
        settings.get("intensity", 5), recording_duration, silence_chunks_to_end_turn))
    logger.log("[MAIN] Session loop running.")

    # ---------------------------------------------------------------
    # Main event loop — turn-based state machine
    # ---------------------------------------------------------------
    try:
        while True:

            # -------------------------------------------------------
            # Control flags (checked in all states)
            # -------------------------------------------------------
            flag = read_control_flag()

            if flag == "end":
                _handle_end(transcriber, robot, logger, mode)
                break

            if flag == "paused":
                transcriber.pause()
                set_state("idle")
                time.sleep(0.5)
                continue
            else:
                if state == "LISTENING":
                    transcriber.resume()  # idempotent if already running

            # -------------------------------------------------------
            # Session timeout
            # -------------------------------------------------------
            if total_session_minutes and total_session_minutes > 0:
                if (time.time() - start_time) >= total_session_minutes * 60:
                    _handle_timeout(transcriber, robot, logger, mode)
                    break

            # -------------------------------------------------------
            # ASR fallback warning (one-time)
            # -------------------------------------------------------
            if not fallback_logged and transcriber.is_using_fallback():
                fallback_logged = True
                logger.log("[MAIN] ASR fell back to PC microphone (NAO mic unreachable).")
                logger.log_transcript("system",
                    "NAO microphone unreachable. Using PC microphone instead.")
                print("[MAIN] ASR fell back to PC microphone.")

            # -------------------------------------------------------
            # State dispatch
            # -------------------------------------------------------
            if state == "LISTENING":
                end_turn = _tick_listening(
                    transcriber, silence_chunks_to_end_turn, logger)
                if end_turn:
                    logger.log("[MAIN] User turn ended. Thinking + speaking.")
                    _do_think_and_speak(robot, transcriber, audience,
                                        conversation_history, settings, logger)
                    # Back to listening after speaking completes

            time.sleep(TICK_INTERVAL)

    except KeyboardInterrupt:
        logger.log("[MAIN] KeyboardInterrupt — shutting down.")
    except Exception as e:
        logger.log_error(e, "Main loop exception")
        print("[MAIN] Fatal error:", e)
    finally:
        transcriber.stop()
        if mode == "real":
            try:
                robot.shutdown()
            except Exception:
                pass
        set_state("idle")
        logger.log("[MAIN] Session ended.")
        logger.close()


# ---------------------------------------------------------
# Session end helpers
# ---------------------------------------------------------

def _handle_end(transcriber, robot, logger, mode):
    """Graceful session end (control flag)."""
    set_state("speaking")
    transcriber.pause()
    goodbye = "Okay, ending the conversation now. Goodbye."
    transcriber.set_last_robot_text(goodbye)
    robot.speak(goodbye)
    logger.log_transcript("robot", goodbye)
    transcriber.stop()
    if mode == "real":
        try:
            robot.shutdown()
        except Exception:
            pass
    set_state("idle")
    logger.log("[MAIN] Session ended via control command.")


def _handle_timeout(transcriber, robot, logger, mode):
    """Session ended due to time limit."""
    set_state("speaking")
    transcriber.pause()
    msg = "The session time is over. Thank you for speaking."
    transcriber.set_last_robot_text(msg)
    robot.speak(msg)
    logger.log_transcript("robot", msg)
    transcriber.stop()
    if mode == "real":
        try:
            robot.shutdown()
        except Exception:
            pass
    set_state("idle")
    logger.log("[MAIN] Session ended via timeout.")


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
