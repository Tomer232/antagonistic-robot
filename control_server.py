import os
import threading
import json
from flask import Flask, jsonify, request
from flask_cors import CORS

# Base project directory
BASE_DIR = r"C:\Users\tomer\Desktop\NAO_LLM"

# Serve React build (run `npm run build` in webui/ first)
WEBUI_BUILD = os.path.join(BASE_DIR, "webui", "build")

app = Flask(__name__, static_folder=WEBUI_BUILD, static_url_path="")
CORS(app)
CONTROL_FILE = os.path.join(BASE_DIR, "control_flags.txt")
LOG_FILE = os.path.join(BASE_DIR, "conversation_log.jsonl")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
ROBOT_STATE_FILE = os.path.join(BASE_DIR, "robot_state.txt")  # same as nao_talk.py

# Track the conversation thread
conversation_thread = None

# Simple status cache for API responses
system_state = {
    "status": "idle"  # "idle" | "running" | "paused" | "ended"
}

# Default settings (must match conversation_loop.py)
DEFAULT_SETTINGS = {
    # Intensity control
    "intensity": 5,                           # 0-10 slider
    "attack_vector": "logic",                 # "logic" | "credibility" | "delivery" | "facts"
    "response_style": "confrontational",      # "dismissive" | "confrontational" | "sarcastic" | "interrogative"
    "recording_duration_seconds": 2,          # audio chunk size for streaming ASR

    # Robot connection
    "mode": "simulation",                     # "simulation" | "real"
    "nao_ip": "192.168.1.100",               # kept for backward compat
    "robot_ip": "192.168.1.100",
    "robot_port": 9559,
    "robot_password": "nao",

    # Session structure
    "total_session_minutes": 0,               # 0 = unlimited
    "warmup_seconds": 0,
    "tts_voice": "onyx",
}


# ---------------------------------------------------------
# Helpers: settings, control flags, process status, log, robot state
# ---------------------------------------------------------
def load_settings():
    settings = DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                settings.update(data)
    except Exception:
        pass

    # Safety: if tts_voice in file is invalid, normalize to default
    allowed_voices = {
        "alloy", "echo", "fable", "onyx", "nova", "shimmer",
        "coral", "verse", "ballad", "ash", "sage", "marin", "cedar",
    }
    tv = settings.get("tts_voice", "onyx")
    if tv not in allowed_voices:
        settings["tts_voice"] = "onyx"

    return settings


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"[CONTROL] Failed to save settings: {e}")


def validate_and_merge_settings(payload):
    """Validate incoming settings and merge into current settings."""
    s = load_settings()

    # --- Intensity control ---

    # intensity (0-10 slider)
    if "intensity" in payload:
        try:
            val = int(payload["intensity"])
            s["intensity"] = max(0, min(10, val))
        except (ValueError, TypeError):
            pass

    # attack_vector
    av = payload.get("attack_vector")
    if av in ("logic", "credibility", "delivery", "facts"):
        s["attack_vector"] = av

    # response_style
    rs = payload.get("response_style")
    if rs in ("dismissive", "confrontational", "sarcastic", "interrogative"):
        s["response_style"] = rs

    # recording_duration_seconds
    try:
        rds = int(payload.get("recording_duration_seconds",
                              s.get("recording_duration_seconds", 2)))
        rds = max(1, min(rds, 10))
        s["recording_duration_seconds"] = rds
    except Exception:
        pass

    # --- Robot connection ---

    # mode
    mode = payload.get("mode")
    if mode in ("simulation", "real"):
        s["mode"] = mode

    # robot_ip (accept both "robot_ip" and legacy "nao_ip")
    rip = payload.get("robot_ip", payload.get("nao_ip"))
    if isinstance(rip, str) and rip.strip():
        s["robot_ip"] = rip.strip()
        s["nao_ip"] = rip.strip()

    # robot_port
    try:
        rport = int(payload.get("robot_port", s.get("robot_port", 9559)))
        rport = max(1, min(rport, 65535))
        s["robot_port"] = rport
    except Exception:
        pass

    # robot_password
    rpwd = payload.get("robot_password")
    if isinstance(rpwd, str):
        s["robot_password"] = rpwd

    # --- Session structure ---

    # total_session_minutes
    try:
        tsm = int(payload.get("total_session_minutes", s["total_session_minutes"]))
        tsm = max(0, min(tsm, 240))
        s["total_session_minutes"] = tsm
    except Exception:
        pass

    # warmup_seconds
    try:
        wus = int(payload.get("warmup_seconds", s["warmup_seconds"]))
        wus = max(0, min(wus, 600))
        s["warmup_seconds"] = wus
    except Exception:
        pass

    # tts_voice
    allowed_voices = {
        "alloy", "echo", "fable", "onyx", "nova", "shimmer",
        "coral", "verse", "ballad", "ash", "sage", "marin", "cedar",
    }
    tv = payload.get("tts_voice")
    if isinstance(tv, str) and tv in allowed_voices:
        s["tts_voice"] = tv

    return s


def write_control_flag(value: str):
    try:
        with open(CONTROL_FILE, "w", encoding="utf-8") as f:
            f.write(value.strip())
    except IOError as e:
        print(f"[CONTROL] Failed to write control flag: {e}")


def read_control_flag() -> str:
    try:
        with open(CONTROL_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() or "running"
    except IOError:
        return "running"


def read_robot_state() -> str:
    """
    Read the current robot state from robot_state.txt.
    States: "idle", "listening", "thinking", "speaking"
    Default: "idle" on error.
    """
    try:
        with open(ROBOT_STATE_FILE, "r", encoding="utf-8") as f:
            val = f.read().strip().lower()
            return val or "idle"
    except IOError:
        return "idle"


def refresh_status_from_process():
    global conversation_thread, system_state

    if conversation_thread is not None:
        if not conversation_thread.is_alive():
            conversation_thread = None
            system_state["status"] = "ended"
            return

    if conversation_thread is not None:
        flag = read_control_flag()
        if flag == "paused":
            system_state["status"] = "paused"
        elif flag == "running":
            system_state["status"] = "running"
        elif flag == "end":
            system_state["status"] = "ended"
        return

    if system_state["status"] not in ("idle", "ended"):
        system_state["status"] = "idle"


# ---------------------------------------------------------
# API endpoints
# ---------------------------------------------------------
@app.route("/")
def serve_index():
    """Serve the React web UI."""
    return app.send_static_file("index.html")


@app.route("/status", methods=["GET"])
def get_status():
    refresh_status_from_process()
    resp = dict(system_state)
    resp["robot_state"] = read_robot_state()
    return jsonify(resp)


@app.route("/settings", methods=["GET", "POST"])
def settings_endpoint():
    if request.method == "GET":
        s = load_settings()
        return jsonify(s)

    # POST
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}

    s = validate_and_merge_settings(payload)
    save_settings(s)
    return jsonify(s)


@app.route("/start", methods=["POST"])
def start_conversation():
    global conversation_thread, system_state

    refresh_status_from_process()

    if conversation_thread is not None and conversation_thread.is_alive():
        return jsonify({"error": "Conversation already running", "status": system_state["status"]}), 400

    # Reset control flag and clear log for a fresh session
    write_control_flag("running")
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")  # truncate
    except IOError as e:
        print(f"[CONTROL] Failed to clear log file: {e}")

    print("[CONTROL] Starting conversation loop (in-process thread).")
    try:
        from conversation_loop import main as conversation_main
        conversation_thread = threading.Thread(
            target=conversation_main, daemon=True, name="conversation-loop"
        )
        conversation_thread.start()
    except Exception as e:
        print(f"[CONTROL] Failed to start conversation loop: {e}")
        return jsonify({"error": str(e), "status": system_state["status"]}), 500

    system_state["status"] = "running"
    return jsonify(system_state)


@app.route("/pause", methods=["POST"])
def pause_conversation():
    global system_state

    refresh_status_from_process()

    if system_state["status"] != "running":
        return jsonify({"error": "Can only pause when running", "status": system_state["status"]}), 400

    write_control_flag("paused")
    system_state["status"] = "paused"
    print("[CONTROL] Pause requested")
    return jsonify(system_state)


@app.route("/resume", methods=["POST"])
def resume_conversation():
    global system_state

    refresh_status_from_process()

    if system_state["status"] != "paused":
        return jsonify({"error": "Can only resume when paused", "status": system_state["status"]}), 400

    write_control_flag("running")
    system_state["status"] = "running"
    print("[CONTROL] Resume requested")
    return jsonify(system_state)


@app.route("/end", methods=["POST"])
def end_conversation():
    global system_state

    refresh_status_from_process()

    if system_state["status"] not in ("running", "paused"):
        return jsonify({"error": "Can only end when running or paused", "status": system_state["status"]}), 400

    write_control_flag("end")
    print("[CONTROL] End requested")
    system_state["status"] = "ended"
    return jsonify(system_state)


@app.route("/log", methods=["GET"])
def get_log():
    """
    Return the conversation log as a JSON array of events.
    Each line in the log file is a JSON object.
    """
    events = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    events.append(obj)
                except Exception:
                    continue
    except IOError:
        pass

    return jsonify(events)


@app.errorhandler(404)
def not_found(e):
    """Serve React app for client-side routes."""
    index_path = os.path.join(WEBUI_BUILD, "index.html")
    if os.path.exists(index_path):
        return app.send_static_file("index.html")
    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    print(f"[CONTROL] BASE_DIR = {BASE_DIR}")
    app.run(host="127.0.0.1", port=5000, debug=False)
