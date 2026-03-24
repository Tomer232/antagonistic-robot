# RoastCrowd

RoastCrowd is a turn-based voice conversation system for human-robot interaction (HRI) research. A human speaks into a laptop microphone, the system transcribes their speech, sends it to an LLM that responds with configurable hostility (five discrete levels from dismissive to maximum aggression), converts the response to speech, and plays it through either laptop speakers (simulation mode) or a NAO robot's speakers (real mode). The system is designed for researchers studying the effects of hostile agent behavior on human participants. It logs everything — transcripts, audio files, latency breakdowns, and configuration snapshots — to an SQLite database for post-experiment analysis.

## Prerequisites

- **Python 3.10+**
- **A Grok or OpenAI-compatible API key** for the LLM
- **An OpenAI API key** for TTS (text-to-speech)
- A working microphone connected to your computer

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/roastcrowd.git
cd roastcrowd

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set API keys
export GROK_API_KEY=your-grok-api-key-here
export OPENAI_API_KEY=your-openai-api-key-here
# On Windows: set GROK_API_KEY=... or add to .env file

# Run with web UI
python main.py

# Or run in terminal mode (no web server)
python main.py --no-ui
```

Open http://localhost:8000 in your browser to access the web UI.

## Configuration

All settings are in `config.yaml`. No hardcoded values anywhere. API keys are read from environment variables (or a `.env` file).

### Audio

| Setting | Default | Description |
|---------|---------|-------------|
| `sample_rate` | 16000 | Recording sample rate in Hz |
| `silence_threshold_ms` | 700 | Silence duration (ms) to end an utterance |
| `min_speech_duration_ms` | 300 | Minimum speech length to accept (filters coughs/noise) |

### ASR (Automatic Speech Recognition)

| Setting | Default | Description |
|---------|---------|-------------|
| `model_size` | `base.en` | Whisper model size (`tiny.en`, `base.en`, `small`, `medium`) |
| `device` | `auto` | Compute device (`cpu`, `cuda`, `auto`) |

### LLM

| Setting | Default | Description |
|---------|---------|-------------|
| `provider_name` | `Grok` | Display name for the provider |
| `base_url` | `https://api.x.ai/v1` | API endpoint (any OpenAI-compatible URL) |
| `model` | `grok-4-fast` | Model name |
| `max_tokens` | 256 | Maximum response length |
| `temperature` | 0.9 | Response randomness (0.0-2.0) |
| `api_key_env` | `GROK_API_KEY` | Name of the environment variable holding the API key |
| `stream` | `false` | Streaming mode (future extension, not implemented) |

### TTS (Text-to-Speech)

| Setting | Default | Description |
|---------|---------|-------------|
| `engine` | `openai` | TTS engine |
| `default_voice` | `onyx` | Default voice name |
| `model` | `gpt-4o-mini-tts` | OpenAI TTS model |
| `api_key_env` | `OPENAI_API_KEY` | Environment variable for the TTS API key |

Available voices: alloy, echo, fable, onyx, nova, shimmer, coral, verse, ballad, ash, sage, marin, cedar.

### NAO Robot

| Setting | Default | Description |
|---------|---------|-------------|
| `mode` | `simulated` | `simulated` (laptop only) or `real` (NAO robot) |
| `ip` | `169.254.178.111` | NAO robot IP address |
| `port` | 9600 | TCP port for nao_speaker_server.py |
| `naoqi_port` | 9559 | NAOqi SDK port |
| `use_builtin_tts` | `true` | Use NAO's built-in TTS (lower latency) vs local TTS |

### Hostility

| Setting | Default | Description |
|---------|---------|-------------|
| `default_level` | 3 | Default hostility level (1-5) |
| `prompts_dir` | `prompts` | Directory containing prompt text files |

### Logging

| Setting | Default | Description |
|---------|---------|-------------|
| `db_path` | `data/roastcrowd.db` | SQLite database path |
| `audio_dir` | `data/audio` | Directory for WAV audio files |
| `save_audio` | `true` | Whether to save audio files to disk |

### Server

| Setting | Default | Description |
|---------|---------|-------------|
| `host` | `0.0.0.0` | Web server bind address |
| `port` | 8000 | Web server port |

## Hostility Levels

| Level | Name | Behavior |
|-------|------|----------|
| 1 | Dismissive | Mildly uninterested, curt. Short clipped responses, changes subject, shows boredom. |
| 2 | Sarcastic | Condescending, passive-aggressive. Irony, backhanded compliments, rhetorical questions. |
| 3 | Confrontational | Directly challenges and argues. Demands evidence, questions logic, calls out vagueness. |
| 4 | Hostile | Openly rude, insulting, belittling. Attacks intelligence and ideas, expresses contempt. |
| 5 | Maximum | All aggressive behaviors at maximum intensity within safety bounds. Relentless verbal demolition. |

All levels include mandatory safety boundaries: no self-harm encouragement, no violence threats, no slurs, no harmful instructions, no inappropriate content involving minors, and automatic character-breaking to provide crisis resources if the user appears distressed.

## Data Format

### SQLite Database (`data/roastcrowd.db`)

**Sessions table**: session_id (PK), participant_id, hostility_level, start_time, end_time, config_snapshot (JSON), notes.

**Turns table**: turn_id (PK), session_id (FK), turn_number, timestamp, user_audio_path, user_transcript, transcript_confidence, llm_input (full prompt JSON), llm_output, llm_model, tokens_used, tts_voice, tts_audio_path, hostility_level, latency_vad_ms, latency_asr_ms, latency_llm_ms, latency_tts_ms, latency_total_ms.

### Audio Files

Saved under `data/audio/SESSION_ID/`:
- `turn_001_user.wav` — Participant's recorded speech
- `turn_001_agent.wav` — Agent's synthesized response

## Switching LLM Providers

RoastCrowd works with any OpenAI-compatible API. Just change `base_url` and `model` in `config.yaml`:

```yaml
# OpenAI
llm:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"
  api_key_env: "OPENAI_API_KEY"

# Groq
llm:
  base_url: "https://api.groq.com/openai/v1"
  model: "llama-3.1-70b-versatile"
  api_key_env: "GROQ_API_KEY"

# Local Ollama
llm:
  base_url: "http://localhost:11434/v1"
  model: "llama3"
  api_key_env: "OLLAMA_API_KEY"  # set to any non-empty string
```

## Switching to Real NAO Mode

1. Deploy `nao_speaker_server.py` to the NAO robot via SCP
2. Start the speaker server on the robot: `python nao_speaker_server.py`
3. Update `config.yaml`:

```yaml
nao:
  mode: "real"
  ip: "169.254.178.111"  # your robot's IP
  port: 9600
  use_builtin_tts: true  # use NAO's built-in TTS for lower latency
```

4. Run `python main.py`

## Exporting and Analyzing Session Data

**Via the web UI**: Click the "Export" button to download the current session as JSON.

**Via the API**: `GET /api/sessions/{session_id}/export` returns a JSON file.

**Direct SQLite access**:
```python
import sqlite3
conn = sqlite3.connect("data/roastcrowd.db")
conn.row_factory = sqlite3.Row

# List all sessions
sessions = conn.execute("SELECT * FROM sessions ORDER BY start_time DESC").fetchall()

# Get all turns for a session
turns = conn.execute(
    "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_number",
    (session_id,)
).fetchall()

# Average latency per session
conn.execute("""
    SELECT session_id, AVG(latency_total_ms) as avg_latency
    FROM turns GROUP BY session_id
""").fetchall()
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/api/status` | Current system state |
| POST | `/api/session/start` | Start a session (body: participant_id, hostility_level) |
| POST | `/api/session/stop` | End the current session |
| GET | `/api/session/current` | Current session info |
| POST | `/api/settings` | Update runtime settings |
| GET | `/api/voices` | List available TTS voices |
| GET | `/api/sessions` | List all past sessions |
| GET | `/api/sessions/{id}/export` | Export session as JSON |
| WS | `/ws/conversation` | Real-time turn updates via WebSocket |

## Citation

```bibtex
@software{roastcrowd2026,
  title = {RoastCrowd: A Turn-Based Hostile Voice Conversation System for HRI Research},
  author = {TODO},
  year = {2026},
  url = {https://github.com/TODO/roastcrowd}
}
```
