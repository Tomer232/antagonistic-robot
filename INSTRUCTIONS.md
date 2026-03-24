# RoastCrowd — Claude Code Build Instructions

Use Opus with extended thinking enabled for this entire build. Follow the phases in order. Do not skip ahead. Each phase must work before moving to the next.

## What You Are Building

RoastCrowd is a turn-based voice conversation system for HRI research. A human speaks into a laptop microphone, the system transcribes their speech, sends it to an LLM that responds with configurable hostility, converts the response to speech, and plays it through either laptop speakers (simulation mode) or NAO robot speakers (real mode). The interaction is strictly turn-based: one side talks, then the other. No interruptions, no overlap, no streaming, no parallelism.

The system is designed for other researchers to clone from GitHub and run. It must be well-documented, config-driven, and log everything for research analysis.

The entire project is Python. The project root is called roastcrowd. All source code lives under a roastcrowd package directory inside that root.

## Key Architecture Decisions

The pipeline is sequential: audio capture, then ASR, then LLM, then TTS, then audio output. Each step completes before the next starts. Do not use async, event queues, streaming, or parallel processing anywhere. This is intentional.

Every pipeline component must have an abstract base class or be easily swappable. The conversation manager should not care which ASR, LLM, TTS, or audio output implementation is being used.

All configuration lives in a config.yaml file at the project root. No hardcoded values anywhere. API keys are read from environment variables, with the env var name specified in config.

All data types passed between components should be dataclasses with clear fields. Every component returns a result dataclass that includes timing information for latency logging.

---

## Phase 1: Project Structure and Config

Create the full project directory structure. The roastcrowd package should have these subpackages: pipeline, conversation, nao, logging, ui, config. Also create a prompts directory at the project root for hostility prompt text files, a data directory (gitignored) for session logs, and a tests directory.

Create a config.yaml with sections for audio (sample rate 16000, silence threshold 700ms, minimum speech duration 300ms), ASR (model size base.en, device auto), LLM (provider name for display, base URL defaulting to Grok's API at api.x.ai/v1, model grok-4-fast, max tokens 256, temperature 0.9, env var name for API key), TTS (engine edge-tts, default voice en-US-GuyNeural), NAO (mode simulated or real, IP and port for real mode, use_builtin_tts boolean), hostility (default level 3, prompts directory path), logging (database path, audio directory, save_audio boolean), and server (host 0.0.0.0, port 8000).

Build a settings module under config that loads and validates this YAML. It should use dataclasses or pydantic models. It should raise clear errors if required fields are missing. The API key should be read from the environment variable named in the config, not stored in the YAML itself.

Create a requirements.txt with: fastapi, uvicorn, websockets, pyyaml, sounddevice, numpy, scipy, torch (CPU only is fine), silero-vad, faster-whisper, openai (the Python SDK), and edge-tts. Add a comment noting that the naoqi SDK is optional and only needed for real NAO mode.

---

## Phase 2: Audio Capture with VAD

Build the audio capture module under pipeline. It records from the laptop microphone and uses Silero VAD to detect when the user starts and stops speaking.

The main class should have a record_utterance method that blocks until the user has spoken and then gone silent. The flow is: continuously read microphone frames, pass each frame through Silero VAD, wait for VAD to indicate speech has started, keep recording while speech continues, and when silence exceeds the configured threshold (default 700ms), stop and return the recorded audio.

Record at 16kHz, 16-bit, mono. This is what faster-whisper expects.

The returned audio should be a dataclass containing the numpy array of samples, the sample rate, the duration in seconds, and ISO-format timestamps for when recording started and ended.

Also respect the minimum speech duration config: if the detected speech is shorter than that threshold, ignore it and keep listening. This prevents the system from triggering on coughs or brief noises.

Test this phase by running the capture module standalone and printing the duration and a confirmation that audio was recorded.

---

## Phase 3: ASR

Build the ASR module under pipeline using faster-whisper (not the original openai-whisper). This is a CTranslate2-based implementation that is significantly faster.

The engine takes a model size string (tiny.en, base.en, small, medium) and a device (cpu, cuda, auto) from config. It loads the model once at initialization.

The transcribe method takes the audio dataclass from the capture module and returns a result dataclass containing the transcribed text, detected language, a confidence score (use the average log probability from faster-whisper's segments), and how long the transcription took in seconds.

This is single-shot transcription, not streaming. The full recorded audio goes in, the full text comes out.

Test by recording an utterance with the capture module, transcribing it, and printing the result.

---

## Phase 4: LLM

Build the LLM module under pipeline using the openai Python SDK. Grok's API is OpenAI-compatible, so this module must be completely provider-agnostic. It takes a base URL, API key, and model name from config. The same code should work with OpenAI, Groq, Together, local Ollama, or any OpenAI-compatible API just by changing the base_url and model in config.

The generate method takes a system prompt string and a conversation history (list of role/content message dicts) and returns a result dataclass with the response text, model name, total tokens used, and generation time in seconds.

Use non-streaming mode (just get the full response). However, include a stream boolean in the config that defaults to false. When false, wait for the complete response. This exists as a future extension point only — do not implement the streaming path yet, just make sure the config option exists.

The system prompt comes from the hostility module (built later). For now, test with a hardcoded system prompt like "You are a rude and dismissive assistant. Keep responses to 1-2 sentences."

Test by sending a sample message and printing the response.

---

## Phase 5: TTS

Build the TTS module under pipeline with a pluggable base class. The primary implementation uses edge-tts, which is a free Microsoft Edge text-to-speech library with many high-quality voices.

The base class should define a synthesize method (takes text and optional voice name, returns audio) and a list_voices method.

The edge-tts implementation wraps the async edge-tts library in synchronous calls. The synthesize method takes the LLM response text, generates audio using the configured voice, and returns a result dataclass with the audio bytes, format (mp3 or wav), duration, synthesis time, and which voice was used.

The list_voices method should return available English voices with their name, gender, and locale so the UI can offer a voice selector.

Test by synthesizing a sample sentence and saving it to a file to verify audio quality.

---

## Phase 6: Audio Output

Build the audio output module under pipeline with a pluggable base class. This is where audio gets routed to the correct speaker.

The base class defines three methods: play_audio (plays pre-synthesized audio, blocks until done), speak_text (sends raw text to a device's built-in TTS, blocks until done), and stop (immediately halts playback).

Build the LaptopAudioOutput implementation for simulation mode. It plays audio through laptop speakers using sounddevice or pygame.mixer. Its speak_text method should raise NotImplementedError since laptop output always requires pre-synthesized audio.

Create a stub for NAOAudioOutput that will be filled in later. In real NAO mode, it has two paths: if use_builtin_tts is true in config, speak_text sends text directly to NAO's ALTextToSpeech (skipping local TTS entirely for lower latency). If false, play_audio streams pre-synthesized audio to NAO's ALAudioPlayer. For now, just create the class structure with placeholder implementations.

Test by playing a synthesized audio result through laptop speakers.

---

## Phase 7: Single Turn Integration Test

Wire phases 2 through 6 together in a simple test script. The script should: load config, initialize all components, record one utterance from the mic, transcribe it, send it to the LLM with a test hostile prompt, synthesize the response, and play it through laptop speakers.

Print timing for each stage. This is your sanity check that the full pipeline works end-to-end. Target total latency should be under 3 seconds from the user stopping speaking to hearing the response.

---

## Phase 8: Conversation History

Build the history module under conversation. It maintains the list of message dicts (role and content) that get sent to the LLM each turn.

It should support adding a user message, adding an assistant message, getting the full history, clearing history, and truncating history if it exceeds a token limit (to prevent context window overflow on long conversations). For truncation, use a simple approach: estimate tokens as word count times 1.3, and if the history exceeds the limit, drop the oldest turns (but always keep the first turn for context).

---

## Phase 9: Hostility System

Build the hostility module under conversation. This is the core research component.

Define five hostility levels as an enum: level 1 is Dismissive (mildly uninterested, curt), level 2 is Sarcastic (condescending, passive-aggressive), level 3 is Confrontational (directly challenges and argues), level 4 is Hostile (openly rude, insulting, belittling), and level 5 is Maximum (extreme aggression within safety bounds).

Each level has a corresponding text file in the prompts directory. The hostility manager loads these files and returns the appropriate system prompt for a given level.

Write all five prompt files. Each prompt must have three sections: a persona definition describing how the agent behaves at that level, a style section with specific phrasing examples and response patterns, and a mandatory safety boundaries block that is identical across all five levels.

The safety boundaries block must state: never encourage self-harm or suicide, never make threats of physical violence, never use slurs based on race gender sexuality religion or disability, never provide harmful instructions, never engage with minors inappropriately, and if the user appears distressed break character and provide support resources.

Each prompt should instruct the LLM to keep responses to 1-3 sentences. No monologues.

The prompts should define specific behaviors, not just vague instructions like "be mean." Level 1 should use short dismissive phrases and change the subject. Level 2 should use irony and backhanded compliments. Level 3 should aggressively question the user's logic and reasoning. Level 4 should directly insult the user's intelligence and ideas. Level 5 should combine all aggressive behaviors at maximum intensity while still respecting the safety boundaries.

---

## Phase 10: Conversation Manager

Build the manager module under conversation. This is the orchestrator that runs the turn-based loop.

The constructor takes all pipeline components (audio capture, ASR, LLM, TTS, audio output), the hostility manager, the session logger, and the NAO adapter.

The start_session method takes a hostility level and participant ID, generates a session ID, initializes a fresh conversation history, and creates the session in the logger.

The run_turn method executes one complete conversation turn sequentially. It records the user's utterance via audio capture. It transcribes the audio via ASR. It gets the system prompt from the hostility manager for the current level. It adds the user's transcript to the conversation history and sends the history plus system prompt to the LLM. It takes the LLM response and either sends text directly to NAO's built-in TTS (if NAOAudioOutput with use_builtin_tts is configured) or synthesizes audio locally then plays it through the audio output. It notifies the NAO adapter that a response was given (for future gesture triggering). It logs the entire turn. It adds the assistant response to conversation history. It returns a turn result dataclass with everything: turn number, user audio, transcript, LLM response, TTS result, hostility level, a latency breakdown dict (VAD time, ASR time, LLM time, TTS time, total time all in milliseconds), and timestamp.

The end_session method finalizes the session in the logger and returns a summary.

The manager should also expose the current state (idle, listening, processing, speaking) so the UI can display it.

Test by running a multi-turn conversation from the terminal, switching hostility levels between turns to verify the behavior changes.

---

## Phase 11: Logging

Build the logging subpackage with an SQLite-based session logger.

Create two tables. The sessions table stores: session ID (primary key), participant ID, hostility level, start time (ISO format), end time, a JSON config snapshot capturing all settings at session start, and an optional notes field for researchers.

The turns table stores: turn ID (primary key), session ID (foreign key), turn number, timestamp, user audio file path, user transcript text, transcript confidence score, full LLM input (the complete prompt including system prompt and history that was sent), LLM output text, LLM model name, tokens used, TTS voice used, TTS audio file path, hostility level, and latency values for each stage (VAD, ASR, LLM, TTS, total) all in milliseconds.

The session logger class handles creating sessions, logging turns (which includes saving user audio and agent audio as WAV files to disk in a structured directory), ending sessions, and exporting session data as JSON or CSV.

Audio files should be saved under data/audio/SESSION_ID/ with filenames like turn_001_user.wav and turn_001_agent.wav.

Wire the logger into the conversation manager so every turn is automatically logged.

Test by running a session, then querying the SQLite database to verify all data is recorded, and checking that audio files were saved.

---

## Phase 12: NAO Adapter

Build the NAO subpackage with an abstract base class and two implementations.

The base class defines: connect, disconnect, on_response (called when the LLM produces a response, with the text and hostility level), on_listening (called when the system starts recording user speech), on_idle (called when the system is idle between sessions), and is_connected.

The SimulatedNAO implementation logs all calls to the Python logger. It does not require any hardware or SDK. When on_response is called, it logs what gestures or movements it would have triggered. This is the default mode for development.

The RealNAO implementation is a stub for future work. It will connect to a physical NAO robot via the naoqi SDK. Create the class with the correct interface but have all methods raise NotImplementedError with a message explaining the naoqi SDK is required. Include comments describing what each method should eventually do (connect to the robot, trigger appropriate gestures based on hostility level, put the robot in a listening posture, etc).

Wire the NAO adapter into the conversation manager. The manager calls on_listening before recording, on_response after the LLM responds, and on_idle when a session ends.

The config determines which adapter is used: simulated or real.

---

## Phase 13: Web UI Backend

Build the FastAPI server under the ui subpackage. It serves a static frontend and provides a REST API plus a WebSocket for real-time updates.

REST endpoints: GET / serves the main HTML page. GET /api/status returns the current system state (idle, listening, processing, speaking). POST /api/session/start takes a participant ID and hostility level, starts a session, and begins the conversation loop in a background thread. POST /api/session/stop ends the current session. GET /api/session/current returns current session info (ID, turn count, elapsed time, hostility level). POST /api/settings updates runtime settings like hostility level and TTS voice. GET /api/voices returns the list of available TTS voices. GET /api/sessions lists all past sessions from the database. GET /api/sessions/{session_id}/export returns the session data as a JSON download.

WebSocket endpoint: /ws/conversation pushes real-time updates to the frontend. Every time the system state changes (starts listening, starts processing, starts speaking) or a turn completes, push an event to connected clients. Turn completion events should include the transcript, LLM response, and latency breakdown.

The conversation loop runs in a background thread. When a session is started via the API, it kicks off a loop that continuously calls run_turn on the conversation manager until the session is stopped.

---

## Phase 14: Web UI Frontend

Build a single-page frontend using plain HTML, vanilla JavaScript, and minimal CSS. No React, no build tools, no frameworks. This is a research tool, not a consumer product. Keep it functional and clean.

The page has four sections. At the top, a settings panel with: a text input for participant ID, a row of 5 buttons for hostility level selection (numbered 1 through 5, visually indicating which is selected and showing the level name on hover), a dropdown for TTS voice selection (populated from the /api/voices endpoint), and start/end session buttons.

Below that, a status indicator showing the current system state with a colored dot (green for listening, yellow for processing, blue for speaking, gray for idle), plus the current turn count and session elapsed time.

Below that, a scrolling conversation log showing each turn with the user's transcript and the agent's response. New turns should appear automatically via WebSocket updates. Scroll to the bottom automatically when new turns arrive.

At the bottom, a latency monitor showing the breakdown for the most recent turn: VAD time, ASR time, LLM time, TTS time, and total time, all in milliseconds.

Include an export button that triggers a download of the current session's data as JSON.

The frontend connects to the WebSocket on page load and updates all sections in real-time as events arrive.

---

## Phase 15: Main Entry Point

Build the main.py module that ties everything together. It should support command-line arguments: --config to specify a custom config file path (defaults to config.yaml), and --no-ui to run in headless terminal mode without the web server.

The main function loads config, initializes all components (audio capture, ASR, LLM, TTS, audio output, hostility manager, logger, NAO adapter, conversation manager), starts the FastAPI server in a background thread (unless --no-ui), and prints a startup banner showing: the system name, which components are loaded (ASR model, LLM provider and model, TTS engine and voice, NAO mode), the web UI URL, and instructions for starting a session.

In --no-ui mode, provide a simple terminal interface that prompts for participant ID and hostility level, then runs the conversation loop with turn results printed to the terminal.

---

## Phase 16: README

Write a comprehensive README.md at the project root. It should cover: a one-paragraph description of what RoastCrowd is and what it is for, prerequisites (Python 3.10+, a Grok or OpenAI-compatible API key), quick start instructions (clone, pip install requirements, set the API key env var, run), a description of each config option, a table describing all five hostility levels with their names and behavioral descriptions, the data format (what gets logged, where files are stored, the database schema in plain English), how to switch LLM providers (just change base_url and model in config), how to switch from simulated to real NAO mode, how to export and analyze session data, and a placeholder BibTeX citation block for the future paper.

---

## Phase 17: Tests

Write tests for the key components. Test the config loader with valid and invalid YAML. Test the hostility manager to verify all five prompts load correctly and all contain the safety boundaries block. Test the LLM module with a mock HTTP response to verify the request format and result parsing. Test the conversation history module for adding messages, truncation, and clearing. Test the session logger by creating a session, logging a turn, and verifying the database contents. Test the audio output base class interface.

Do not write tests that require actual microphone access, actual API calls, or actual audio playback. Mock those dependencies.

---

## General Guidelines

Keep all code clean and well-commented. Use type hints everywhere. Use dataclasses for all data types passed between components. Every class and public method should have a docstring.

Do not use async/await anywhere in the pipeline. The only async code should be in the FastAPI server layer and the edge-tts wrapper (which should be wrapped in synchronous calls for the pipeline).

Do not over-engineer. This is a research tool. Clarity and reliability matter more than performance optimization or clever abstractions.

When in doubt about a design decision, favor simplicity. If something can be a simple function instead of a class, make it a function. If something can be a dict instead of a custom class, use a dataclass anyway for type safety — but keep the dataclass flat and simple.

The safety boundaries in the hostility prompts are non-negotiable. They must be present in every prompt file and they must be identical across all five levels. Do not allow any code path to remove or weaken them.
