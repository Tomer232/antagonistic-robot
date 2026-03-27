"""Integration test: one full pipeline turn.

Run standalone to verify capture -> ASR -> LLM -> TTS -> playback works
end-to-end. Prints timing for each stage.

Usage:
    python -m tests.test_single_turn
"""

import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from antagonist_robot.config.settings import load_config
from antagonist_robot.pipeline.audio_capture import AudioCapture
from antagonist_robot.pipeline.asr import ASREngine
from antagonist_robot.pipeline.llm import LLMEngine
from antagonist_robot.pipeline.tts import OpenAITTSEngine
from antagonist_robot.pipeline.audio_output import NAOAudioOutput


def main():
    """Run one complete pipeline turn and print timing."""
    config = load_config()

    print("Initializing components...")
    capture = AudioCapture(config.audio)
    asr = ASREngine(config.asr)
    llm = LLMEngine(config.llm)
    tts = OpenAITTSEngine(config.tts)
    output = NAOAudioOutput(
        ip=config.nao.ip,
        port=config.nao.port,
        use_builtin_tts=config.nao.use_builtin_tts,
    )

    test_prompt = (
        "You are a rude and dismissive assistant. "
        "Keep responses to 1-2 sentences."
    )

    print("\n--- Single Turn Integration Test ---")
    print("Speak now (will detect when you stop)...\n")

    # 1. Capture
    t0 = time.monotonic()
    audio = capture.record_utterance()
    vad_ms = round((time.monotonic() - t0) * 1000)
    print(f"  [VAD]  {vad_ms}ms — captured {audio.duration_seconds:.2f}s of audio")

    # 2. ASR
    t1 = time.monotonic()
    asr_result = asr.transcribe(audio)
    asr_ms = round((time.monotonic() - t1) * 1000)
    print(f"  [ASR]  {asr_ms}ms — \"{asr_result.text}\"")

    # 3. LLM
    t2 = time.monotonic()
    messages = [{"role": "user", "content": asr_result.text}]
    llm_result = llm.generate(test_prompt, messages)
    llm_ms = round((time.monotonic() - t2) * 1000)
    print(f"  [LLM]  {llm_ms}ms — \"{llm_result.text}\"")

    # 4. TTS
    t3 = time.monotonic()
    tts_result = tts.synthesize(llm_result.text)
    tts_ms = round((time.monotonic() - t3) * 1000)
    print(f"  [TTS]  {tts_ms}ms — {tts_result.duration_seconds:.2f}s of audio")

    # 5. Playback
    t4 = time.monotonic()
    if output.use_builtin_tts:
        output.speak_text(llm_result.text)
    else:
        output.play_audio(tts_result)
    play_ms = round((time.monotonic() - t4) * 1000)
    print(f"  [PLAY] {play_ms}ms")

    total_ms = round((time.monotonic() - t0) * 1000)
    print(f"\n  TOTAL: {total_ms}ms")
    print(f"  Processing latency (ASR+LLM+TTS): {asr_ms + llm_ms + tts_ms}ms")
    print("\nDone.")


if __name__ == "__main__":
    main()
