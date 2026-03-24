# backends/simulation_backend.py
import sys
import os

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nao_interface import NaoInterface
from audio.tts import synthesize_and_play
from audio.asr import transcribe_from_mic


class SimulationNao(NaoInterface):
    def speak(self, text):
        """
        Speak the given text using the PC speakers via pyttsx3.
        Returns True on success, False on error.
        """
        if not text:
            return True

        print("[SIMULATION] Robot says:", text)
        synthesize_and_play(text)
        return True

    def listen(self):
        """
        Listen via the PC microphone and return transcribed text.
        """
        print("[SIMULATION] Listening via PC microphone...")
        return transcribe_from_mic()
