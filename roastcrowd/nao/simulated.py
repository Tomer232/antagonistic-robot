"""Simulated NAO adapter for development without hardware.

Logs all calls to the Python logger. No hardware or SDK required.
This is the default mode for development and testing.
"""

import logging

from roastcrowd.nao.base import NAOAdapter

logger = logging.getLogger(__name__)


class SimulatedNAO(NAOAdapter):
    """Simulated NAO that logs what gestures it would trigger.

    When on_response is called, it logs what gestures or movements
    it would have triggered on a real NAO robot.
    """

    def __init__(self):
        self._connected = False

    def connect(self) -> None:
        """Mark as connected (no actual connection needed)."""
        self._connected = True
        logger.info("[SimulatedNAO] Connected (simulated)")

    def disconnect(self) -> None:
        """Mark as disconnected."""
        self._connected = False
        logger.info("[SimulatedNAO] Disconnected")

    def on_response(self, text: str, hostility_level: int) -> None:
        """Log what gesture would be triggered for the given hostility level."""
        gesture = {
            1: "slight shrug",
            2: "dismissive hand wave",
            3: "pointing gesture",
            4: "aggressive arm movement",
            5: "emphatic double arm gesture",
        }.get(hostility_level, "neutral")
        logger.info(
            "[SimulatedNAO] on_response: would trigger '%s' gesture "
            "at hostility level %d",
            gesture,
            hostility_level,
        )

    def on_listening(self) -> None:
        """Log that the robot would enter a listening posture."""
        logger.info(
            "[SimulatedNAO] on_listening: would enter listening posture "
            "(right hand near ear)"
        )

    def on_idle(self) -> None:
        """Log that the robot would return to neutral posture."""
        logger.info("[SimulatedNAO] on_idle: would return to neutral posture")

    def is_connected(self) -> bool:
        """Return simulated connection state."""
        return self._connected
