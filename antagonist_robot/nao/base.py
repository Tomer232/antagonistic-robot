"""Abstract NAO adapter base class.

Defines the interface for interacting with the NAO robot.
The conversation manager calls these methods at specific points
in the turn lifecycle.
"""

from abc import ABC, abstractmethod


class NAOAdapter(ABC):
    """Abstract interface for NAO robot interaction."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the robot."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the robot."""
        ...

    @abstractmethod
    def on_response(self, text: str, hostility_level: int) -> None:
        """Called when the LLM produces a response.

        Implementations should trigger appropriate gestures
        based on the hostility level.
        """
        ...

    @abstractmethod
    def on_listening(self) -> None:
        """Called when the system starts recording user speech.

        Implementations should put the robot in a listening posture.
        """
        ...

    @abstractmethod
    def on_idle(self) -> None:
        """Called when the system is idle between sessions.

        Implementations should return the robot to a neutral posture.
        """
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if connected to the robot."""
        ...
