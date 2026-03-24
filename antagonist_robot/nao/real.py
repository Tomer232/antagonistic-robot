"""Real NAO adapter stub for future implementation.

Requires the naoqi Python SDK which is only available on systems
with NAO developer tools installed. All methods currently raise
NotImplementedError.
"""

from antagonist_robot.nao.base import NAOAdapter


class RealNAO(NAOAdapter):
    """Real NAO robot adapter. Requires naoqi SDK.

    Future implementation should:
    - Connect to NAO via naoqi ALProxy
    - Use ALMotion for gesture control (setAngles)
    - Use ALRobotPosture for posture management
    - Map hostility levels to gesture intensity
    - Reference nao_speaker_server.py for arm angle constants
    """

    def __init__(self, ip: str, naoqi_port: int = 9559, password: str = "nao"):
        self._ip = ip
        self._port = naoqi_port
        self._password = password

    def connect(self) -> None:
        """Connect to NAO via naoqi ALProxy.

        Would create proxies for ALMotion, ALRobotPosture, etc.
        """
        raise NotImplementedError(
            "RealNAO requires the naoqi Python SDK. "
            "Install it from Aldebaran's developer portal."
        )

    def disconnect(self) -> None:
        """Disconnect from NAO and release resources."""
        raise NotImplementedError("naoqi SDK required")

    def on_response(self, text: str, hostility_level: int) -> None:
        """Trigger appropriate gestures via ALMotion.setAngles().

        Higher hostility levels should produce more emphatic arm
        movements. Reference the ANGLES_SPEAKING constants in
        nao_speaker_server.py for angle ranges.
        """
        raise NotImplementedError("naoqi SDK required")

    def on_listening(self) -> None:
        """Set listening posture via ALMotion.setAngles().

        Should move the right hand near ear (ANGLES_LISTENING)
        and keep the left arm relaxed.
        """
        raise NotImplementedError("naoqi SDK required")

    def on_idle(self) -> None:
        """Return to neutral via ALRobotPosture.goToPosture("StandInit", 0.5)."""
        raise NotImplementedError("naoqi SDK required")

    def is_connected(self) -> bool:
        """Always returns False since connection is not implemented."""
        return False
