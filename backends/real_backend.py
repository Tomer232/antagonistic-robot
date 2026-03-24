# backends/real_backend.py
# Runs on your PC (Python 3).
# speak(): sends text to nao_speaker_server.py running on NAO robot over TCP (text protocol).
# record_from_nao(): records audio on NAO mic via SSH+SFTP using paramiko.

import os
import socket
import tempfile
import wave

from nao_interface import NaoInterface

NAO_TCP_PORT = 9600
SOCKET_TIMEOUT = 30   # seconds to wait for robot to finish speaking


class RealNao(NaoInterface):
    """
    v2 Real NAO backend.

    speak(): TCP text protocol → nao_speaker_server.py → ALTextToSpeech
    record_from_nao(): SSH to robot → ALAudioRecorder → SFTP fetch → raw PCM bytes
    """

    def __init__(self, robot_ip, tcp_port=NAO_TCP_PORT,
                 robot_port=9559, robot_password="nao"):
        self.robot_ip = robot_ip
        self.tcp_port = tcp_port
        self.robot_port = robot_port          # NAOqi port (stored for reference)
        self.robot_password = robot_password
        self._ssh = None                      # persistent paramiko SSH connection

        print("[REAL NAO] v2 backend. Robot IP: {}, TCP port: {}".format(
            robot_ip, tcp_port
        ))

    # ------------------------------------------------------------------
    # speak() — TCP text protocol (unchanged from v1)
    # ------------------------------------------------------------------

    def speak(self, text):
        """
        Send text to nao_speaker_server.py over TCP.
        Robot speaks via ALTextToSpeech and replies "ok\n" when done.
        Returns True on success, False on error.
        """
        if not text:
            return True
        print("[REAL NAO] Sending to robot:", text[:80])
        try:
            with socket.create_connection(
                (self.robot_ip, self.tcp_port), timeout=SOCKET_TIMEOUT
            ) as s:
                s.sendall((text.strip() + "\n").encode("utf-8"))
                s.shutdown(socket.SHUT_WR)
                # Wait for acknowledgement
                response = b""
                while True:
                    chunk = s.recv(64)
                    if not chunk:
                        break
                    response += chunk
                    if b"ok" in response:
                        break
            return True
        except Exception as e:
            print("[REAL NAO] Socket error:", e)
            return False

    # ------------------------------------------------------------------
    # listen() — not used by continuous loop (StreamingTranscriber handles ASR)
    # ------------------------------------------------------------------

    def listen(self):
        """
        Not used in v2 continuous mode.
        StreamingTranscriber calls record_from_nao() directly as a record_fn.
        """
        raise NotImplementedError(
            "Use StreamingTranscriber(record_fn=robot.record_from_nao) instead."
        )

    # ------------------------------------------------------------------
    # record_from_nao() — NAO mic via SSH + SFTP
    # ------------------------------------------------------------------

    def record_from_nao(self, duration_sec=3, sample_rate=16000):
        """
        Record audio from the NAO robot's microphone.

        Protocol:
            1. SSH to robot as user "nao"
            2. Run Python 2 one-liner to record via ALAudioRecorder
            3. SFTP-fetch the WAV file from /tmp/nao_chunk.wav
            4. Read and return raw PCM frames (int16 bytes)
            5. Delete temp file locally; remote file cleaned on next call

        Args:
            duration_sec: recording duration in seconds (usually 3)
            sample_rate: sample rate (16000 Hz)

        Returns:
            bytes -- raw int16 PCM audio, or b"" on error
        """
        try:
            import paramiko
        except ImportError:
            print("[REAL NAO] paramiko not installed. Run: pip install paramiko")
            return b""

        remote_path = "/tmp/nao_chunk.wav"

        # NAOqi's Python lives at a specific path on the robot.
        # Plain "python" on the robot does NOT have naoqi on its PYTHONPATH.
        nao_cmd = (
            "PYTHONPATH=/opt/aldebaran/lib/python2.7/site-packages "
            "/usr/bin/python2.7 -c \""
            "from naoqi import ALProxy; import time; "
            "rec = ALProxy('ALAudioRecorder', '127.0.0.1', 9559); "
            "rec.startMicrophonesRecording('{remote}', 'wav', {rate}, (0,0,1,0)); "
            "time.sleep({dur}); "
            "rec.stopMicrophonesRecording()"
            "\""
        ).format(remote=remote_path, rate=sample_rate, dur=float(duration_sec))

        local_tmp = tempfile.mktemp(suffix="_nao.wav")
        try:
            ssh = self._get_ssh()
            if ssh is None:
                return b""

            # Execute recording command; blocks until NAO finishes
            _, stdout, stderr = ssh.exec_command(nao_cmd)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                err = stderr.read().decode("utf-8", errors="replace").strip()
                print("[REAL NAO] Recording command failed (exit {}): {}".format(
                    exit_status, err[:200]
                ))
                return b""

            # Fetch the WAV file via SFTP
            sftp = ssh.open_sftp()
            try:
                sftp.get(remote_path, local_tmp)
                # Remove from robot to keep /tmp clean
                try:
                    sftp.remove(remote_path)
                except Exception:
                    pass
            finally:
                sftp.close()

            # Extract raw PCM frames from WAV
            with wave.open(local_tmp, "rb") as wf:
                pcm_bytes = wf.readframes(wf.getnframes())
            return pcm_bytes

        except Exception as e:
            print("[REAL NAO] record_from_nao error:", e)
            self._ssh = None   # force reconnect next call
            return b""
        finally:
            try:
                os.remove(local_tmp)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # SSH connection management
    # ------------------------------------------------------------------

    def _get_ssh(self):
        """
        Return a live paramiko SSH connection to the robot.
        Reconnects if the connection is broken or not yet established.
        """
        try:
            import paramiko
        except ImportError:
            return None

        # Check if existing connection is still alive
        if self._ssh is not None:
            transport = self._ssh.get_transport()
            if transport is not None and transport.is_active():
                return self._ssh
            # Connection is dead; clean up
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None

        # Establish new connection
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                self.robot_ip,
                username="nao",
                password=self.robot_password,
                timeout=5,
                look_for_keys=False,
                allow_agent=False,
            )
            self._ssh = ssh
            print("[REAL NAO] SSH connected to", self.robot_ip)
            return self._ssh
        except Exception as e:
            print("[REAL NAO] SSH connect error:", e)
            self._ssh = None
            return None

    def shutdown(self):
        """Close SSH connection cleanly on session end."""
        if self._ssh is not None:
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None
