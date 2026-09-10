"""Bounded local application process lifecycle helpers."""
from __future__ import annotations
import socket
import subprocess
import time
from dataclasses import dataclass

def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

@dataclass
class ManagedProcess:
    command: list[str]
    timeout: float = 30.0
    process: subprocess.Popen[str] | None = None
    def start(self) -> None:
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired: self.process.kill()
    def wait_ready(self, host: str, port: int) -> bool:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.2): return True
            except OSError: time.sleep(0.1)
        return False
