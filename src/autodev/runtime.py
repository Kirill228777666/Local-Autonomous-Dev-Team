"""Bounded local application lifecycle and screenshot acquisition."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def free_port() -> int:
    """Return a currently free loopback port; callers start immediately after."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class ManagedProcess:
    command: list[str]
    timeout: float = 30.0
    cwd: Path | None = None
    process: subprocess.Popen[str] | None = None
    _logs: list[str] = field(default_factory=list, init=False, repr=False)

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("application process is already running")
        options: dict[str, object] = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
        if self.cwd is not None:
            options["cwd"] = self.cwd
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        self.process = subprocess.Popen(self.command, **options)  # type: ignore[arg-type]

    def wait_ready(self, host: str, port: int, health_url: str | None = None) -> bool:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                self._collect_logs()
                return False
            if health_url:
                try:
                    with urlopen(health_url, timeout=0.5) as response:
                        if 200 <= response.status < 500:
                            return True
                except (URLError, OSError):
                    pass
            else:
                try:
                    with socket.create_connection((host, port), timeout=0.2):
                        return True
                except OSError:
                    pass
            time.sleep(0.1)
        self._collect_logs()
        return False

    def logs(self) -> str:
        self._collect_logs()
        return "".join(self._logs)[-12000:]

    def _collect_logs(self) -> None:
        if self.process is None or self.process.stdout is None or self.process.poll() is None:
            return
        try:
            output, _ = self.process.communicate(timeout=0.2)
        except subprocess.TimeoutExpired:
            return
        if output:
            self._logs.append(output)

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self._collect_logs()
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                self.process.terminate()
            self.process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=self.timeout)
        finally:
            self._collect_logs()


class ScreenshotPipeline:
    """Capture desktop and narrow screenshots using optional Playwright."""

    def __init__(self, workspace: Path, capture: object | None = None) -> None:
        self.workspace = workspace.resolve()
        self.capture_function = capture
        self.last_diagnostic = ""

    def capture(self, url: str) -> list[Path]:
        directory = self.workspace / ".autodev" / "artifacts" / "screenshots"
        directory.mkdir(parents=True, exist_ok=True)
        artifacts: list[Path] = []
        try:
            for name, width, height in (("desktop.png", 1440, 1000), ("narrow.png", 390, 844)):
                output = directory / name
                if self.capture_function is not None:
                    self.capture_function(url, output, width, height)  # type: ignore[operator]
                else:
                    self._playwright_capture(url, output, width, height)
                if not output.is_file() or not output.stat().st_size:
                    raise RuntimeError(f"screenshot was not produced: {output.name}")
                artifacts.append(output)
        except Exception as error:
            self.last_diagnostic = f"Screenshot unavailable: {error}"
            return []
        self.last_diagnostic = ""
        return artifacts

    @staticmethod
    def _playwright_capture(url: str, output: Path, width: int, height: int) -> None:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("Playwright Python package/browser is unavailable") from error
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": width, "height": height})
                page.goto(url, wait_until="networkidle", timeout=30_000)
                page.screenshot(path=str(output), full_page=True)
            finally:
                browser.close()


class ApplicationScreenshotPipeline(ScreenshotPipeline):
    """Start one configured local application, wait by condition, then capture it."""

    def __init__(
        self,
        workspace: Path,
        command: tuple[str, ...],
        health_url: str = "",
        ready_timeout: float = 30.0,
        capture: object | None = None,
    ) -> None:
        super().__init__(workspace, capture=capture)
        self.command = command
        self.health_url = health_url
        self.ready_timeout = ready_timeout

    def capture(self, url: str) -> list[Path]:
        port = free_port()
        resolved_url = url.replace("{port}", str(port))
        resolved_health = self.health_url.replace("{port}", str(port)) if self.health_url else ""
        command = [part.replace("{port}", str(port)) for part in self.command]
        process = ManagedProcess(command, timeout=self.ready_timeout, cwd=self.workspace)
        process.start()
        try:
            if not process.wait_ready("127.0.0.1", port, resolved_health or None):
                self.last_diagnostic = "Application readiness timed out. Logs:\n" + process.logs()
                return []
            return super().capture(resolved_url)
        finally:
            process.stop()
