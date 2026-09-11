"""Bounded local application lifecycle and screenshot acquisition."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4


def free_port() -> int:
    """Return a currently free loopback port; callers start immediately after."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def kill_process_tree(pid: int) -> None:
    """Terminate a whole server tree, including a Windows Flask reloader child."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False, timeout=10)
        return
    try:
        os.killpg(pid, 15)
    except (ProcessLookupError, PermissionError):
        return


@dataclass(slots=True)
class ProcessRecord:
    id: str
    pid: int
    command: list[str]
    cwd: str
    started_at: str
    purpose: str
    expected_port: int | None
    status: str
    log_dir: str
    ownership_token: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ManagedProcess:
    command: list[str]
    timeout: float = 30.0
    cwd: Path | None = None
    env: dict[str, str] | None = None
    log_dir: Path | None = None
    process: subprocess.Popen[str] | None = None
    _log_handle: object | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("application process is already running")
        options: dict[str, object] = {"text": True, "shell": False}
        if self.cwd is not None:
            options["cwd"] = self.cwd
        if self.env is not None:
            options["env"] = self.env
        if self.log_dir is None:
            options.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_handle = (self.log_dir / "stdout.log").open("a", encoding="utf-8", buffering=1)
            options.update(stdout=self._log_handle, stderr=subprocess.STDOUT)
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        self.process = subprocess.Popen(self.command, **options)  # type: ignore[arg-type]

    def wait_ready(self, host: str, port: int, health_url: str | None = None, timeout: float | None = None) -> bool:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
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
        return False

    def logs(self) -> str:
        if self.log_dir is not None:
            path = self.log_dir / "stdout.log"
            return path.read_text(encoding="utf-8", errors="replace")[-12000:] if path.is_file() else ""
        if self.process is None or self.process.stdout is None or self.process.poll() is None:
            return ""
        try:
            output, _ = self.process.communicate(timeout=0.2)
        except subprocess.TimeoutExpired:
            return ""
        return (output or "")[-12000:]

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                kill_process_tree(self.process.pid)
                try:
                    self.process.wait(timeout=min(max(self.timeout, 1), 10))
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            if self._log_handle is not None:
                self._log_handle.close()  # type: ignore[union-attr]
                self._log_handle = None


class ManagedProcessManager:
    """Owns non-finite commands, their logs, and their durable lifecycle records."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self._processes: dict[str, ManagedProcess] = {}
        self._records: dict[str, ProcessRecord] = {}

    def start(self, command: list[str], *, purpose: str, expected_port: int | None = None, env: dict[str, str] | None = None, timeout: float = 30.0) -> ProcessRecord:
        command = self._automated_command(command)
        identifier = str(uuid4())
        log_dir = self.workspace / ".autodev" / "processes" / identifier
        launch_env = os.environ.copy()
        # Flask CLI respects these; app code that already uses debug=False is unchanged.
        launch_env.update({"FLASK_DEBUG": "0", "FLASK_ENV": "production", "WERKZEUG_DEBUG_PIN": "off"})
        if env:
            launch_env.update(env)
        launcher_command = [sys.executable, "-m", "autodev.process_launcher", identifier, "--", *command]
        process = ManagedProcess(launcher_command, timeout=timeout, cwd=self.workspace, env=launch_env, log_dir=log_dir)
        process.start()
        if process.process is None:
            raise RuntimeError("managed process did not start")
        record = ProcessRecord(identifier, process.process.pid, list(command), str(self.workspace), datetime.now(UTC).isoformat(), purpose, expected_port, "STARTED", str(log_dir), identifier)
        self._processes[identifier] = process
        self._records[identifier] = record
        return record

    @staticmethod
    def _automated_command(command: list[str]) -> list[str]:
        """Disable Flask's reloader/debugger when its supported CLI is used."""
        joined = " ".join(command).lower()
        if "flask run" in joined:
            result = list(command)
            if "--no-reload" not in result:
                result.append("--no-reload")
            if "--no-debugger" not in result:
                result.append("--no-debugger")
            return result
        return list(command)

    def wait_ready(self, identifier: str, host: str, port: int, *, health_url: str | None = None, timeout: float = 30.0) -> bool:
        ready = self._processes[identifier].wait_ready(host, port, health_url, timeout)
        self._records[identifier].status = "READY" if ready else "FAILED"
        return ready

    def stop(self, identifier: str) -> None:
        process = self._processes.get(identifier)
        if process is not None:
            process.stop()
        if identifier in self._records:
            self._records[identifier].status = "STOPPED"

    def stop_all(self) -> None:
        for identifier in list(self._processes):
            self.stop(identifier)

    def records(self) -> list[dict[str, object]]:
        return [record.to_dict() for record in self._records.values()]

    def logs(self, identifier: str) -> str:
        process = self._processes.get(identifier)
        return process.logs() if process is not None else ""

    def recover(self, records: list[dict[str, object]]) -> list[str]:
        """Never kill a PID unless its live command line matches this owned record."""
        messages: list[str] = []
        for raw in records:
            identifier = str(raw.get("id", "unknown"))
            pid, command = raw.get("pid"), raw.get("command")
            token = raw.get("ownership_token")
            if not isinstance(pid, int) or not isinstance(command, list) or not all(isinstance(item, str) for item in command) or not isinstance(token, str) or not token:
                messages.append(f"Skipped invalid managed process record {identifier}")
            elif not self._command_matches(pid, token):
                messages.append(f"Skipped unverified managed process {identifier}")
            else:
                kill_process_tree(pid)
                messages.append(f"Cleaned stale managed process {identifier}")
        return messages

    @staticmethod
    def _command_matches(pid: int, token: str) -> bool:
        if os.name == "nt":
            probe = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"], capture_output=True, text=True, check=False, timeout=5)
            actual = probe.stdout.lower()
        else:
            proc = Path(f"/proc/{pid}/cmdline")
            actual = proc.read_bytes().decode("utf-8", errors="replace").replace("\0", " ").lower() if proc.exists() else ""
        return "autodev.process_launcher" in actual and token.lower() in actual


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

    def __init__(self, workspace: Path, command: tuple[str, ...], health_url: str = "", ready_timeout: float = 30.0, capture: object | None = None) -> None:
        super().__init__(workspace, capture=capture)
        self.command = command
        self.health_url = health_url
        self.ready_timeout = ready_timeout

    def capture(self, url: str) -> list[Path]:
        port = free_port()
        resolved_url = url.replace("{port}", str(port))
        resolved_health = self.health_url.replace("{port}", str(port)) if self.health_url else ""
        command = [part.replace("{port}", str(port)) for part in self.command]
        manager = ManagedProcessManager(self.workspace)
        record = manager.start(command, purpose="screenshot", expected_port=port, timeout=self.ready_timeout)
        try:
            if not manager.wait_ready(record.id, "127.0.0.1", port, health_url=resolved_health or None, timeout=self.ready_timeout):
                self.last_diagnostic = "Application readiness timed out. Logs:\n" + manager.logs(record.id)
                return []
            return super().capture(resolved_url)
        finally:
            manager.stop(record.id)
