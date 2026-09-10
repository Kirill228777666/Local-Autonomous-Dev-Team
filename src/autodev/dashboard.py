"""Small local-only web dashboard for observing and controlling one run."""

from __future__ import annotations

import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from .orchestrator import AutonomousRunner
from .state_store import StateStore


def render_dashboard(store: StateStore) -> str:
    state = store.load()
    if state is None:
        raise ValueError("project is not initialized")
    tasks = "".join(
        f"<li><strong>{html.escape(task.status.value)}</strong> {html.escape(task.title)}</li>" for task in state.tasks
    ) or "<li>No tasks planned yet.</li>"
    events = "\n".join(state.run_history[-30:]) or "No activity yet."
    heartbeat = state.heartbeat
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>AutoDev Dashboard</title>
<style>body{{font-family:system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}}pre{{white-space:pre-wrap;background:#f4f4f4;padding:1rem}}button{{margin:.25rem;padding:.5rem 1rem}}</style></head>
<body><h1>{html.escape(store.workspace.name)}</h1><p><b>Status:</b> {html.escape(state.status)} · <b>Model:</b> {html.escape(state.model)}</p>
<p><b>Heartbeat:</b> {html.escape(heartbeat.timestamp)} — {html.escape(heartbeat.agent)} / {html.escape(heartbeat.phase)}</p>
<form method=\"post\">{''.join(f'<button name="action" value="{action.lower()}">{action}</button>' for action in ('START','PAUSE','RESUME','STOP'))}</form>
<h2>Original specification</h2><pre>{html.escape(state.original_spec)}</pre><h2>Progress</h2><ul>{tasks}</ul>
<h2>Final QA</h2><pre>{html.escape(state.final_qa_status)}\n{html.escape(chr(10).join(state.final_qa_findings))}</pre>
<h2>Recent activity</h2><pre>{html.escape(events)}</pre></body></html>"""


class DashboardController:
    def __init__(self, runner: AutonomousRunner) -> None:
        self.runner = runner

    def apply(self, action: str) -> None:
        if action == "pause":
            self.runner.pause()
        elif action == "stop":
            self.runner.stop()
        elif action == "resume":
            self.runner.resume()
            threading.Thread(target=self.runner.run, daemon=True).start()
        elif action == "start":
            threading.Thread(target=self.runner.run, daemon=True).start()
        else:
            raise ValueError(f"unsupported dashboard action: {action}")


def serve_dashboard(store: StateStore, controller: DashboardController, host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._respond()

        def do_POST(self) -> None:  # noqa: N802
            size = int(self.headers.get("Content-Length", "0"))
            action = parse_qs(self.rfile.read(size).decode("utf-8")).get("action", [""])[0]
            try:
                controller.apply(action)
                self.send_response(303)
                self.send_header("Location", "/")
                self.end_headers()
            except ValueError as error:
                self.send_error(400, str(error))

        def _respond(self) -> None:
            body = render_dashboard(store).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Dashboard available at http://{host}:{port}")
    server.serve_forever()
