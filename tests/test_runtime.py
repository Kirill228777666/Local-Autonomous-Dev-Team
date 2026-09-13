from pathlib import Path

from autodev.runtime import ApplicationScreenshotPipeline, ManagedProcessManager, free_port

def test_free_port_is_bindable() -> None:
    assert 0 < free_port() < 65536


def test_screenshot_pipeline_discovers_root_app_when_configured_backend_path_is_absent(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n", encoding="utf-8")
    pipeline = ApplicationScreenshotPipeline(tmp_path, ("python", "backend/app.py"))

    assert pipeline.resolve_command(49123) == ["python", "app.py"]


def test_screenshot_pipeline_uses_default_server_port_without_port_control(tmp_path: Path) -> None:
    pipeline = ApplicationScreenshotPipeline(tmp_path, ("python", "app.py"))

    assert pipeline.expected_port() == 5000


def test_screenshot_pipeline_keeps_dynamic_port_for_declared_port_placeholder(tmp_path: Path) -> None:
    pipeline = ApplicationScreenshotPipeline(tmp_path, ("python", "app.py", "--port", "{port}"))

    assert pipeline.expected_port() is None


def test_screenshot_pipeline_can_bind_the_runner_managed_process_manager(tmp_path: Path) -> None:
    pipeline = ApplicationScreenshotPipeline(tmp_path, ("python", "app.py"))
    manager = ManagedProcessManager(tmp_path)

    pipeline.bind_manager(manager)

    assert pipeline.manager is manager
