from pathlib import Path

from autodev.runtime import ApplicationScreenshotPipeline, free_port

def test_free_port_is_bindable() -> None:
    assert 0 < free_port() < 65536


def test_screenshot_pipeline_discovers_root_app_when_configured_backend_path_is_absent(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n", encoding="utf-8")
    pipeline = ApplicationScreenshotPipeline(tmp_path, ("python", "backend/app.py"))

    assert pipeline.resolve_command(49123) == ["python", "app.py"]
