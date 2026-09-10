from pathlib import Path

import pytest

from autodev.designer import DesignerAgent, VisualIssue, VisualReview, detect_ui_project
from autodev.models import ProjectState, Task
from autodev.providers import AgentReply, ScriptedProvider


def test_designer_returns_structured_visual_review(tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(b"png")
    provider = ScriptedProvider({"DESIGNER": [AgentReply({"verdict": "FAIL", "issues": [{"severity": "high", "category": "overflow", "description": "Input overlaps."}]})]})
    review = DesignerAgent(provider).review(image, "responsive UI", ProjectState.create("Build UI"), Task.create("UI", "Build UI"), {"width": 390})
    assert review.verdict == "FAIL"
    assert review.issues[0].category == "overflow"


def test_designer_rejects_unknown_visual_categories(tmp_path: Path) -> None:
    image = tmp_path / "screen.png"; image.write_bytes(b"png")
    provider = ScriptedProvider({"DESIGNER": [AgentReply({"verdict": "FAIL", "issues": [{"severity": "low", "category": "taste", "description": "bad"}]})]})
    with pytest.raises(ValueError, match="category"):
        DesignerAgent(provider).review(image, "UI", ProjectState.create("UI"), Task.create("UI", "UI"), {})


def test_ui_detection_uses_frontend_files(tmp_path: Path) -> None:
    assert detect_ui_project(tmp_path) is False
    (tmp_path / "index.html").write_text("<main>", encoding="utf-8")
    assert detect_ui_project(tmp_path) is True
