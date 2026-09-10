"""Objective visual QA role with optional Ollama image input."""
from __future__ import annotations
import base64
from dataclasses import dataclass
from pathlib import Path
from .models import ProjectState, Task
from .providers import AgentRequest, LLMProvider

_CATEGORIES = {"layout", "overflow", "alignment", "readability", "missing_element", "broken_state", "responsive_issue", "visual_glitch"}
_SEVERITIES = {"low", "medium", "high", "critical"}

@dataclass(frozen=True, slots=True)
class VisualIssue:
    severity: str
    category: str
    description: str

@dataclass(frozen=True, slots=True)
class VisualReview:
    verdict: str
    issues: list[VisualIssue]

def detect_ui_project(workspace: Path) -> bool:
    return any((workspace / name).exists() for name in ("package.json", "index.html", "templates"))

class DesignerAgent:
    def __init__(self, provider: LLMProvider) -> None: self.provider = provider
    def review(self, screenshot: Path, requirements: str, state: ProjectState, task: Task, viewport: dict[str, object]) -> VisualReview:
        if not screenshot.is_file(): raise ValueError("screenshot does not exist")
        encoded = base64.b64encode(screenshot.read_bytes()).decode("ascii")
        prompt = f"Review only objective UI defects. Requirements: {requirements}. Task: {task.title}. Viewport: {viewport}. Return a JSON object with verdict PASS or FAIL and issues array."
        data = self.provider.complete(AgentRequest(role="DESIGNER", prompt=prompt, system_prompt="You are Designer. Report objective UI defects only. JSON only.", images=(encoded,))).data
        verdict, issues = data.get("verdict"), data.get("issues")
        if verdict not in {"PASS", "FAIL"} or not isinstance(issues, list): raise ValueError("invalid visual verdict")
        parsed = []
        for issue in issues:
            if not isinstance(issue, dict): raise ValueError("visual issue must be object")
            severity, category, description = issue.get("severity"), issue.get("category"), issue.get("description")
            if severity not in _SEVERITIES: raise ValueError("invalid severity")
            if category not in _CATEGORIES: raise ValueError("invalid visual category")
            if not isinstance(description, str) or not description.strip(): raise ValueError("invalid visual description")
            parsed.append(VisualIssue(severity, category, description))
        return VisualReview(verdict, parsed)
