"""Deterministic command builders for common validation classes."""

from __future__ import annotations

from pathlib import Path

from .models import Task


class ValidationPlanner:
    """Return a safe validator only when workspace evidence is unambiguous."""

    @staticmethod
    def command_for(workspace: Path, task: Task, environment: dict[str, object]) -> list[str] | None:
        context = environment.get("execution_context")
        interpreter = context.get("python_interpreter") if isinstance(context, dict) else None
        if not isinstance(interpreter, str) or not interpreter:
            return None

        tests = sorted((workspace / "tests").rglob("test*.py")) if (workspace / "tests").is_dir() else []
        if tests:
            pytest_project = any("pytest" in path.read_text(encoding="utf-8", errors="ignore")[:8000] for path in tests)
            if pytest_project:
                return [interpreter, "-m", "pytest", "-q"]
            return [interpreter, "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py"]

        text = f"{task.title} {task.description}".lower()
        frontend = any(word in text for word in ("frontend", "ui", "html", "css", "responsive", "интерфейс", "фронтенд"))
        if frontend and not (workspace / "package.json").is_file():
            patterns = ["*.html"]
            if any(word in text for word in ("style", "css", "responsive", "стил")):
                patterns.append("*.css")
            if any(word in text for word in ("action", "javascript", "search", "filter", "crud", "js", "поиск", "фильтр")):
                patterns.append("*.js")
            checks = "; ".join(
                f"assert any(p.is_file() and p.stat().st_size for p in root.rglob('{pattern}')), '{pattern} missing'"
                for pattern in patterns
            )
            return [interpreter, "-c", f"from pathlib import Path; root=Path('.'); {checks}"]
        return None
