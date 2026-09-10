"""Safe, deterministic discovery and execution of project-wide regression checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .tools import CommandResult, WorkspaceTools


@dataclass(frozen=True, slots=True)
class RegressionCheck:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class RegressionResult:
    results: list[RegressionCheck]

    @property
    def passed(self) -> bool:
        return all(result.exit_code == 0 for result in self.results)

    @property
    def summary(self) -> str:
        if not self.results:
            return "No applicable regression commands detected."
        status = "passed" if self.passed else "failed"
        lines = [f"Full regression {status}: {len(self.results)} command(s)."]
        lines.extend(
            f"{' '.join(result.command)} → exit {result.exit_code}; {result.stdout.strip() or result.stderr.strip()}"
            for result in self.results
        )
        return "\n".join(lines)


class RegressionRunner:
    def __init__(self, workspace: Path, tools: WorkspaceTools) -> None:
        self.workspace = workspace.resolve()
        self.tools = tools

    def detect_commands(self) -> list[list[str]]:
        commands: list[list[str]] = []
        if (self.workspace / "tests").is_dir() or (self.workspace / "pyproject.toml").exists():
            commands.append(["py", "-3", "-m", "pytest", "-q"])
        package_path = self.workspace / "package.json"
        if package_path.exists():
            package = json.loads(package_path.read_text(encoding="utf-8"))
            scripts = package.get("scripts", {})
            if isinstance(scripts, dict):
                for name in ("test", "build", "lint", "typecheck"):
                    if isinstance(scripts.get(name), str):
                        commands.append(["npm", "test"] if name == "test" else ["npm", "run", name])
        return commands

    def run(self) -> RegressionResult:
        checks: list[RegressionCheck] = []
        for command in self.detect_commands():
            result: CommandResult = self.tools.run_tests(command)
            checks.append(RegressionCheck(command, result.exit_code, result.stdout, result.stderr))
        return RegressionResult(checks)
