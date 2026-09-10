"""Windows-friendly command-line entry point for Local Autonomous Dev Team."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .orchestrator import AutonomousRunner
from .providers import OllamaProvider, RoleModelProvider, ScriptedProvider
from .state_store import StateStore
from .tools import WorkspaceTools


@dataclass(frozen=True, slots=True)
class AgentProfile:
    model: str
    temperature: float = 0.1
    timeout: float = 120.0
    retries: int = 2
    context_budget: int = 12000


@dataclass(frozen=True, slots=True)
class AppConfig:
    model: str = "qwen3:14b"
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    max_attempts: int = 3
    profiles: dict[str, AgentProfile] = field(default_factory=dict)


def load_config(path: Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    ollama = config.get("ollama", {})
    runner = config.get("runner", {})
    models = config.get("models", {})
    agents = config.get("agents", {})
    if not all(isinstance(value, dict) for value in (ollama, runner, models, agents)):
        raise ValueError("[ollama], [runner], [models], and [agents] must be TOML tables")
    defaults = AppConfig()
    default_model = str(models.get("default", ollama.get("model", defaults.model)))
    profile_names = set(models).union(agents) - {"default"}
    profiles: dict[str, AgentProfile] = {}
    for name in profile_names:
        settings = agents.get(name, {})
        if not isinstance(settings, dict):
            raise ValueError(f"[agents.{name}] must be a TOML table")
        profiles[name.upper()] = AgentProfile(
            model=str(models.get(name, default_model)),
            temperature=float(settings.get("temperature", 0.1)),
            timeout=float(settings.get("timeout", ollama.get("timeout", defaults.timeout))),
            retries=int(settings.get("retries", 2)),
            context_budget=int(settings.get("context_budget", 12000)),
        )
    return AppConfig(
        model=default_model,
        base_url=str(ollama.get("base_url", defaults.base_url)),
        timeout=float(ollama.get("timeout", defaults.timeout)),
        max_attempts=int(runner.get("max_attempts", defaults.max_attempts)),
        profiles=profiles,
    )


def ensure_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / ".git").exists():
        _git(workspace, "init")
    gitignore = workspace / ".gitignore"
    lines = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    if ".autodev/" not in lines:
        lines.append(".autodev/")
        gitignore.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not _git(workspace, "config", "user.email", check=False).strip():
        _git(workspace, "config", "user.email", "autodev@local")
    if not _git(workspace, "config", "user.name", check=False).strip():
        _git(workspace, "config", "user.name", "Local Autonomous Dev Team")


def _git(workspace: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(["git", *args], cwd=workspace, capture_output=True, text=True, check=False)
    if check and completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout


def make_runner(workspace: Path, config: AppConfig, scripted: bool = False) -> AutonomousRunner:
    if scripted:
        provider = ScriptedProvider({})
    else:
        default_provider = OllamaProvider(model=config.model, base_url=config.base_url, timeout=config.timeout)
        profiles = {
            role: OllamaProvider(
                model=profile.model,
                base_url=config.base_url,
                temperature=profile.temperature,
                timeout=profile.timeout,
                retries=profile.retries,
            )
            for role, profile in config.profiles.items()
        }
        provider = RoleModelProvider(default_provider, profiles)
    return AutonomousRunner(
        workspace=workspace,
        store=StateStore(workspace),
        tools=WorkspaceTools(workspace),
        provider=provider,
        max_attempts=config.max_attempts,
    )


def format_status(workspace: Path) -> str:
    state = StateStore(workspace).load()
    if state is None:
        raise ValueError("project is not initialized")
    done = len([task for task in state.tasks if task.status.value == "DONE"])
    current = next((task.title for task in state.tasks if task.id == state.current_task_id), "None")
    last_action = state.run_history[-1] if state.run_history else "None"
    last_test = next((entry for entry in reversed(state.run_history) if entry.startswith("Tester")), "None")
    return (
        f"PROJECT\n{workspace.name}\n\nSTATUS\n{state.status}\n\nMODEL\n{state.model}\n\n"
        f"CURRENT AGENT\n{'CODER' if state.current_task_id else 'MANAGER'}\n\nCURRENT TASK\n{current}\n\n"
        f"PROGRESS\n{done} / {len(state.tasks)} major tasks\n\nLAST ACTION\n{last_action}\n\n"
        f"LAST TEST RESULT\n{last_test}\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autodev", description="Local Autonomous Dev Team")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create a workspace from one natural-language specification")
    init.add_argument("workspace", type=Path)
    specification = init.add_mutually_exclusive_group(required=True)
    specification.add_argument("--spec")
    specification.add_argument("--spec-file", type=Path)
    for name in ("start", "resume"):
        run = commands.add_parser(name, help=f"{name} autonomous development")
        run.add_argument("workspace", type=Path)
        run.add_argument("--config", type=Path)
        run.add_argument("--model")
        run.add_argument("--max-cycles", type=int, default=100)
    for name in ("status", "pause", "stop"):
        control = commands.add_parser(name)
        control.add_argument("workspace", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    try:
        if args.command == "init":
            ensure_workspace(workspace)
            spec = args.spec_file.read_text(encoding="utf-8") if args.spec_file else args.spec
            make_runner(workspace, AppConfig(), scripted=True).initialize(spec)
            print(format_status(workspace))
            return 0
        if args.command == "status":
            print(format_status(workspace))
            return 0
        if args.command in {"pause", "stop"}:
            runner = make_runner(workspace, AppConfig(), scripted=True)
            getattr(runner, args.command)()
            print(format_status(workspace))
            return 0
        config = load_config(args.config)
        if args.model:
            config = AppConfig(args.model, config.base_url, config.timeout, config.max_attempts, config.profiles)
        runner = make_runner(workspace, config)
        state = runner._required_state()
        state.model = config.model
        runner.store.save(state)
        if args.command == "resume":
            runner.resume()
        runner.run(args.max_cycles)
        print(format_status(workspace))
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"autodev error: {error}", file=sys.stderr)
        return 1
