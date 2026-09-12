"""Windows-friendly command-line entry point for Local Autonomous Dev Team."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .orchestrator import AutonomousRunner
from .dashboard import DashboardController, serve_dashboard
from .providers import OllamaProvider, RoleModelProvider, ScriptedProvider
from .runtime import ApplicationScreenshotPipeline
from .state_store import StateStore
from .tools import WorkspaceTools


@dataclass(frozen=True, slots=True)
class AgentProfile:
    model: str
    temperature: float = 0.1
    timeout: float = 120.0
    retries: int = 2
    context_budget: int = 12000
    context_limit: int = 16384
    keep_alive: str = "10m"


@dataclass(frozen=True, slots=True)
class VisualConfig:
    command: tuple[str, ...] = ()
    url: str = ""
    health_url: str = ""
    ready_timeout: float = 30.0
    max_repairs: int = 2


@dataclass(frozen=True, slots=True)
class PermissionConfig:
    allow_project_dependency_install: bool = True
    allow_system_package_install: bool = False


@dataclass(frozen=True, slots=True)
class AppConfig:
    model: str = "qwen3:14b"
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    max_attempts: int = 3
    profiles: dict[str, AgentProfile] = field(default_factory=dict)
    visual: VisualConfig = field(default_factory=VisualConfig)
    permissions: PermissionConfig = field(default_factory=PermissionConfig)


def load_config(path: Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    ollama = config.get("ollama", {})
    runner = config.get("runner", {})
    models = config.get("models", {})
    agents = config.get("agents", {})
    visual = config.get("visual", {})
    permissions = config.get("permissions", {})
    if not all(isinstance(value, dict) for value in (ollama, runner, models, agents, visual, permissions)):
        raise ValueError("[ollama], [runner], [models], [agents], and [visual] must be TOML tables")
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
            context_limit=int(settings.get("context_limit", 16384)),
            keep_alive=str(settings.get("keep_alive", "10m")),
        )
    command = visual.get("command", [])
    if not isinstance(command, list) or not all(isinstance(part, str) and part for part in command):
        raise ValueError("[visual].command must be a string array")
    visual_config = VisualConfig(
        command=tuple(command),
        url=str(visual.get("url", "")),
        health_url=str(visual.get("health_url", "")),
        ready_timeout=float(visual.get("ready_timeout", 30.0)),
        max_repairs=int(visual.get("max_repairs", 2)),
    )
    permission_config = PermissionConfig(bool(permissions.get("allow_project_dependency_install", True)), bool(permissions.get("allow_system_package_install", False)))
    return AppConfig(
        model=default_model,
        base_url=str(ollama.get("base_url", defaults.base_url)),
        timeout=float(ollama.get("timeout", defaults.timeout)),
        max_attempts=int(runner.get("max_attempts", defaults.max_attempts)),
        profiles=profiles,
        visual=visual_config,
        permissions=permission_config,
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
                context_limit=profile.context_limit,
                keep_alive=profile.keep_alive,
            )
            for role, profile in config.profiles.items()
        }
        provider = RoleModelProvider(default_provider, profiles)
    pipeline = None
    if config.visual.command and config.visual.url:
        pipeline = ApplicationScreenshotPipeline(
            workspace, config.visual.command, config.visual.health_url, config.visual.ready_timeout
        )
    return AutonomousRunner(
        workspace=workspace,
        store=StateStore(workspace),
        tools=WorkspaceTools(workspace),
        provider=provider,
        max_attempts=config.max_attempts,
        visual_pipeline=pipeline,
        visual_url=config.visual.url or None,
        max_visual_repairs=config.visual.max_repairs,
        allow_project_dependency_install=config.permissions.allow_project_dependency_install,
        allow_system_package_install=config.permissions.allow_system_package_install,
    )


def format_status(workspace: Path) -> str:
    state = StateStore(workspace).load()
    if state is None:
        raise ValueError("project is not initialized")
    done = len([task for task in state.tasks if task.status.value == "DONE"])
    current = next((task.title for task in state.tasks if task.id == state.current_task_id), "None")
    from .metrics import metrics

    data = metrics(state)
    seconds = int(data["run_duration_seconds"])
    elapsed = f"{seconds // 3600:02}:{(seconds % 3600) // 60:02}:{seconds % 60:02}"
    heartbeat = "healthy" if state.heartbeat.agent != "IDLE" else "idle"
    return (
        f"Project: {workspace.name}\nStatus: {state.status}\n"
        f"Agent: {state.heartbeat.agent}\nTask: {current}\nTasks: {done}/{len(state.tasks)} complete\n"
        f"Repairs: {data['repairs']}\nReviewer rejects: {data['reviewer_rejects']}\n"
        f"Designer: {state.visual_status}\nLLM calls: {data['total_llm_calls']}\nRun time: {elapsed}\n"
        f"Heartbeat: {heartbeat}\nLast checkpoint: {state.last_checkpoint or 'None'}\n"
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
    requirement = commands.add_parser("requirement", help="add a durable requirement amendment")
    requirement_commands = requirement.add_subparsers(dest="requirement_command", required=True)
    add_requirement = requirement_commands.add_parser("add")
    add_requirement.add_argument("workspace", type=Path)
    add_requirement.add_argument("text")
    dashboard = commands.add_parser("dashboard", help="serve a local status and control dashboard")
    dashboard.add_argument("workspace", type=Path)
    dashboard.add_argument("--config", type=Path)
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    e2e = commands.add_parser("e2e", help="run reproducible autonomous endurance scenarios")
    e2e_commands = e2e.add_subparsers(dest="e2e_command", required=True)
    notes = e2e_commands.add_parser("notes", help="run the live Ollama Notes endurance scenario")
    notes.add_argument("--live", action="store_true", help="required acknowledgement that this invokes local Ollama")
    notes.add_argument("--config", type=Path)
    notes.add_argument("--artifacts", type=Path, default=Path(".e2e-runs"))
    notes.add_argument("--max-cycles", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "e2e":
            if args.e2e_command != "notes" or not args.live:
                raise ValueError("only `autodev e2e notes --live` is supported; --live is required")
            from .endurance import run_notes_live

            config = load_config(args.config)
            workspace, summary = run_notes_live(config, args.config, args.artifacts, args.max_cycles)
            print(f"Notes endurance workspace: {workspace}\nStatus: {summary['status']}")
            return 0
        workspace = args.workspace.resolve()
        if args.command == "init":
            ensure_workspace(workspace)
            spec = args.spec_file.read_text(encoding="utf-8") if args.spec_file else args.spec
            make_runner(workspace, AppConfig(), scripted=True).initialize(spec)
            print(format_status(workspace))
            return 0
        if args.command == "status":
            print(format_status(workspace))
            return 0
        if args.command == "requirement":
            runner = make_runner(workspace, AppConfig(), scripted=True)
            runner.add_requirement(args.text)
            print(format_status(workspace))
            return 0
        if args.command == "dashboard":
            config = load_config(args.config)
            runner = make_runner(workspace, config)
            serve_dashboard(StateStore(workspace), DashboardController(runner), args.host, args.port)
            return 0
        if args.command in {"pause", "stop"}:
            runner = make_runner(workspace, AppConfig(), scripted=True)
            getattr(runner, args.command)()
            print(format_status(workspace))
            return 0
        config = load_config(args.config)
        if args.model:
            config = AppConfig(args.model, config.base_url, config.timeout, config.max_attempts, config.profiles, config.visual, config.permissions)
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
