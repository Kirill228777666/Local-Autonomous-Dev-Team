from pathlib import Path

from autodev.cli import AppConfig, VisualConfig, format_status, load_config, main
from autodev.state_store import StateStore


def test_cli_initializes_workspace_and_exposes_status_controls(tmp_path: Path, capsys: object) -> None:
    workspace = tmp_path / "my-app"

    assert main(["init", str(workspace), "--spec", "Build a local TODO application"]) == 0
    assert main(["status", str(workspace)]) == 0
    assert "Status: READY" in capsys.readouterr().out  # type: ignore[attr-defined]
    assert main(["pause", str(workspace)]) == 0
    assert StateStore(workspace).load().status == "PAUSED"  # type: ignore[union-attr]
    assert main(["stop", str(workspace)]) == 0
    assert StateStore(workspace).load().status == "STOPPED"  # type: ignore[union-attr]


def test_load_config_reads_ollama_model_and_timeout(tmp_path: Path) -> None:
    config_path = tmp_path / "autodev.toml"
    config_path.write_text(
        "[ollama]\nmodel = 'qwen3-coder'\ntimeout = 45\n[runner]\nmax_attempts = 4\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config == AppConfig(model="qwen3-coder", timeout=45, max_attempts=4)


def test_load_config_reads_role_model_profile_with_generation_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "profiles.toml"
    config_path.write_text(
        "[models]\ndefault = 'qwen3:14b'\ncoder = 'qwen2.5-coder:7b'\n"
        "[agents.coder]\ntemperature = 0.05\ntimeout = 60\nretries = 1\ncontext_budget = 6000\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model == "qwen3:14b"
    assert config.profiles["CODER"].model == "qwen2.5-coder:7b"
    assert config.profiles["CODER"].context_budget == 6000


def test_cli_adds_requirement_as_a_durable_amendment(tmp_path: Path) -> None:
    workspace = tmp_path / "app"
    assert main(["init", str(workspace), "--spec", "Build notes"]) == 0

    assert main(["requirement", "add", str(workspace), "Add JSON export"]) == 0

    state = StateStore(workspace).load()
    assert state.amendments == ["Add JSON export"]  # type: ignore[union-attr]
    assert state.tasks[-1].title == "Implement amendment: Add JSON export"  # type: ignore[union-attr]


def test_config_and_compact_status_expose_visual_runtime_and_recovery_data(tmp_path: Path) -> None:
    config_path = tmp_path / "visual.toml"
    config_path.write_text(
        "[visual]\ncommand = ['py', '-3', '-m', 'http.server', '{port}']\nurl = 'http://127.0.0.1:{port}'\nhealth_url = 'http://127.0.0.1:{port}'\nready_timeout = 11\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.visual == VisualConfig(
        command=("py", "-3", "-m", "http.server", "{port}"),
        url="http://127.0.0.1:{port}",
        health_url="http://127.0.0.1:{port}",
        ready_timeout=11,
    )

    workspace = tmp_path / "notes"
    assert main(["init", str(workspace), "--spec", "Build Notes"]) == 0
    state = StateStore(workspace).load()
    state.run_history.append("Run resumed")  # type: ignore[union-attr]
    state.visual_status = "PASS"  # type: ignore[union-attr]
    StateStore(workspace).save(state)  # type: ignore[arg-type]
    status = format_status(workspace)
    assert "Project: notes" in status
    assert "Designer: PASS" in status
    assert "Run time:" in status
