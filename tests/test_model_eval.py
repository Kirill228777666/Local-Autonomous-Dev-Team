from pathlib import Path

from scripts.model_ab_eval import EvalCase, evaluate_reply


def test_model_eval_applies_targeted_edit_only_inside_disposable_fixture(tmp_path: Path) -> None:
    case = EvalCase(
        name="addition",
        files={"calculator.py": "def add(a, b):\n    return a - b\n"},
        instruction="Fix addition.",
        required={"calculator.py": ("return a + b",)},
    )

    result = evaluate_reply(
        tmp_path,
        case,
        {"actions": [{"kind": "edit_file", "path": "calculator.py", "old": "return a - b", "new": "return a + b"}]},
    )

    assert result["valid_structured_response"] is True
    assert result["valid_tool_actions"] is True
    assert result["successful_task_fix"] is True
    assert result["full_file_rewrites"] == 0
    assert (tmp_path / "calculator.py").read_text(encoding="utf-8").endswith("return a + b\n")


def test_model_eval_scores_noop_and_unnecessary_existing_file_rewrite(tmp_path: Path) -> None:
    case = EvalCase("flag", {"flag.py": "enabled = False\n"}, "Enable it.", {"flag.py": ("True",)})

    noop = evaluate_reply(tmp_path / "noop", case, {"actions": []})
    rewrite = evaluate_reply(
        tmp_path / "rewrite", case,
        {"actions": [{"kind": "write_file", "path": "flag.py", "content": "enabled = True\n"}]},
    )

    assert noop["noop"] is True and noop["successful_task_fix"] is False
    assert rewrite["successful_task_fix"] is True
    assert rewrite["full_file_rewrites"] == 1
