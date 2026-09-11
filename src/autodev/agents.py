"""Role-specific prompt boundaries for the sequential autonomous team."""

from __future__ import annotations

from .context import ContextBuilder
from .models import ProjectState, Task
from .providers import AgentReply, AgentRequest, LLMProvider, ProviderError


ROLE_PROMPTS = {
    "MANAGER": "You are Manager. Keep scope faithful to the original specification. Reply in JSON only.",
    "ARCHITECT": "You are Architect. Recommend the smallest practical alternative after repeated failure. Reply in JSON only.",
    "CODER": "You are Coder. Return only a JSON object with an actions array. Use only requested task scope. Prefer available standard-library tools; never assume dependencies or executables exist. Do not start a persistent server as verification.",
    "TESTER": "You are Tester. Return only JSON with a command array that independently verifies the task.",
    "REVIEWER": "You are Reviewer. Return only JSON: approved boolean and optional reasons array. Do not add scope.",
    "FINAL_QA": "You are Final QA. Independently compare the completed product against the original specification. Return only JSON with status PASS or FAIL and a findings array. Do not add scope.",
}


class RoleAgents:
    def __init__(self, provider: LLMProvider, context: ContextBuilder | None = None, structured_retries: int = 1) -> None:
        self.provider = provider
        self.context = context or ContextBuilder()
        self.structured_retries = structured_retries

    def plan(self, state: ProjectState) -> AgentReply:
        return self._ask(
            "MANAGER",
            f"Create the minimal major task list for this project. Return {{\"tasks\":[{{\"title\":str,\"description\":str}}]}}.\n\n{state.original_spec}",
            self._valid_plan,
        )

    def select(self, state: ProjectState) -> AgentReply:
        pending = [{"id": task.id, "title": task.title} for task in state.tasks if task.status.value == "PENDING"]
        return self._ask(
            "MANAGER",
            f"Choose the next pending task without adding scope. Return {{\"next_task_id\": string|null}}.\nPending: {pending}",
            self._valid_selection,
        )

    def code(self, state: ProjectState, task: Task) -> AgentReply:
        schema = (
            "Return {\"actions\":[...]}. Each action must be exactly one of: "
            "{\"kind\":\"write_file\",\"path\":\"relative/path\",\"content\":\"text\"}; "
            "{\"kind\":\"edit_file\",\"path\":\"relative/path\",\"old\":\"exact text\",\"new\":\"replacement\"}; "
            "{\"kind\":\"delete_file\",\"path\":\"relative/path\"}; "
            "{\"kind\":\"append_file\",\"path\":\"relative/path\",\"content\":\"text\"}; "
            "{\"kind\":\"run_command\",\"command\":[\"program\",\"arg\"]}. "
            "Paths must be relative to the workspace; do not use shell wrappers.\n\n"
        )
        return self._ask("CODER", schema + self.context.for_task(state, task, []), self._valid_actions)

    def test(self, state: ProjectState, task: Task) -> AgentReply:
        instruction = (
            "Return {\"command\":[\"program\",\"arg\"]} with one independent, non-shell verification command. "
            "Do not merely report success.\n\n"
        )
        return self._ask("TESTER", instruction + self.context.for_task(state, task, []), self._valid_command)

    def review(self, state: ProjectState, task: Task, evidence: str) -> AgentReply:
        return self._ask("REVIEWER", f"{self.context.for_task(state, task, [])}\n\nEvidence:\n{evidence}", self._valid_review)

    def diagnose(self, state: ProjectState, task: Task) -> AgentReply:
        return self._ask("ARCHITECT", "Return {\"diagnosis\": string}.\n\n" + self.context.for_task(state, task, []), self._valid_diagnosis)

    def final_qa(self, state: ProjectState, evidence: str) -> AgentReply:
        completed = [task.title for task in state.tasks if task.status.value == "DONE"]
        blocked = [task.title for task in state.tasks if task.status.value == "BLOCKED"]
        prompt = (
            f"Original specification:\n{state.original_spec}\n\nCompleted tasks: {completed}\n"
            f"Blocked tasks: {blocked}\nDecisions: {state.decisions[-8:]}\n\n"
            f"Independent evidence:\n{evidence}\n\n"
            "Return {\"status\":\"PASS\"|\"FAIL\",\"findings\":[{\"title\":str,\"description\":str}]}."
        )
        return self._ask("FINAL_QA", prompt, self._valid_final_qa)

    def _ask(self, role: str, prompt: str, validator: callable) -> AgentReply:
        error = ""
        for attempt in range(self.structured_retries + 1):
            reply = self.provider.complete(
                AgentRequest(
                    role=role,
                    prompt=prompt if not error else prompt + f"\n\nYour previous JSON was invalid: {error}. Return only the required schema.",
                    system_prompt=ROLE_PROMPTS[role],
                )
            )
            try:
                validator(reply.data)
                return reply
            except ValueError as exc:
                error = str(exc)
        raise ProviderError(f"{role} returned invalid structured output after {self.structured_retries + 1} attempt(s): {error}")

    @staticmethod
    def _valid_plan(data: dict[str, object]) -> None:
        tasks = data.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("tasks must be a non-empty array")
        for task in tasks:
            if not isinstance(task, dict) or not isinstance(task.get("title"), str) or not task["title"].strip() or not isinstance(task.get("description"), str) or not task["description"].strip():
                raise ValueError("every task needs non-empty title and description")

    @staticmethod
    def _valid_selection(data: dict[str, object]) -> None:
        selected = data.get("next_task_id")
        if selected is not None and not isinstance(selected, str):
            raise ValueError("next_task_id must be string or null")

    @staticmethod
    def _valid_actions(data: dict[str, object]) -> None:
        actions = data.get("actions")
        if not isinstance(actions, list):
            raise ValueError("actions must be an array")
        required = {"write_file": ("path", "content"), "append_file": ("path", "content"), "edit_file": ("path", "old", "new"), "delete_file": ("path",), "run_command": ("command",)}
        for action in actions:
            if not isinstance(action, dict) or action.get("kind") not in required:
                raise ValueError("action kind is invalid")
            for field in required[action["kind"]]:  # type: ignore[index]
                value = action.get(field)
                if field == "command":
                    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
                        raise ValueError("command must be a non-empty string array")
                elif not isinstance(value, str) or not value:
                    raise ValueError(f"action field {field} must be non-empty text")

    @staticmethod
    def _valid_command(data: dict[str, object]) -> None:
        command = data.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise ValueError("command must be a non-empty string array")

    @staticmethod
    def _valid_review(data: dict[str, object]) -> None:
        if not isinstance(data.get("approved"), bool):
            raise ValueError("approved must be boolean")
        reasons = data.get("reasons", [])
        if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
            raise ValueError("reasons must be a string array")

    @staticmethod
    def _valid_diagnosis(data: dict[str, object]) -> None:
        if not isinstance(data.get("diagnosis"), str) or not data["diagnosis"].strip():
            raise ValueError("diagnosis must be non-empty text")

    @staticmethod
    def _valid_final_qa(data: dict[str, object]) -> None:
        if data.get("status") not in {"PASS", "FAIL"} or not isinstance(data.get("findings"), list):
            raise ValueError("Final QA requires PASS/FAIL and findings array")
        for finding in data["findings"]:  # type: ignore[index]
            if not isinstance(finding, dict) or not isinstance(finding.get("title"), str) or not finding["title"].strip() or not isinstance(finding.get("description"), str) or not finding["description"].strip():
                raise ValueError("Final QA finding requires non-empty title and description")
