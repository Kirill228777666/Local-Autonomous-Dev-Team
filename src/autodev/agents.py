"""Role-specific prompt boundaries for the sequential autonomous team."""

from __future__ import annotations

from .context import ContextBuilder
from .models import ProjectState, Task
from .providers import AgentReply, AgentRequest, LLMProvider, ProviderError


ROLE_PROMPTS = {
    "MANAGER": "You are Manager. Keep scope faithful to the original specification. Reply in JSON only.",
    "ARCHITECT": "You are Architect. Recommend the smallest practical alternative after repeated failure. Reply in JSON only.",
    "CODER": "You are Coder. Return only a JSON object with an actions array. Use only requested task scope. Prefer available standard-library tools; never assume dependencies or executables exist. Do not start a persistent server as verification. For a web app, read PORT for the listen port and disable debug/reloader in automated launch. For mutable-data tests, honor AUTODEV_TEST_DATABASE and AUTODEV_TESTING so every validator has isolated schema/data.",
    "TESTER": "You are Tester. Return only JSON with a command array that independently verifies the task.",
    "REVIEWER": "You are Reviewer. Return only JSON: approved boolean and optional reasons array. Do not add scope.",
    "FINAL_QA": "You are Final QA. Independently compare the completed product against the original specification. Return only JSON with status PASS or FAIL and a findings array. Do not add scope.",
}


class RoleAgents:
    def __init__(self, provider: LLMProvider, context: ContextBuilder | None = None, structured_retries: int = 1) -> None:
        self.provider = provider
        self.context = context or ContextBuilder()
        self.structured_retries = structured_retries
        self.last_structured_role = ""
        self.last_structured_invalid_count = 0
        self.last_structured_repaired = False

    def plan(self, state: ProjectState) -> AgentReply:
        return self._ask(
            "MANAGER",
            "Create the minimal major task list for this project. Return "
            "{\"tasks\":[{\"title\":str,\"description\":str,\"capability_id\":str,\"intent\":str,\"acceptance_criteria\":[str],\"depends_on\":[capability_id]}]}. "
            "capability_id is stable lowercase dotted identity; never duplicate an existing capability. "
            f"Authoritative project contract: {state.project_contract or state.architecture}\n\n{state.original_spec}",
            self._valid_plan,
        )

    def select(self, state: ProjectState) -> AgentReply:
        pending = [{"id": task.id, "title": task.title} for task in state.tasks if task.status.value == "PENDING"]
        return self._ask(
            "MANAGER",
            f"Choose the next pending task without adding scope. Return {{\"next_task_id\": string|null}}.\nPending: {pending}",
            self._valid_selection,
        )

    def code(self, state: ProjectState, task: Task, relevant_files: list[str] | None = None) -> AgentReply:
        schema = (
            "Return {\"actions\":[...],\"task_status\":\"continue\"|\"ready_for_validation\"}. Each action must be exactly one of: "
            "{\"kind\":\"mutate_file\",\"path\":\"relative/path\",\"intent\":\"small concrete mutation goal\"}; "
            "{\"kind\":\"delete_file\",\"path\":\"relative/path\"}; "
            "{\"kind\":\"read_file\",\"path\":\"relative/path\"}; "
            "{\"kind\":\"run_command\",\"command\":[\"program\",\"arg\"]}; "
            "{\"kind\":\"start_process\",\"command\":[\"program\",\"arg\"]}. "
            "For every source change use mutate_file: NEVER place source text, a full file, a patch, old text, or new text in this decision response. "
            "The controller will request one bounded patch for each mutation. Return at most 4 actions. "
            "Use task_status=continue after a bounded batch; the controller will persist it and request the next batch in the same attempt. "
            "For a valid no-op return exactly {\"actions\":[],\"task_status\":\"ready_for_validation\"}; never emit an action with empty content. "
            "Paths must be relative to the workspace; do not use shell wrappers.\n\n"
        )
        return self._ask("CODER", schema + self.context.for_task(state, task, relevant_files or []), self._valid_actions)

    def patch(
        self,
        state: ProjectState,
        task: Task,
        *,
        path: str,
        intent: str,
        current: str,
        existing: bool,
        continuation: int,
    ) -> AgentReply:
        """Ask for one independently bounded mutation, never a task-sized body."""
        mode = "existing" if existing else "new"
        prompt = (
            "Return exactly one JSON patch object for this one file. No prose. "
            "Schema: {\"operation\":\"create\"|\"append\"|\"replace\",\"content\":str,\"old\":str optional for replace,\"done\":bool}. "
            "content and old together must contain at most 6000 characters. "
            "For an existing file use replace with exact old text from the supplied context, or append. "
            "For a new file use create first; if more content is needed, set done=false and return only the next section. "
            "Never return a full unrelated file or repeat earlier sections.\n\n"
            f"Path: {path}\nFile state: {mode}\nMutation goal: {intent}\nContinuation: {continuation}\n"
            f"CURRENT RELEVANT CONTENT:\n{current}"
        )
        return self._ask("CODER", prompt, self._valid_patch)

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
        self.last_structured_role = role
        self.last_structured_invalid_count = 0
        self.last_structured_repaired = False
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
                self.last_structured_repaired = self.last_structured_invalid_count > 0
                return reply
            except ValueError as exc:
                error = str(exc)
                self.last_structured_invalid_count += 1
        raise ProviderError(f"{role} returned invalid structured output after {self.structured_retries + 1} attempt(s): {error}")

    @staticmethod
    def _valid_plan(data: dict[str, object]) -> None:
        tasks = data.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("tasks must be a non-empty array")
        for task in tasks:
            if not isinstance(task, dict) or not isinstance(task.get("title"), str) or not task["title"].strip() or not isinstance(task.get("description"), str) or not task["description"].strip():
                raise ValueError("every task needs non-empty title and description")
            dependencies = task.get("depends_on", [])
            if not isinstance(dependencies, list) or not all(isinstance(item, str) and item.strip() for item in dependencies):
                raise ValueError("task depends_on must be a string array")

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
        status = data.get("task_status", "ready_for_validation")
        if status not in {"continue", "ready_for_validation"}:
            raise ValueError("task_status must be continue or ready_for_validation")
        if len(actions) > 4:
            raise ValueError("action batch exceeds maximum action count")
        required = {"mutate_file": ("path", "intent"), "write_file": ("path", "content"), "append_file": ("path", "content"), "edit_file": ("path", "old", "new"), "delete_file": ("path",), "read_file": ("path",), "run_command": ("command",), "start_process": ("command",)}
        textual_payload = 0
        for action in actions:
            # Some local models label the discriminator ``action``.  This is
            # a lossless representation change, not an inferred tool call.
            if isinstance(action, dict) and "kind" not in action and isinstance(action.get("action"), str):
                action["kind"] = action["action"]
            if not isinstance(action, dict) or action.get("kind") not in required:
                raise ValueError("action kind is invalid")
            for field in required[action["kind"]]:  # type: ignore[index]
                value = action.get(field)
                if field == "command":
                    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
                        raise ValueError("command must be a non-empty string array")
                elif not isinstance(value, str) or not value:
                    raise ValueError(f"action field {field} must be non-empty text")
                if isinstance(value, str) and field in {"content", "old", "new"}:
                    textual_payload += len(value)
        if textual_payload > 12_000:
            raise ValueError("action batch textual payload exceeds maximum")

    @staticmethod
    def _valid_patch(data: dict[str, object]) -> None:
        operation = data.get("operation")
        if operation not in {"create", "append", "replace"}:
            raise ValueError("patch operation is invalid")
        content = data.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError("patch content must be non-empty text")
        old = data.get("old", "")
        if operation == "replace" and (not isinstance(old, str) or not old):
            raise ValueError("replace patch requires non-empty old text")
        if operation != "replace" and old not in {"", None}:
            raise ValueError("only replace patch may supply old text")
        if not isinstance(data.get("done"), bool):
            raise ValueError("patch done must be boolean")
        if len(content) + (len(old) if isinstance(old, str) else 0) > 6_000:
            raise ValueError("patch textual payload exceeds maximum")

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
