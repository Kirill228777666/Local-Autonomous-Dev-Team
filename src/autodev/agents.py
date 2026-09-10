"""Role-specific prompt boundaries for the sequential autonomous team."""

from __future__ import annotations

from .context import ContextBuilder
from .models import ProjectState, Task
from .providers import AgentReply, AgentRequest, LLMProvider


ROLE_PROMPTS = {
    "MANAGER": "You are Manager. Keep scope faithful to the original specification. Reply in JSON only.",
    "ARCHITECT": "You are Architect. Recommend the smallest practical alternative after repeated failure. Reply in JSON only.",
    "CODER": "You are Coder. Return only a JSON object with an actions array. Use only requested task scope.",
    "TESTER": "You are Tester. Return only JSON with a command array that independently verifies the task.",
    "REVIEWER": "You are Reviewer. Return only JSON: approved boolean and optional reasons array. Do not add scope.",
}


class RoleAgents:
    def __init__(self, provider: LLMProvider, context: ContextBuilder | None = None) -> None:
        self.provider = provider
        self.context = context or ContextBuilder()

    def plan(self, state: ProjectState) -> AgentReply:
        return self._ask(
            "MANAGER",
            f"Create the minimal major task list for this project. Return {{\"tasks\":[{{\"title\":str,\"description\":str}}]}}.\n\n{state.original_spec}",
        )

    def select(self, state: ProjectState) -> AgentReply:
        pending = [{"id": task.id, "title": task.title} for task in state.tasks if task.status.value == "PENDING"]
        return self._ask(
            "MANAGER",
            f"Choose the next pending task without adding scope. Return {{\"next_task_id\": string|null}}.\nPending: {pending}",
        )

    def code(self, state: ProjectState, task: Task) -> AgentReply:
        schema = (
            "Return {\"actions\":[...]}. Each action must be exactly one of: "
            "{\"kind\":\"write_file\",\"path\":\"relative/path\",\"content\":\"text\"}; "
            "{\"kind\":\"edit_file\",\"path\":\"relative/path\",\"old\":\"exact text\",\"new\":\"replacement\"}; "
            "{\"kind\":\"delete_file\",\"path\":\"relative/path\"}; "
            "{\"kind\":\"run_command\",\"command\":[\"program\",\"arg\"]}. "
            "Paths must be relative to the workspace; do not use shell wrappers.\n\n"
        )
        return self._ask("CODER", schema + self.context.for_task(state, task, []))

    def test(self, state: ProjectState, task: Task) -> AgentReply:
        instruction = (
            "Return {\"command\":[\"program\",\"arg\"]} with one independent, non-shell verification command. "
            "Do not merely report success.\n\n"
        )
        return self._ask("TESTER", instruction + self.context.for_task(state, task, []))

    def review(self, state: ProjectState, task: Task, evidence: str) -> AgentReply:
        return self._ask("REVIEWER", f"{self.context.for_task(state, task, [])}\n\nEvidence:\n{evidence}")

    def diagnose(self, state: ProjectState, task: Task) -> AgentReply:
        return self._ask("ARCHITECT", self.context.for_task(state, task, []))

    def _ask(self, role: str, prompt: str) -> AgentReply:
        return self.provider.complete(AgentRequest(role=role, prompt=prompt, system_prompt=ROLE_PROMPTS[role]))
