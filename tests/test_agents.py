from autodev.agents import RoleAgents
from autodev.models import ProjectState, Task
from autodev.providers import AgentReply, AgentRequest


class RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    def complete(self, request: AgentRequest) -> AgentReply:
        self.requests.append(request)
        if request.role == "TESTER":
            return AgentReply({"command": ["py", "-3", "-c", "print('verified')"]})
        return AgentReply({"actions": []})


def test_coder_prompt_uses_small_mutation_decisions_instead_of_source_bodies() -> None:
    provider = RecordingProvider()
    state = ProjectState.create("Create a text file")
    task = Task.create("Write file", "Create hello.txt")

    RoleAgents(provider).code(state, task)

    prompt = provider.requests[0].prompt
    assert "mutate_file" in prompt
    assert "NEVER place source text" in prompt
    assert "run_command" in prompt
    assert "start_process" in prompt


def test_tester_prompt_requires_an_independent_command_array() -> None:
    provider = RecordingProvider()
    state = ProjectState.create("Test a text file")
    task = Task.create("Test file", "Verify hello.txt")

    RoleAgents(provider).test(state, task)

    assert '"command"' in provider.requests[0].prompt


def test_large_legacy_source_action_is_reduced_to_a_small_mutation_decision() -> None:
    class LargeReplyProvider:
        def complete(self, _request: AgentRequest) -> AgentReply:
            return AgentReply({"actions": [{"kind": "write_file", "path": "huge.py", "content": "x" * 55_000}]})

    state = ProjectState.create("Build a large application")
    task = Task.create("Implement source", "Implement the application")

    reply = RoleAgents(LargeReplyProvider()).code(state, task)

    assert reply.data["actions"] == [{
        "kind": "mutate_file",
        "path": "huge.py",
        "intent": "Materialize the requested source change through bounded patches.",
    }]
