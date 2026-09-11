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


def test_coder_prompt_documents_the_only_supported_action_schema() -> None:
    provider = RecordingProvider()
    state = ProjectState.create("Create a text file")
    task = Task.create("Write file", "Create hello.txt")

    RoleAgents(provider).code(state, task)

    prompt = provider.requests[0].prompt
    assert "write_file" in prompt
    assert "edit_file" in prompt
    assert "run_command" in prompt


def test_tester_prompt_requires_an_independent_command_array() -> None:
    provider = RecordingProvider()
    state = ProjectState.create("Test a text file")
    task = Task.create("Test file", "Verify hello.txt")

    RoleAgents(provider).test(state, task)

    assert '"command"' in provider.requests[0].prompt
