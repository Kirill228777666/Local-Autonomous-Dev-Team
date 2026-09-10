from autodev.providers import AgentReply, AgentRequest, RoleModelProvider, ScriptedProvider


def test_role_model_provider_uses_role_specific_provider_then_default() -> None:
    default = ScriptedProvider({"MANAGER": [AgentReply({"source": "default"})]})
    coder = ScriptedProvider({"CODER": [AgentReply({"source": "coder"})]})
    provider = RoleModelProvider(default, {"CODER": coder})

    assert provider.complete(AgentRequest(role="CODER", prompt="x")).data == {"source": "coder"}
    assert provider.complete(AgentRequest(role="MANAGER", prompt="x")).data == {"source": "default"}
