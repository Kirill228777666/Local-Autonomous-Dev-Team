import json
from urllib.error import URLError

import pytest

from autodev.providers import (
    AgentReply,
    AgentRequest,
    OllamaProvider,
    ProviderError,
    ScriptedProvider,
)


def test_scripted_provider_returns_role_specific_replies_in_order() -> None:
    provider = ScriptedProvider(
        {
            "MANAGER": [AgentReply({"tasks": [{"title": "Plan", "description": "Plan it"}]})],
            "CODER": [AgentReply({"actions": []})],
        }
    )

    manager_reply = provider.complete(AgentRequest(role="MANAGER", prompt="spec"))
    coder_reply = provider.complete(AgentRequest(role="CODER", prompt="task"))

    assert manager_reply.data["tasks"][0]["title"] == "Plan"
    assert coder_reply.data == {"actions": []}


def test_scripted_provider_raises_when_a_role_has_no_reply() -> None:
    with pytest.raises(ProviderError, match="no scripted reply"):
        ScriptedProvider({}).complete(AgentRequest(role="REVIEWER", prompt="review"))


def test_ollama_provider_sends_low_temperature_and_parses_json_response() -> None:
    sent_payloads: list[dict[str, object]] = []

    def transport(url: str, payload: bytes, timeout: float) -> bytes:
        assert url == "http://localhost:11434/api/chat"
        assert timeout == 30
        sent_payloads.append(json.loads(payload))
        return json.dumps({"message": {"content": '{"approved": true}'}}).encode()

    reply = OllamaProvider(model="qwen3:14b", timeout=30, transport=transport).complete(
        AgentRequest(role="REVIEWER", prompt="review", system_prompt="strict")
    )

    assert reply.data == {"approved": True}
    assert sent_payloads[0]["model"] == "qwen3:14b"
    assert sent_payloads[0]["options"] == {"temperature": 0.1}
    assert sent_payloads[0]["format"] == "json"


def test_ollama_provider_retries_transient_transport_error() -> None:
    attempts = 0

    def transport(_url: str, _payload: bytes, _timeout: float) -> bytes:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("Ollama unavailable")
        return b'{"message": {"content": "{\\\"actions\\\": []}"}}'

    reply = OllamaProvider(model="qwen", transport=transport, retries=1).complete(
        AgentRequest(role="CODER", prompt="implement")
    )

    assert attempts == 2
    assert reply.data == {"actions": []}
