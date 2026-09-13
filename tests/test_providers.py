import json
from urllib.error import HTTPError, URLError

import pytest
import autodev.providers as providers_module

from autodev.providers import (
    AgentReply,
    AgentRequest,
    OllamaProvider,
    ProviderError,
    ProviderUnavailableError,
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
        assert url == "http://127.0.0.1:11434/api/chat"
        assert 29.0 < timeout <= 30
        sent_payloads.append(json.loads(payload))
        return json.dumps({"message": {"content": '{"approved": true}'}}).encode()

    reply = OllamaProvider(model="qwen3:14b", timeout=30, transport=transport).complete(
        AgentRequest(role="REVIEWER", prompt="review", system_prompt="strict")
    )

    assert reply.data == {"approved": True}
    assert sent_payloads[0]["model"] == "qwen3:14b"
    assert sent_payloads[0]["options"] == {"temperature": 0.1, "num_ctx": 16384}
    assert sent_payloads[0]["keep_alive"] == "10m"
    assert sent_payloads[0]["format"] == "json"
    assert sent_payloads[0]["think"] is False


def test_ollama_provider_can_explicitly_enable_model_thinking() -> None:
    payloads: list[dict[str, object]] = []

    def transport(_url: str, payload: bytes, _timeout: float) -> bytes:
        payloads.append(json.loads(payload))
        return b'{"message":{"content":"{\\"actions\\":[]}"}}'

    OllamaProvider(model="qwen3.6:27b", think=True, transport=transport).complete(
        AgentRequest(role="CODER", prompt="x")
    )

    assert payloads[0]["think"] is True


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


def test_ollama_provider_accepts_json_wrapped_in_markdown_fence() -> None:
    reply = OllamaProvider(model="qwen", transport=lambda *_: b'{"message":{"content":"```json\\n{\\\"actions\\\": []}\\n```"}}').complete(
        AgentRequest(role="CODER", prompt="implement")
    )

    assert reply.data == {"actions": []}


def test_ollama_http_503_is_an_endpoint_outage_after_bounded_retry() -> None:
    attempts = 0

    def transport(_url: str, _payload: bytes, _timeout: float) -> bytes:
        nonlocal attempts
        attempts += 1
        raise HTTPError("http://127.0.0.1:11434/api/chat", 503, "busy", {}, None)

    with pytest.raises(ProviderUnavailableError, match="endpoint unavailable"):
        OllamaProvider(model="qwen", transport=transport, retries=1).complete(
            AgentRequest(role="CODER", prompt="implement")
        )
    assert attempts == 2


def test_retries_share_one_hard_wall_clock_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_timeouts: list[float] = []
    clock = iter((100.0, 100.0, 104.0))
    monkeypatch.setattr(providers_module.time, "monotonic", lambda: next(clock))

    def transport(_url: str, _payload: bytes, timeout: float) -> bytes:
        observed_timeouts.append(timeout)
        if len(observed_timeouts) == 1:
            raise URLError("temporary")
        return b'{"message":{"content":"{\\"actions\\":[]}"}}'

    reply = OllamaProvider(model="qwen", timeout=5, retries=1, transport=transport).complete(
        AgentRequest(role="CODER", prompt="implement")
    )

    assert reply.data == {"actions": []}
    assert observed_timeouts == [5.0, 1.0]


def test_provider_reports_one_terminal_outcome_for_each_http_attempt() -> None:
    outcomes: list[tuple[str, str]] = []
    attempts = 0

    def transport(_url: str, _payload: bytes, _timeout: float) -> bytes:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("connection refused")
        return b'{"message":{"content":"{\\"actions\\":[]}"}}'

    provider = OllamaProvider(model="qwen", retries=1, transport=transport)
    provider.set_outcome_observer(lambda role, outcome, _latency: outcomes.append((role, outcome)))

    provider.complete(AgentRequest(role="CODER", prompt="implement"))

    assert outcomes == [
        ("CODER", "ATTEMPT"),
        ("CODER", "CONNECTION_REFUSED"),
        ("CODER", "ATTEMPT"),
        ("CODER", "SUCCESS"),
    ]


def test_provider_distinguishes_timeout_and_malformed_response_outcomes() -> None:
    timeout_events: list[str] = []
    timeout = OllamaProvider(model="qwen", retries=0, transport=lambda *_: (_ for _ in ()).throw(TimeoutError("late")))
    timeout.set_outcome_observer(lambda _role, outcome, _latency: timeout_events.append(outcome))

    with pytest.raises(ProviderUnavailableError):
        timeout.complete(AgentRequest(role="CODER", prompt="x"))

    malformed_events: list[str] = []
    malformed = OllamaProvider(model="qwen", retries=0, transport=lambda *_: b'{"message":{"content":"not json"}}')
    malformed.set_outcome_observer(lambda _role, outcome, _latency: malformed_events.append(outcome))

    with pytest.raises(ProviderError):
        malformed.complete(AgentRequest(role="TESTER", prompt="x"))

    assert timeout_events == ["ATTEMPT", "TIMEOUT"]
    assert malformed_events == ["ATTEMPT", "MALFORMED_STRUCTURED_OUTPUT"]
