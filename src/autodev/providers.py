"""Structured local LLM providers used by role agents."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """The provider could not produce a valid structured reply."""


@dataclass(frozen=True, slots=True)
class AgentRequest:
    role: str
    prompt: str
    system_prompt: str = "You are a careful local software-development agent. Reply with JSON only."
    images: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentReply:
    data: dict[str, object]


class LLMProvider(Protocol):
    def complete(self, request: AgentRequest) -> AgentReply: ...


Transport = Callable[[str, bytes, float], bytes]


def _http_transport(url: str, payload: bytes, timeout: float) -> bytes:
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


class OllamaProvider:
    """One narrow, configurable integration point for Ollama's chat endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: float = 120.0,
        retries: int = 2,
        transport: Transport = _http_transport,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.transport = transport

    def complete(self, request: AgentRequest) -> AgentReply:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": self.temperature},
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.prompt, **({"images": list(request.images)} if request.images else {})},
                ],
            }
        ).encode("utf-8")
        error: Exception | None = None
        for _attempt in range(self.retries + 1):
            try:
                raw = self.transport(f"{self.base_url}/api/chat", payload, self.timeout)
                envelope = json.loads(raw)
                content = envelope["message"]["content"]
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise ProviderError("Ollama response content must be a JSON object")
                return AgentReply(data=data)
            except (URLError, OSError, KeyError, TypeError, json.JSONDecodeError, ProviderError) as exc:
                error = exc
        raise ProviderError(f"Ollama request failed after {self.retries + 1} attempts: {error}")


class ScriptedProvider:
    """Deterministic provider used for tests and offline demonstration runs."""

    def __init__(self, replies: dict[str, list[AgentReply]]) -> None:
        self._replies = defaultdict(deque, {role: deque(values) for role, values in replies.items()})

    def complete(self, request: AgentRequest) -> AgentReply:
        replies = self._replies[request.role]
        if not replies:
            raise ProviderError(f"no scripted reply for role {request.role}")
        return replies.popleft()


class RoleModelProvider:
    """Routes sequential role requests to profile-specific providers with a default fallback."""

    def __init__(self, default: LLMProvider, profiles: dict[str, LLMProvider] | None = None) -> None:
        self.default = default
        self.profiles = profiles or {}

    def complete(self, request: AgentRequest) -> AgentReply:
        return self.profiles.get(request.role, self.default).complete(request)
