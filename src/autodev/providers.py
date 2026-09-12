"""Structured local LLM providers used by role agents."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """The provider could not produce a valid structured reply."""


class ProviderUnavailableError(ProviderError):
    """The model endpoint could not be contacted or rejected the request."""


class ProviderResponseError(ProviderError):
    """The endpoint replied, but not with a usable structured response."""


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
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.1,
        timeout: float = 600.0,
        retries: int = 2,
        context_limit: int = 16384,
        keep_alive: str = "10m",
        transport: Transport = _http_transport,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.context_limit = context_limit
        self.keep_alive = keep_alive
        self.transport = transport

    def complete(self, request: AgentRequest) -> AgentReply:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "keep_alive": self.keep_alive,
                "options": {"temperature": self.temperature, "num_ctx": self.context_limit},
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.prompt, **({"images": list(request.images)} if request.images else {})},
                ],
            }
        ).encode("utf-8")
        transport_error: Exception | None = None
        response_error: Exception | None = None
        for _attempt in range(self.retries + 1):
            try:
                raw = self.transport(f"{self.base_url}/api/chat", payload, self.timeout)
                envelope = json.loads(raw)
                content = envelope["message"]["content"]
                if isinstance(content, str) and content.strip().startswith("```"):
                    lines = content.strip().splitlines()
                    content = "\n".join(lines[1:-1]) if len(lines) >= 3 else content
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise ProviderResponseError("Ollama response content must be a JSON object")
                return AgentReply(data=data)
            except (URLError, OSError, HTTPError) as exc:
                transport_error = exc
            except (KeyError, TypeError, json.JSONDecodeError, ProviderError) as exc:
                response_error = exc
        if transport_error is not None and response_error is None:
            raise ProviderUnavailableError(
                f"Ollama endpoint unavailable after {self.retries + 1} attempts: {transport_error}"
            )
        raise ProviderResponseError(
            f"Ollama response invalid after {self.retries + 1} attempts: {response_error or transport_error}"
        )

    def health(self) -> bool:
        try:
            with urlopen(f"{self.base_url}/api/tags", timeout=5) as response:
                return 200 <= response.status < 300
        except (URLError, OSError, HTTPError):
            return False


class ScriptedProvider:
    """Deterministic provider used for tests and offline demonstration runs."""

    def __init__(self, replies: dict[str, list[AgentReply]]) -> None:
        self._replies = defaultdict(deque, {role: deque(values) for role, values in replies.items()})

    def complete(self, request: AgentRequest) -> AgentReply:
        replies = self._replies[request.role]
        if not replies:
            raise ProviderResponseError(f"no scripted reply for role {request.role}")
        return replies.popleft()


class RoleModelProvider:
    """Routes sequential role requests to profile-specific providers with a default fallback."""

    def __init__(self, default: LLMProvider, profiles: dict[str, LLMProvider] | None = None) -> None:
        self.default = default
        self.profiles = profiles or {}

    def complete(self, request: AgentRequest) -> AgentReply:
        return self.profiles.get(request.role, self.default).complete(request)

    def health(self) -> bool:
        probe = getattr(self.default, "health", None)
        return bool(probe()) if callable(probe) else True
