"""Structured local LLM providers used by role agents."""

from __future__ import annotations

import json
import time
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
OutcomeObserver = Callable[[str, str, float], None]


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
        think: bool = False,
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
        self.think = think
        self.transport = transport
        self._outcome_observer: OutcomeObserver | None = None

    def set_outcome_observer(self, observer: OutcomeObserver | None) -> None:
        self._outcome_observer = observer

    def _observe(self, role: str, outcome: str, started: float) -> None:
        if self._outcome_observer is not None:
            self._outcome_observer(role, outcome, max(0.0, time.monotonic() - started))

    def complete(self, request: AgentRequest) -> AgentReply:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "keep_alive": self.keep_alive,
                # A bounded JSON tool decision benefits from a direct answer;
                # long hidden reasoning can otherwise keep a local 30B request
                # alive long after useful work has stopped.
                "think": self.think,
                "options": {"temperature": self.temperature, "num_ctx": self.context_limit},
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.prompt, **({"images": list(request.images)} if request.images else {})},
                ],
            }
        ).encode("utf-8")
        deadline = time.monotonic() + self.timeout
        transport_error: Exception | None = None
        response_error: Exception | None = None
        for _attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                transport_error = TimeoutError(f"Ollama request exceeded hard wall-clock timeout of {self.timeout}s")
                break
            started = time.monotonic() if self._outcome_observer is not None else 0.0
            self._observe(request.role, "ATTEMPT", started)
            try:
                raw = self.transport(f"{self.base_url}/api/chat", payload, remaining)
                envelope = json.loads(raw)
                content = envelope["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ProviderResponseError("Ollama response content is empty")
                if isinstance(content, str) and content.strip().startswith("```"):
                    lines = content.strip().splitlines()
                    content = "\n".join(lines[1:-1]) if len(lines) >= 3 else content
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise ProviderResponseError("Ollama response content must be a JSON object")
                self._observe(request.role, "SUCCESS", started)
                return AgentReply(data=data)
            except TimeoutError as exc:
                transport_error = exc
                self._observe(request.role, "TIMEOUT", started)
            except HTTPError as exc:
                transport_error = exc
                self._observe(request.role, "HTTP_ERROR", started)
            except (URLError, OSError) as exc:
                transport_error = exc
                outcome = "CONNECTION_REFUSED" if "refused" in str(exc).lower() or "10061" in str(exc) else "OTHER_ERROR"
                self._observe(request.role, outcome, started)
            except (KeyError, TypeError, json.JSONDecodeError, ProviderError) as exc:
                response_error = exc
                outcome = "EMPTY_RESPONSE" if "empty" in str(exc).lower() else "MALFORMED_STRUCTURED_OUTPUT"
                self._observe(request.role, outcome, started)
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
        self._outcome_observer: OutcomeObserver | None = None

    def set_outcome_observer(self, observer: OutcomeObserver | None) -> None:
        self._outcome_observer = observer

    def complete(self, request: AgentRequest) -> AgentReply:
        started = time.monotonic()
        if self._outcome_observer is not None:
            self._outcome_observer(request.role, "ATTEMPT", 0.0)
        replies = self._replies[request.role]
        if not replies:
            if self._outcome_observer is not None:
                self._outcome_observer(request.role, "OTHER_ERROR", time.monotonic() - started)
            raise ProviderResponseError(f"no scripted reply for role {request.role}")
        reply = replies.popleft()
        if self._outcome_observer is not None:
            self._outcome_observer(request.role, "SUCCESS", time.monotonic() - started)
        return reply


class RoleModelProvider:
    """Routes sequential role requests to profile-specific providers with a default fallback."""

    def __init__(self, default: LLMProvider, profiles: dict[str, LLMProvider] | None = None) -> None:
        self.default = default
        self.profiles = profiles or {}

    def complete(self, request: AgentRequest) -> AgentReply:
        return self.profiles.get(request.role, self.default).complete(request)

    def set_outcome_observer(self, observer: OutcomeObserver | None) -> None:
        seen: set[int] = set()
        for provider in (self.default, *self.profiles.values()):
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            setter = getattr(provider, "set_outcome_observer", None)
            if callable(setter):
                setter(observer)

    def health(self) -> bool:
        probe = getattr(self.default, "health", None)
        return bool(probe()) if callable(probe) else True
