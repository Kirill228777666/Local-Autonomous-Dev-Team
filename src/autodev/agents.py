"""Role-specific prompt boundaries for the sequential autonomous team."""

from __future__ import annotations

import re
from uuid import uuid4

from .context import ContextBuilder
from .models import ProjectState, Task
from .providers import AgentReply, AgentRequest, LLMProvider, ProviderError, ProviderResponseError


ROLE_PROMPTS = {
    "MANAGER": "You are Manager. Keep scope faithful to the original specification. Reply in JSON only.",
    "ARCHITECT": "You are Architect. Recommend the smallest practical alternative after repeated failure. Reply in JSON only.",
    "CODER": "You are Coder. Return only a JSON object with an actions array. Use only requested task scope. Prefer available standard-library tools; never assume dependencies or executables exist. Do not start a persistent server as verification. For a web app, read PORT for the listen port and disable debug/reloader in automated launch. For mutable-data tests, honor AUTODEV_TEST_DATABASE and AUTODEV_TESTING so every validator has isolated schema/data.",
    "TESTER": "You are Tester. Return only JSON with a command array that independently verifies the task.",
    "REVIEWER": "You are Reviewer. Return only JSON: approved boolean and optional reasons array. Do not add scope.",
    "FINAL_QA": "You are Final QA. Independently compare the completed product against the original specification. Return only JSON with status PASS or FAIL and a findings array. Do not add scope.",
}


class MutationFrameError(ProviderError):
    """A bounded mutation envelope is ambiguous or malformed and was not applied."""

    def __init__(self, classification: str, message: str) -> None:
        self.classification = classification
        super().__init__(f"MUTATION_PROTOCOL_{classification}: {message}")


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

    def code(
        self,
        state: ProjectState,
        task: Task,
        relevant_files: list[str] | None = None,
        *,
        instruction: str = "",
    ) -> AgentReply:
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
        return self._ask(
            "CODER",
            schema + (instruction + "\n\n" if instruction else "") + self.context.for_task(state, task, relevant_files or []),
            self._valid_actions,
            max_output_tokens=2048,
            max_output_characters=12_000,
        )

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
        request_id: str | None = None,
        max_patch_characters: int = 6_000,
        recovery_reason: str = "",
        representation: str = "framed",
    ) -> AgentReply:
        """Ask for one independently bounded mutation, never a task-sized body."""
        mode = "existing" if existing else "new"
        token_budget = 1_536 if max_patch_characters > 3_000 else 768 if max_patch_characters > 1_500 else 384
        response_budget = 16_000
        marker = request_id or uuid4().hex[:16]
        content_start = f"CONTENT-BEGIN-{marker}"
        content_end = f"CONTENT-END-{marker}"
        old_start = f"OLD-BEGIN-{marker}"
        old_end = f"OLD-END-{marker}"
        new_start = f"NEW-BEGIN-{marker}"
        new_end = f"NEW-END-{marker}"
        common = (
            f"Mutation request id: {marker}\nPath: {path}\nFile state: {mode}\nMutation goal: {intent}\nContinuation: {continuation}\n"
            + (f"Recovery: {recovery_reason}. Do not repeat the same framing; use the requested bounded representation.\n" if recovery_reason else "")
            + f"The combined old and new source text must be at most {max_patch_characters} characters.\nCURRENT RELEVANT CONTENT:\n{current}"
        )
        if representation == "json":
            prompt = (
                "The previous text frame was ambiguous. Return one bounded JSON mutation object and nothing else. "
                "This is a single-file patch, not the high-level action batch. Required exact keys: request_id, path, operation, old, content, done. "
                "request_id and path must exactly match the values below. operation is create, append, or replace; old is empty except for replace. "
                "Do not include markdown or prose. Source text belongs only in content and must not exceed the stated combined limit.\n"
                + common
            )
            raw_response = False
            system_prompt = "Return exactly one JSON object for the bounded file mutation. No prose or markdown."
        elif representation == "framed":
            prompt = (
                "Return one bounded file patch using the exact plain-text framing below. Do not return JSON, markdown fences, or prose. "
                f"The source body must be at most {max_patch_characters} characters.\n"
                "First line: OPERATION=create, append, or replace. Second line: DONE=true or DONE=false. Third line must be PATH with the exact target path.\n"
                f"For create/append, put only the source body between {content_start} and {content_end}, each marker on its own line.\n"
                f"For replace, put exact old text between {old_start} and {old_end}, then replacement text between {new_start} and {new_end}.\n"
                "For an existing file prefer append or a small exact replacement. For a new file use create for the first section, then append. "
                "If continuing, return only the next atomic section, not the whole file. Preserve source literally; no JSON escaping is needed.\n"
                f"OPERATION=...\nDONE=...\nPATH={path}\n"
                f"Mutation request id: {marker}\nPath: {path}\nFile state: {mode}\nMutation goal: {intent}\nContinuation: {continuation}\n"
                + (f"Recovery: {recovery_reason}. Make this patch smaller than the failed response.\n" if recovery_reason else "")
                + f"CURRENT RELEVANT CONTENT:\n{current}"
            )
            raw_response = True
            system_prompt = "You are Coder. Return exactly one bounded plain-text patch using the requested framing. Source is raw text, not JSON."
        else:
            raise ValueError(f"unknown mutation representation: {representation}")
        try:
            reply = self.provider.complete(
                AgentRequest(
                    role="CODER",
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_output_tokens=token_budget,
                    max_output_characters=response_budget,
                    raw_response=raw_response,
                )
            )
        except ProviderResponseError as error:
            if "exceeded character budget" in str(error).lower():
                raise MutationFrameError("AMBIGUOUS", "mutation response exceeded the bounded payload limit") from error
            raise
        if not raw_response:
            return AgentReply(self._parse_json_mutation(reply.data, marker, path, max_patch_characters))
        raw = reply.data.get("raw_text")
        if not isinstance(raw, str):
            raise MutationFrameError("MALFORMED", "mutation response did not contain raw patch text")
        return AgentReply(self._parse_raw_patch(
            raw,
            marker,
            max_patch_characters,
            expected_path=path,
            response_complete=reply.data.get("response_complete") is True,
        ))

    @staticmethod
    def _parse_raw_patch(
        raw: str,
        marker: str,
        max_patch_characters: int,
        *,
        expected_path: str,
        response_complete: bool = False,
    ) -> dict[str, object]:
        """Parse the mutation wire format as an explicit, fail-closed state machine.

        NEW-BEGIN is an unambiguous structural boundary for OLD when OLD-END is
        omitted. A missing final terminator is not recoverable: with arbitrary
        source text there is no deterministic way to distinguish payload from
        trailing prose, even if the transport reports normal completion.
        """
        lines = raw.splitlines(keepends=True)
        header = [line.rstrip("\r\n") for line in lines[:3]]
        if len(header) != 3 or not header[0].startswith("OPERATION=") or not header[1].startswith("DONE=") or not header[2].startswith("PATH="):
            raise MutationFrameError("MALFORMED", "mutation response header is invalid")
        operation = header[0].partition("=")[2].strip()
        done_text = header[1].partition("=")[2].strip().lower()
        path = header[2].partition("=")[2]
        if operation not in {"create", "append", "replace"} or done_text not in {"true", "false"}:
            raise MutationFrameError("MALFORMED", "mutation response operation/status is invalid")
        if path != expected_path:
            raise MutationFrameError("AMBIGUOUS", "mutation response path does not match the requested target")

        body_lines = [line.rstrip("\r\n") for line in lines[3:]]
        marker_re = re.compile(r"^(?:OLD|NEW|CONTENT)-(?:BEGIN|END)-([A-Za-z0-9]+)$")
        discovered = [(index + 3, text, marker_re.fullmatch(text)) for index, text in enumerate(body_lines)]
        allowed_markers = (
            {"OLD-BEGIN", "OLD-END", "NEW-BEGIN", "NEW-END"}
            if operation == "replace"
            else {"CONTENT-BEGIN", "CONTENT-END"}
        )
        for _index, text, match in discovered:
            if text.startswith(("OLD-", "NEW-", "CONTENT-")) and (match is None or match.group(1) != marker):
                raise MutationFrameError("AMBIGUOUS", "mutation response contains a foreign or malformed boundary marker")
            if match is not None and text.rsplit("-", 1)[0] not in allowed_markers:
                raise MutationFrameError("AMBIGUOUS", "mutation response contains an unexpected boundary for this operation")

        def positions(name: str) -> list[int]:
            wanted = f"{name}-{marker}"
            return [index for index, text, _match in discovered if text == wanted]

        def exactly_one(name: str) -> int:
            found = positions(name)
            if len(found) != 1:
                raise MutationFrameError("AMBIGUOUS", f"expected exactly one {name}-{marker} boundary")
            return found[0]

        def optional_one(name: str) -> int | None:
            found = positions(name)
            if len(found) > 1:
                raise MutationFrameError("AMBIGUOUS", f"duplicate {name}-{marker} boundary")
            return found[0] if found else None

        def section(start: int, end: int) -> str:
            if end <= start:
                raise MutationFrameError("AMBIGUOUS", "mutation boundary order is invalid")
            value = "".join(lines[start + 1 : end])
            # The newline immediately before a marker belongs to the framing,
            # not to the source body. Any additional newline remains verbatim.
            if value.endswith("\r\n"):
                value = value[:-2]
            elif value.endswith(("\n", "\r")):
                value = value[:-1]
            return value

        recovered = False
        if operation == "replace":
            old_start = exactly_one("OLD-BEGIN")
            new_start = exactly_one("NEW-BEGIN")
            old_end = optional_one("OLD-END")
            new_end = optional_one("NEW-END")
            if new_end is None:
                reason = "normal EOF cannot distinguish source text from trailing prose" if response_complete else "provider response ended without a final boundary"
                raise MutationFrameError("AMBIGUOUS", f"NEW-END is missing; {reason}")
            if not old_start < new_start < new_end:
                raise MutationFrameError("AMBIGUOUS", "replace boundary order is invalid")
            if any(text.strip() for text in lines[3:old_start]):
                raise MutationFrameError("AMBIGUOUS", "unexpected content precedes OLD-BEGIN")
            if old_end is None:
                # The unique matching NEW-BEGIN safely closes OLD.
                old_end = new_start
                recovered = True
            elif not old_start < old_end < new_start:
                raise MutationFrameError("AMBIGUOUS", "OLD-END is not between OLD-BEGIN and NEW-BEGIN")
            if any(text.strip() for text in lines[old_end + 1 : new_start]):
                raise MutationFrameError("AMBIGUOUS", "unexpected content appears between replace sections")
            old = section(old_start, old_end)
            content = section(new_start, new_end)
            trailing = "".join(lines[new_end + 1 :])
            if trailing.strip():
                raise MutationFrameError("AMBIGUOUS", "mutation response has trailing unstructured content")
            data: dict[str, object] = {
                "operation": operation,
                "old": old,
                "content": content,
                "done": done_text == "true",
                "frame_status": "RECOVERED_VALID" if recovered else "VALID",
            }
        else:
            content_start = exactly_one("CONTENT-BEGIN")
            content_end = optional_one("CONTENT-END")
            if content_end is None:
                reason = "normal EOF cannot distinguish source text from trailing prose" if response_complete else "provider response ended without a final boundary"
                raise MutationFrameError("AMBIGUOUS", f"CONTENT-END is missing; {reason}")
            if not content_start < content_end:
                raise MutationFrameError("AMBIGUOUS", "content boundary order is invalid")
            if any(text.strip() for text in lines[3:content_start]):
                raise MutationFrameError("AMBIGUOUS", "unexpected content precedes CONTENT-BEGIN")
            content = section(content_start, content_end)
            trailing = "".join(lines[content_end + 1 :])
            if trailing.strip():
                raise MutationFrameError("AMBIGUOUS", "mutation response has trailing unstructured content")
            data = {"operation": operation, "content": content, "done": done_text == "true", "frame_status": "VALID"}

        if not content or len(content) > max_patch_characters or len(str(data.get("old", ""))) + len(content) > max_patch_characters:
            raise MutationFrameError("AMBIGUOUS", "mutation patch exceeded its active text budget")
        return data

    @staticmethod
    def _parse_json_mutation(raw: dict[str, object], marker: str, expected_path: str, max_patch_characters: int) -> dict[str, object]:
        required = {"request_id", "path", "operation", "old", "content", "done"}
        if set(raw) != required:
            raise MutationFrameError("MALFORMED", "fallback mutation JSON has unexpected or missing fields")
        if raw.get("request_id") != marker or raw.get("path") != expected_path:
            raise MutationFrameError("AMBIGUOUS", "fallback mutation identity/path mismatch")
        operation, old, content, done = raw.get("operation"), raw.get("old"), raw.get("content"), raw.get("done")
        if not isinstance(operation, str) or operation not in {"create", "append", "replace"} or not isinstance(old, str) or not isinstance(content, str) or not isinstance(done, bool):
            raise MutationFrameError("MALFORMED", "fallback mutation fields have invalid types")
        if operation != "replace" and old:
            raise MutationFrameError("AMBIGUOUS", "fallback old text is only valid for replace")
        if not content or len(old) + len(content) > max_patch_characters:
            raise MutationFrameError("AMBIGUOUS", "fallback mutation exceeded its active text budget")
        return {"operation": operation, "old": old, "content": content, "done": done, "frame_status": "FALLBACK_VALID"}

    @staticmethod
    def _framed_section(
        raw: str,
        start_marker: str,
        end_marker: str,
        *,
        allow_trailing: bool = False,
        allow_eof: bool = False,
    ) -> str:
        start = start_marker + "\n"
        end = "\n" + end_marker
        if start not in raw:
            raise ProviderError(f"mutation response is missing {start_marker}")
        body = raw.split(start, 1)[1]
        if end not in body and allow_eof:
            return body
        if end not in body:
            raise ProviderError(f"mutation response is missing {end_marker}")
        content, trailing = body.split(end, 1)
        if trailing.strip() and not allow_trailing:
            raise ProviderError("mutation response has trailing content")
        return content

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

    def _ask(
        self,
        role: str,
        prompt: str,
        validator: callable,
        *,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        max_output_characters: int | None = None,
    ) -> AgentReply:
        error = ""
        self.last_structured_role = role
        self.last_structured_invalid_count = 0
        self.last_structured_repaired = False
        for attempt in range(self.structured_retries + 1):
            reply = self.provider.complete(
                AgentRequest(
                    role=role,
                    prompt=prompt if not error else prompt + f"\n\nYour previous JSON was invalid: {error}. Return only the required schema.",
                    system_prompt=system_prompt or ROLE_PROMPTS[role],
                    max_output_tokens=max_output_tokens,
                    max_output_characters=max_output_characters,
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
        # Older model replies can still carry a complete source file in a
        # legacy write/edit action. Never pass an oversized source envelope to
        # the controller: retain only its path and use the task context to
        # materialize the change through the bounded patch protocol instead.
        source_kinds = {"write_file", "append_file", "edit_file"}
        legacy_source_size = sum(
            sum(len(value) for key, value in action.items() if key in {"content", "old", "new"} and isinstance(value, str))
            for action in actions
            if isinstance(action, dict) and action.get("kind", action.get("action")) in source_kinds
        )
        if legacy_source_size > 6_000:
            for action in actions:
                if not isinstance(action, dict):
                    continue
                kind = action.get("kind", action.get("action"))
                path = action.get("path")
                if kind in source_kinds and isinstance(path, str) and path:
                    action.clear()
                    action.update({
                        "kind": "mutate_file",
                        "path": path,
                        "intent": "Materialize the requested source change through bounded patches.",
                    })
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
