"""Controller-owned staging for bounded, literal mutation response chunks."""

from __future__ import annotations

from dataclasses import dataclass, field


class MutationStreamError(ValueError):
    """A staged mutation stream cannot be safely continued or applied."""


@dataclass(slots=True)
class StagedMutationBody:
    request_id: str
    path: str
    expected_file_hash: str
    start_offset: int
    end_offset: int
    expected_slice: str
    max_bytes: int = 256_000
    max_chunks: int = 128
    chunks: list[str] = field(default_factory=list)
    content: str = ""
    total_bytes: int = 0
    complete: bool = False
    last_join_outcome: str = "APPENDED"

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def add_chunk(self, chunk: str, *, response_complete: bool, done_reason: str | None) -> bool:
        """Stage one provider response; return true only on a normal final stop."""
        if self.complete:
            raise MutationStreamError("STREAM_ALREADY_COMPLETE")
        if not response_complete:
            raise MutationStreamError("PROVIDER_RESPONSE_INCOMPLETE")
        if not isinstance(chunk, str) or not chunk:
            raise MutationStreamError("EMPTY_CHUNK")
        if len(self.chunks) >= self.max_chunks:
            raise MutationStreamError("MAX_CHUNKS_EXCEEDED")
        chunk_bytes = len(chunk.encode("utf-8"))
        if self.chunks and chunk == self.chunks[-1]:
            raise MutationStreamError("NO_PROGRESS_IDENTICAL_CHUNK")

        overlap = self._unique_exact_overlap(self.content, chunk)
        if overlap is None:
            raise MutationStreamError("AMBIGUOUS_OVERLAP")
        if overlap:
            chunk = chunk[overlap:]
            chunk_bytes = len(chunk.encode("utf-8"))
            self.last_join_outcome = "EXACT_OVERLAP_REMOVED"
        else:
            self.last_join_outcome = "APPENDED"
        if not chunk and done_reason == "length":
            raise MutationStreamError("NO_PROGRESS_EMPTY_CONTINUATION")
        if self.total_bytes + chunk_bytes > self.max_bytes:
            raise MutationStreamError("MAX_BYTES_EXCEEDED")

        self.chunks.append(chunk)
        self.content += chunk
        self.total_bytes += chunk_bytes
        if done_reason == "length":
            return False
        if done_reason not in {None, "stop", "end_turn", "eos_token"}:
            raise MutationStreamError(f"UNSUPPORTED_FINISH_REASON:{done_reason}")
        self.complete = True
        return True

    @staticmethod
    def _unique_exact_overlap(previous: str, following: str, *, minimum: int = 32, maximum: int = 4096) -> int | None:
        """Return a unique long suffix/prefix overlap, zero for no safe overlap."""
        pattern = following[:maximum]
        if len(previous) < minimum or len(pattern) < minimum:
            return 0
        table = StagedMutationBody._prefix_table(pattern)
        matched = 0
        # Only a bounded suffix is needed to find the longest possible join.
        for char in previous[-min(len(previous), len(pattern)):]:
            while matched and char != pattern[matched]:
                matched = table[matched - 1]
            if char == pattern[matched]:
                matched += 1
                if matched == len(pattern):
                    matched = table[matched - 1]
        if matched < minimum:
            return 0

        candidate = following[:matched]
        candidate_table = table[:matched]
        count = 0
        matched = 0
        for char in previous:
            while matched and char != candidate[matched]:
                matched = candidate_table[matched - 1]
            if char == candidate[matched]:
                matched += 1
                if matched == len(candidate):
                    count += 1
                    if count > 1:
                        return None
                    matched = candidate_table[matched - 1]
        return len(candidate)

    @staticmethod
    def _prefix_table(pattern: str) -> list[int]:
        table = [0] * len(pattern)
        matched = 0
        for index in range(1, len(pattern)):
            while matched and pattern[index] != pattern[matched]:
                matched = table[matched - 1]
            if pattern[index] == pattern[matched]:
                matched += 1
                table[index] = matched
        return table
