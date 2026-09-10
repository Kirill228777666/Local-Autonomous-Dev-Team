"""Passive watchdog checks for persisted autonomous-run liveness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .models import Heartbeat


@dataclass(frozen=True, slots=True)
class Watchdog:
    stale_after_seconds: float = 900.0

    def is_stale(self, heartbeat: Heartbeat) -> bool:
        timestamp = datetime.fromisoformat(heartbeat.timestamp)
        age = (datetime.now(UTC) - timestamp).total_seconds()
        return age > self.stale_after_seconds
