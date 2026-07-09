"""
Genie — review_queue.py
Human Review Pipeline (§3.3).

Provides:
  - ReviewQueue: append completed tasks, read/update review status.
  - Statuses: pending → approved | rejected → revision_pending → (re-enters pending)
  - Storage: single JSONL file (logs/review_queue.jsonl), one JSON object per line.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Literal

from config import LOG_DIR

REVIEW_QUEUE_PATH = os.path.join(LOG_DIR, "review_queue.jsonl")

ReviewStatus = Literal[
    "pending",            # awaiting human review
    "approved",           # human approved — ready for delivery
    "rejected",           # human rejected — feedback attached
    "revision_pending",   # sent back to Genie for revision
    "delivered",          # final state after approval
]


@dataclass
class ReviewEntry:
    task_id: str
    goal: str
    summary: str
    output_files: list[str]
    cost_usd: float
    iterations: int
    self_assessment: str
    status: ReviewStatus = "pending"
    feedback: str = ""
    revision_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


class ReviewQueue:
    """JSONL-backed human review queue.

    Thread-safe for single-writer (orchestrator writes, bot reads/updates).
    """

    def __init__(self, path: str = REVIEW_QUEUE_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # ── Write ────────────────────────────────────────────────────────────

    def enqueue(self, entry: ReviewEntry) -> None:
        """Append a new review entry to the queue."""
        entry.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    # ── Read ─────────────────────────────────────────────────────────────

    def _load_all(self) -> list[dict]:
        """Read all entries from the JSONL file."""
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def get_pending(self) -> list[dict]:
        """Return all entries with status 'pending'."""
        return [e for e in self._load_all() if e.get("status") == "pending"]

    def get_by_task_id(self, task_id: str) -> dict | None:
        """Return the latest entry for the given task_id."""
        matches = [e for e in self._load_all() if e.get("task_id") == task_id]
        return matches[-1] if matches else None

    # ── Update ───────────────────────────────────────────────────────────

    def update_status(
        self,
        task_id: str,
        new_status: ReviewStatus,
        feedback: str = "",
    ) -> bool:
        """Update the status of a task in the queue.

        Rewrites the JSONL (safe for small queues; fine for < 10K entries).
        Returns True if the task was found and updated.
        """
        entries = self._load_all()
        found = False
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        for entry in entries:
            if entry.get("task_id") == task_id:
                entry["status"] = new_status
                entry["updated_at"] = now
                if feedback:
                    entry["feedback"] = feedback
                if new_status == "rejected":
                    entry["revision_count"] = entry.get("revision_count", 0) + 1
                found = True
        if found:
            with open(self.path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return found

    def mark_delivered(self, task_id: str) -> bool:
        """Convenience: mark a task as delivered."""
        return self.update_status(task_id, "delivered")
