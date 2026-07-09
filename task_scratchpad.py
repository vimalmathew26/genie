"""
Genie — task_scratchpad.py
Shared Working Memory for the ReAct brain loop.

A single structured JSON object living for the lifetime of one task,
injected into context before every iteration (ScratchpadReader),
updated after every iteration (ScratchpadWriter).

Schema: {facts, files, decisions, subtask_outcomes, errors}
- facts:            verbatim key-value pairs extracted from observations
- files:            path → role mapping discovered during execution
- decisions:        architectural/tool decisions made during the task
- subtask_outcomes: per-subtask completion status (collapsed when consecutive "done")
- errors:           failure context keyed by subtask or iteration

All categories except subtask_outcomes use a parallel _touched dict to
track which subtask last wrote each key (for recency-window eviction).

Token budget is a compression target (not a hard cap).  If protected
entries exceed the budget, the scratchpad renders in full — silent data
loss is worse than extra tokens in context.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

from config import (
    LOG_DIR,
    SCRATCHPAD_RECENCY_WINDOW,
    SCRATCHPAD_TOKEN_BUDGET,
)

# Rough chars-per-token estimate for budget calculation.
_CHARS_PER_TOKEN = 4


# =========================================================================
# TaskScratchpad
# =========================================================================

@dataclass
class TaskScratchpad:
    """Structured working memory for a single task."""

    facts: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    decisions: dict[str, str] = field(default_factory=dict)
    subtask_outcomes: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    # Parallel metadata: category → {key → subtask_index} tracking when
    # each entry was last written.  Not serialized into the rendered block
    # or the LLM-facing output — internal bookkeeping only.
    # subtask_outcomes doesn't need this — its keys ARE subtask indices.
    _touched: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "facts": {},
        "files": {},
        "decisions": {},
        "errors": {},
    })

    # Current subtask index — set by orchestrator at subtask start.
    _current_subtask: int = 0

    # Structured handoff list — kept in sync with subtask_outcomes but
    # retains the full dict form needed by generate_continuation_draft().
    # Capped at last 5 entries.
    _handoffs: list[dict] = field(default_factory=list)

    # ----------------------------------------------------------------
    # Write interface (called by ScratchpadWriter after parsing LLM output)
    # ----------------------------------------------------------------

    def update_category(
        self, category: str, updates: dict[str, str],
    ) -> None:
        """Merge updates into a category dict.  Stamps each key with
        the current subtask index for recency tracking."""
        store = self._get_store(category)
        for k, v in updates.items():
            store[k] = v
            if category in self._touched:
                self._touched[category][k] = self._current_subtask

    def replace_categories(
        self,
        subtask_outcomes: dict[str, str] | None = None,
        decisions: dict[str, str] | None = None,
        errors: dict[str, str] | None = None,
    ) -> None:
        """Wholesale replacement for inter-subtask writer output.
        Replaces entire dicts — not a merge.  Re-stamps all keys."""
        if subtask_outcomes is not None:
            self.subtask_outcomes = subtask_outcomes
        if decisions is not None:
            self.decisions = decisions
            self._touched["decisions"] = {
                k: self._current_subtask for k in decisions
            }
        if errors is not None:
            self.errors = errors
            self._touched["errors"] = {
                k: self._current_subtask for k in errors
            }

    def set_subtask(self, subtask_index: int) -> None:
        """Called by orchestrator at the start of each subtask."""
        self._current_subtask = subtask_index

    # ----------------------------------------------------------------
    # Compatibility shims (called by orchestrator handoff logic)
    # ----------------------------------------------------------------

    def add_handoff(self, handoff: dict) -> None:
        """Write a subtask handoff into subtask_outcomes.

        Converts the structured handoff dict into a compact outcome entry
        so the baseline scratchpad architecture carries it forward.
        """
        n = handoff.get("subtask_n", "?")
        status = handoff.get("status", "?")
        msg = handoff.get("handoff_message", "")
        files_w = handoff.get("files_written", [])
        cmds_f = handoff.get("commands_failed", [])

        parts = [status]
        if msg:
            parts.append(msg[:400])
        if files_w:
            parts.append(f"files={','.join(str(f) for f in files_w[:10])}")
        if cmds_f:
            parts.append(f"failed_cmds={';'.join(str(c) for c in cmds_f[:5])}")

        self.update_category("subtask_outcomes", {str(n): " | ".join(parts)})

        # Also store structured handoff for continuation planner.
        self._handoffs.append(handoff)
        # Keep only last 5.
        if len(self._handoffs) > 5:
            self._handoffs = self._handoffs[-5:]

    def update_files(self, updates: dict[str, str]) -> None:
        """Compatibility shim — delegates to update_category('files', ...)."""
        self.update_category("files", updates)

    @property
    def handoffs(self) -> list[dict]:
        """Structured handoff list for continuation planner."""
        return self._handoffs

    # ----------------------------------------------------------------
    # Eviction / compression
    # ----------------------------------------------------------------

    def compress(self) -> None:
        """Apply eviction and compression to fit within token budget.

        Eviction order (outside recency window):
          errors → subtask_outcomes (collapsed) → decisions (truncated) →
          files (oldest first) → facts (oldest first, last resort)

        Hard floor: most recent entry in every non-empty category is kept.
        Budget is a compression target — if protected entries still exceed
        it, rendering proceeds in full (no silent data loss).
        """
        # Always collapse subtask_outcomes regardless of budget.
        self._collapse_outcomes()

        budget_chars = SCRATCHPAD_TOKEN_BUDGET * _CHARS_PER_TOKEN
        if self._char_count() <= budget_chars:
            return

        window_floor = max(0, self._current_subtask - SCRATCHPAD_RECENCY_WINDOW + 1)

        # Phase 1: evict errors outside recency window (oldest first)
        self._evict_outside_window("errors", window_floor, budget_chars)
        if self._char_count() <= budget_chars:
            return

        # Phase 2: truncate decision values to 60 chars
        for k, v in list(self.decisions.items()):
            if len(v) > 60:
                self.decisions[k] = v[:57] + "..."
        if self._char_count() <= budget_chars:
            return

        # Phase 3: evict decisions outside recency window (oldest first)
        self._evict_outside_window("decisions", window_floor, budget_chars)
        if self._char_count() <= budget_chars:
            return

        # Phase 4: evict files by budget pressure only (never by subtask age).
        # File-role entries are the only persistent record of codebase discovery
        # during fix/modify tasks — evicting them by age causes the agent to
        # re-read files it already processed, generating scratchpad churn.
        self._evict_outside_window("files", 0, budget_chars)
        if self._char_count() <= budget_chars:
            return

        # Phase 5: evict facts outside recency window (oldest first, last resort)
        self._evict_outside_window("facts", window_floor, budget_chars)
        # If still over budget after all eviction, render in full.
        # Silent data loss is worse than extra tokens.

    def _collapse_outcomes(self) -> None:
        """Collapse consecutive 'done' entries in subtask_outcomes to ranges.

        {"1": "done", "2": "done", "3": "done", "4": "failed — retried"}
        becomes {"1-3": "done", "4": "failed — retried"}

        Failed/non-done entries are never collapsed.
        """
        if not self.subtask_outcomes:
            return

        # Parse keys into (int_key, original_key, value) where possible.
        # Range keys like "1-3" are left as-is (already collapsed).
        entries: list[tuple[int | None, str, str]] = []
        for k, v in self.subtask_outcomes.items():
            # Try to parse as single int (subtask index).
            try:
                entries.append((int(k), k, v))
            except ValueError:
                # Already a range key or non-numeric — keep as-is.
                entries.append((None, k, v))

        # Sort: numeric keys first (by int), then non-numeric in original order.
        numeric = sorted(
            [(i, k, v) for i, k, v in entries if i is not None],
            key=lambda x: x[0],
        )
        non_numeric = [(i, k, v) for i, k, v in entries if i is None]

        # Collapse consecutive "done" in numeric entries.
        collapsed: dict[str, str] = {}
        run_start: int | None = None
        run_end: int | None = None

        for idx, _orig_key, val in numeric:
            if val == "done":
                if run_start is None:
                    run_start = idx
                    run_end = idx
                elif idx == run_end + 1:
                    run_end = idx
                else:
                    # Non-consecutive done — flush previous run.
                    collapsed[self._range_key(run_start, run_end)] = "done"
                    run_start = idx
                    run_end = idx
            else:
                # Flush any pending done-run.
                if run_start is not None:
                    collapsed[self._range_key(run_start, run_end)] = "done"
                    run_start = None
                    run_end = None
                collapsed[str(idx)] = val

        # Flush trailing done-run.
        if run_start is not None:
            collapsed[self._range_key(run_start, run_end)] = "done"

        # Append non-numeric keys at the end.
        for _i, k, v in non_numeric:
            collapsed[k] = v

        self.subtask_outcomes = collapsed

    @staticmethod
    def _range_key(start: int, end: int) -> str:
        return str(start) if start == end else f"{start}-{end}"

    def _evict_outside_window(
        self, category: str, window_floor: int, budget_chars: int,
    ) -> None:
        """Remove entries outside the recency window, oldest first.
        Always keeps at least the most recent entry (hard floor)."""
        store = self._get_store(category)
        touched = self._touched.get(category, {})
        if len(store) <= 1:
            return

        # Build list of (key, subtask_touched), sort oldest first.
        eviction_candidates = [
            (k, touched.get(k, 0))
            for k in store
            if touched.get(k, 0) < window_floor
        ]
        eviction_candidates.sort(key=lambda x: x[1])

        # Must keep at least one entry (hard floor = most recent).
        max_removable = len(store) - 1

        removed = 0
        for k, _st in eviction_candidates:
            if removed >= max_removable:
                break
            del store[k]
            touched.pop(k, None)
            removed += 1
            if self._char_count() <= budget_chars:
                break

    def _get_store(self, category: str) -> dict[str, str]:
        stores = {
            "facts": self.facts,
            "files": self.files,
            "decisions": self.decisions,
            "subtask_outcomes": self.subtask_outcomes,
            "errors": self.errors,
        }
        if category not in stores:
            raise ValueError(f"Unknown scratchpad category: {category}")
        return stores[category]

    def _char_count(self) -> int:
        """Rough character count of the rendered scratchpad."""
        return len(self.render())

    # ----------------------------------------------------------------
    # Render (ScratchpadReader uses this)
    # ----------------------------------------------------------------

    def render(self) -> str:
        """Render scratchpad as a compact key=value block for context injection.

        Returns empty string if all categories are empty.
        """
        sections: list[str] = []
        for label, store in [
            ("facts", self.facts),
            ("files", self.files),
            ("decisions", self.decisions),
            ("subtask_outcomes", self.subtask_outcomes),
            ("errors", self.errors),
        ]:
            if store:
                lines = [f"  {k}={v}" for k, v in store.items()]
                sections.append(f"{label}:\n" + "\n".join(lines))

        if not sections:
            return ""
        return "\n".join(sections)

    def is_empty(self) -> bool:
        return not any([
            self.facts, self.files, self.decisions,
            self.subtask_outcomes, self.errors,
        ])

    # ----------------------------------------------------------------
    # Serialization (disk persistence alongside checkpoint.json)
    # ----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for checkpoint persistence."""
        return {
            "facts": dict(self.facts),
            "files": dict(self.files),
            "decisions": dict(self.decisions),
            "subtask_outcomes": dict(self.subtask_outcomes),
            "errors": dict(self.errors),
            "_touched": {
                cat: dict(keys) for cat, keys in self._touched.items()
            },
            "_current_subtask": self._current_subtask,
            "_handoffs": list(self._handoffs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskScratchpad:
        """Deserialize from a checkpoint dict.  Tolerant of missing keys."""
        pad = cls(
            facts=data.get("facts", {}),
            files=data.get("files", {}),
            decisions=data.get("decisions", {}),
            subtask_outcomes=data.get("subtask_outcomes", {}),
            errors=data.get("errors", {}),
        )
        raw_touched = data.get("_touched", {})
        for cat in ("facts", "files", "decisions", "errors"):
            if cat in raw_touched and isinstance(raw_touched[cat], dict):
                # JSON keys are strings — convert back to int values.
                pad._touched[cat] = {
                    k: int(v) for k, v in raw_touched[cat].items()
                }
        pad._current_subtask = int(data.get("_current_subtask", 0))
        pad._handoffs = list(data.get("_handoffs", []))
        return pad

    # ----------------------------------------------------------------
    # Disk I/O (separate file, same dir as checkpoint)
    # ----------------------------------------------------------------

    _SCRATCHPAD_PATH = os.path.join(LOG_DIR, "scratchpad.json")

    def save(self) -> None:
        """Atomic write to scratchpad.json."""
        os.makedirs(LOG_DIR, exist_ok=True)
        tmp_path = self._SCRATCHPAD_PATH + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(self.to_dict(), f, default=str)
            os.rename(tmp_path, self._SCRATCHPAD_PATH)
        except OSError as exc:
            print(
                f"[SCRATCHPAD] save failed: {exc}",
                file=sys.stderr,
            )

    @classmethod
    def load(cls) -> TaskScratchpad | None:
        """Load from scratchpad.json.  Returns None if missing or corrupt."""
        try:
            with open(cls._SCRATCHPAD_PATH, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(
                f"[SCRATCHPAD] corrupt file, ignoring: {exc}",
                file=sys.stderr,
            )
            return None

    @classmethod
    def clear_disk(cls) -> None:
        """Remove scratchpad.json (called on task completion)."""
        try:
            os.remove(cls._SCRATCHPAD_PATH)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(
                f"[SCRATCHPAD] clear_disk failed: {exc}",
                file=sys.stderr,
            )