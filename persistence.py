"""
Genie — persistence.py
Cost tracking, task counting, and checkpoint I/O.

Extracted from orchestrator.py to isolate pure file I/O operations
from the brain loop.
"""
from __future__ import annotations

import json
import os
import sys
import time

from config import CHECKPOINT_PATH, COST_MONTHLY_PATH, LOG_DIR, TASK_LOG_PATH


# =========================================================================
# Monthly Cost
# =========================================================================

def load_monthly_cost() -> tuple[float, bool]:
    """Load monthly cost from cost_monthly.json.

    Returns (cost_usd, month_reset).  *month_reset* is True when the
    stored month doesn't match the current month (caller should reset
    its budget-warning flag).
    """
    current_month = time.strftime("%Y-%m")
    try:
        with open(COST_MONTHLY_PATH, "r") as f:
            data = json.load(f)
        if data.get("month") == current_month:
            return (data.get("cost_usd", 0.0), False)
        else:
            # Month boundary — reset
            write_monthly_cost(0.0)
            return (0.0, True)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return (0.0, True)


def write_monthly_cost(cost_usd: float) -> None:
    """Atomic write to cost_monthly.json."""
    current_month = time.strftime("%Y-%m")
    data = {
        "month": current_month,
        "cost_usd": cost_usd,
        "task_count": get_task_count(),
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    tmp_path = COST_MONTHLY_PATH + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.rename(tmp_path, COST_MONTHLY_PATH)
    except OSError as exc:
        print(
            f"[PERSISTENCE] cost_monthly write failed: {exc}",
            file=sys.stderr,
        )


# =========================================================================
# Task Count
# =========================================================================

def get_task_count() -> int:
    """Read current task_count from cost_monthly.json."""
    try:
        with open(COST_MONTHLY_PATH, "r") as f:
            data = json.load(f)
        return data.get("task_count", 0)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0


def increment_task_count(monthly_cost_usd: float) -> None:
    """Increment task_count in cost_monthly.json."""
    current_month = time.strftime("%Y-%m")
    try:
        with open(COST_MONTHLY_PATH, "r") as f:
            data = json.load(f)
        if data.get("month") != current_month:
            data = {"month": current_month, "cost_usd": 0.0, "task_count": 0}
        data["task_count"] = data.get("task_count", 0) + 1
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        data = {
            "month": current_month,
            "cost_usd": monthly_cost_usd,
            "task_count": 1,
        }

    os.makedirs(LOG_DIR, exist_ok=True)
    tmp_path = COST_MONTHLY_PATH + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.rename(tmp_path, COST_MONTHLY_PATH)
    except OSError as exc:
        print(
            f"[PERSISTENCE] task_count write failed: {exc}",
            file=sys.stderr,
        )


# =========================================================================
# Checkpoint
# =========================================================================

def load_checkpoint() -> dict | None:
    """Read CHECKPOINT_PATH and return full dict, or None."""
    try:
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        # Validate required keys
        _ = data["task_id"], data["goal"], data["iteration"]
        return data
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, KeyError) as exc:
        print(
            f"[PERSISTENCE] corrupt checkpoint, deleting: {exc}",
            file=sys.stderr,
        )
        try:
            os.remove(CHECKPOINT_PATH)
        except OSError:
            pass
        return None


def write_checkpoint(
    task_id: str,
    goal: str,
    task_type: str,
    per_task_budget: float,
    iteration: int,
    sequence: int,
    cost_usd: float,
    last_observation: dict | None,
    goaltracker: dict | None = None,
    scratchpad: dict | None = None,
) -> None:
    """Atomic checkpoint write."""
    data = {
        "task_id": task_id,
        "goal": goal,
        "task_type": task_type,
        "per_task_budget": per_task_budget,
        "iteration": iteration,
        "sequence": sequence,
        "cost_usd": cost_usd,
        "last_observation": last_observation,
    }
    if goaltracker is not None:
        data["goaltracker"] = goaltracker
    if scratchpad is not None:
        data["scratchpad"] = scratchpad
    os.makedirs(LOG_DIR, exist_ok=True)
    tmp_path = CHECKPOINT_PATH + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, default=str)
        os.rename(tmp_path, CHECKPOINT_PATH)
    except OSError as exc:
        print(
            f"[PERSISTENCE] checkpoint write failed: {exc}",
            file=sys.stderr,
        )


# =========================================================================
# Task Log
# =========================================================================

def append_task_log(
    task_id: str,
    goal: str,
    task_type: str,
    model: str,
    outcome: str,
    iterations: int,
    cost_usd: float,
    wall_time_s: float,
    scratchpad_misses: int = 0,
) -> None:
    """Atomically append a single record to TASK_LOG_PATH (JSONL format)."""
    record = {
        "task_id": task_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "goal": goal.strip()[:120],
        "task_type": task_type,
        "model": model,
        "outcome": outcome,
        "iterations": iterations,
        "cost_usd": cost_usd,
        "wall_time_s": wall_time_s,
        "scratchpad_misses": scratchpad_misses,
    }
    new_line = json.dumps(record, default=str) + "\n"
    os.makedirs(LOG_DIR, exist_ok=True)
    tmp_path = TASK_LOG_PATH + ".tmp"
    try:
        try:
            with open(TASK_LOG_PATH, "r") as f:
                existing = f.read()
        except FileNotFoundError:
            existing = ""
        content = existing + new_line
        with open(tmp_path, "w") as f:
            f.write(content)
        os.rename(tmp_path, TASK_LOG_PATH)
    except OSError as exc:
        print(f"[PERSISTENCE] task_log write failed: {exc}", file=sys.stderr)
