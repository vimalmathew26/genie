"""
Genie — task_queue.py
Persistent SQLite task queue for Phase 6.

Provides:
  - enqueue()    — insert a new pending task from CLI or API
  - run_queue()  — daemon loop: poll, claim, execute via GenieOrchestrator
  - get_status() — inspect all tasks (CLI status command)

CLI usage:
  python task_queue.py add "<goal>" [--type <task_type>] [--budget <float>]
                                    [--source <str>] [--repo <path>]
                                    [--priority <int>]
  python task_queue.py run
  python task_queue.py status
"""
from __future__ import annotations

import argparse
import os
import signal
import sqlite3
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

from config import (
    TASK_BUDGET_DEFAULTS,
    TASK_QUEUE_DB_PATH,
)


# =========================================================================
# Constants
# =========================================================================

# Seconds between polls when the queue is empty.  5 s keeps latency low
# without measurable CPU overhead on SQLite reads.
POLL_INTERVAL_S = 5.0

# A running task whose heartbeat_at is older than this many seconds AND
# whose worker_id differs from the current process is considered stale.
STALE_HEARTBEAT_THRESHOLD_S = 600  # 10 minutes


# =========================================================================
# Schema
# =========================================================================

# Known-gap columns — present in the schema but have NO automatic write
# path from TaskResult or from the current run_task() return value:
#
#   client_id          — reserved for future multi-tenant / API-key routing.
#   parent_task_id     — reserved for future subtask decomposition at queue
#                        level (GoalTracker operates inside a single task).
#   estimated_iterations — plan_phase() produces this, but it is not
#                          surfaced through TaskResult or stored after
#                          run_task() returns.
#   github_pr_url      — open_pr action may create a PR but the URL is
#                         not captured in TaskResult.
#   handoff_path       — assemble_handoff action may produce a path but
#                         it is not captured in TaskResult.
#
# These columns will remain NULL until orchestrator.py / TaskResult are
# extended to populate them.

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS tasks (
    task_id              TEXT    PRIMARY KEY,
    queued_at            TEXT    NOT NULL,
    started_at           TEXT,
    completed_at         TEXT,
    status               TEXT    NOT NULL DEFAULT 'pending',
    goal                 TEXT    NOT NULL,
    task_type            TEXT    NOT NULL DEFAULT 'default',
    budget               REAL,
    source               TEXT,
    repo_path            TEXT,
    priority             INTEGER NOT NULL DEFAULT 0,
    client_id            TEXT,
    parent_task_id       TEXT,
    worker_id            TEXT,
    heartbeat_at         TEXT,
    retry_count          INTEGER NOT NULL DEFAULT 0,
    max_retries          INTEGER NOT NULL DEFAULT 3,
    error_message        TEXT,
    estimated_iterations INTEGER,
    result_outcome       TEXT,
    result_cost_usd      REAL,
    result_iterations    INTEGER,
    result_wall_time_s   REAL,
    model                TEXT,
    github_pr_url        TEXT,
    handoff_path         TEXT
);
"""


# =========================================================================
# TaskQueue
# =========================================================================

class TaskQueue:
    """Persistent SQLite task queue."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or TASK_QUEUE_DB_PATH
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._init_db()

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a fresh connection.  Each call-site should use its own
        connection for thread safety (SQLite allows concurrent readers but
        only one writer at a time)."""
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        # WAL mode gives concurrent readers + one writer without blocking.
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._conn()
        try:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _worker_id() -> str:
        """Worker identity: PID @ hostname.  Unique enough for stale detection."""
        import socket
        return f"{os.getpid()}@{socket.gethostname()}"

    # -----------------------------------------------------------------
    # enqueue
    # -----------------------------------------------------------------

    def enqueue(
        self,
        goal: str,
        task_type: str = "default",
        budget_override: float | None = None,
        source: str | None = None,
        repo_path: str | None = None,
        priority: int = 0,
    ) -> str:
        """Insert a new pending task and return its task_id.

        task_id is generated as uuid4 — consistent with how
        GenieOrchestrator.run_task() creates task_id.
        """
        task_id = str(uuid.uuid4())
        budget = budget_override if budget_override is not None else TASK_BUDGET_DEFAULTS.get(task_type, 2.0)
        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO tasks
                   (task_id, queued_at, status, goal, task_type, budget,
                    source, repo_path, priority)
                   VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?)""",
                (task_id, self._now_iso(), goal, task_type, budget,
                 source, repo_path, priority),
            )
            conn.commit()
        finally:
            conn.close()
        return task_id

    # -----------------------------------------------------------------
    # Stale task reset
    # -----------------------------------------------------------------

    def _reset_stale_tasks(self, current_worker: str) -> int:
        """Detect and reset stale running tasks.

        A task is stale when:
          1. status = 'running'
          2. heartbeat_at is older than STALE_HEARTBEAT_THRESHOLD_S
          3. worker_id is NOT the current process

        Reset behaviour:
          - If retry_count < max_retries → reset to 'pending', bump retry_count
          - If retry_count >= max_retries → mark 'dead'

        Returns the number of rows affected.
        """
        cutoff = datetime.fromtimestamp(
            time.time() - STALE_HEARTBEAT_THRESHOLD_S, tz=timezone.utc,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        conn = self._conn()
        try:
            # Use BEGIN IMMEDIATE so we hold the write lock for the
            # read-modify-write cycle.
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT task_id, retry_count, max_retries FROM tasks
                   WHERE status = 'running'
                     AND worker_id != ?
                     AND heartbeat_at < ?""",
                (current_worker, cutoff),
            ).fetchall()

            affected = 0
            for row in rows:
                tid = row["task_id"]
                retries = row["retry_count"]
                max_ret = row["max_retries"]
                if retries < max_ret:
                    conn.execute(
                        """UPDATE tasks
                           SET status = 'pending',
                               worker_id = NULL,
                               heartbeat_at = NULL,
                               started_at = NULL,
                               retry_count = retry_count + 1
                           WHERE task_id = ?""",
                        (tid,),
                    )
                else:
                    conn.execute(
                        """UPDATE tasks
                           SET status = 'dead',
                               completed_at = ?,
                               error_message = 'Stale: heartbeat timeout after max retries'
                           WHERE task_id = ?""",
                        (self._now_iso(), tid),
                    )
                affected += 1

            conn.commit()
            return affected
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Claim
    # -----------------------------------------------------------------

    def _claim_next(self, worker_id: str) -> dict | None:
        """Atomically claim the highest-priority pending task.

        Uses BEGIN IMMEDIATE to acquire a write lock before reading,
        preventing two concurrent workers from claiming the same row.
        Priority is DESC (higher number = higher priority); ties broken
        by queued_at ASC (oldest first).

        Returns a dict of the claimed row, or None if no pending tasks.
        """
        now = self._now_iso()
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT task_id, goal, task_type, budget, repo_path
                   FROM tasks
                   WHERE status = 'pending'
                   ORDER BY priority DESC, queued_at ASC
                   LIMIT 1""",
            ).fetchone()

            if row is None:
                conn.commit()
                return None

            tid = row["task_id"]
            conn.execute(
                """UPDATE tasks
                   SET status = 'running',
                       worker_id = ?,
                       started_at = ?,
                       heartbeat_at = ?
                   WHERE task_id = ? AND status = 'pending'""",
                (worker_id, now, now, tid),
            )
            conn.commit()
            return dict(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Heartbeat
    # -----------------------------------------------------------------

    def update_heartbeat(self, task_id: str) -> None:
        """Update heartbeat_at for a running task.  Called from the
        on_update callback that GenieOrchestrator fires every iteration."""
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE tasks SET heartbeat_at = ? WHERE task_id = ?",
                (self._now_iso(), task_id),
            )
            conn.commit()
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Result write-back
    # -----------------------------------------------------------------

    def _write_result(
        self,
        task_id: str,
        *,
        status: str,
        result_outcome: str | None = None,
        result_cost_usd: float | None = None,
        result_iterations: int | None = None,
        result_wall_time_s: float | None = None,
        model: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Write back result columns after run_task() returns or raises.

        Only populates fields that are actually available.  Columns with
        no write path (github_pr_url, handoff_path, estimated_iterations,
        client_id, parent_task_id) are left NULL.
        """
        conn = self._conn()
        try:
            conn.execute(
                """UPDATE tasks
                   SET status = ?,
                       completed_at = ?,
                       result_outcome = ?,
                       result_cost_usd = ?,
                       result_iterations = ?,
                       result_wall_time_s = ?,
                       model = ?,
                       error_message = ?
                   WHERE task_id = ?""",
                (
                    status,
                    self._now_iso(),
                    result_outcome,
                    result_cost_usd,
                    result_iterations,
                    result_wall_time_s,
                    model,
                    error_message,
                    task_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # get_status
    # -----------------------------------------------------------------

    def get_status(self) -> list[dict]:
        """Return all task rows ordered by queued_at DESC."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY queued_at DESC",
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # run_queue  (daemon loop)
    # -----------------------------------------------------------------

    def run_queue(self, orchestrator) -> None:
        """Blocking daemon loop.  Polls for pending tasks, executes them
        sequentially via orchestrator.run_task(), and sleeps between polls.

        Exits cleanly on KeyboardInterrupt or SIGTERM.

        Args:
            orchestrator: a fully constructed GenieOrchestrator instance.
        """
        shutdown = False

        def _signal_handler(signum, frame):
            nonlocal shutdown
            print(f"\n[TASK_QUEUE] Received signal {signum}, shutting down…",
                  file=sys.stderr)
            shutdown = True

        signal.signal(signal.SIGTERM, _signal_handler)

        worker = self._worker_id()
        print(f"[TASK_QUEUE] Daemon started (worker={worker})", flush=True)

        try:
            while not shutdown:
                # ── Stale reset ─────────────────────────────────────
                try:
                    n = self._reset_stale_tasks(worker)
                    if n:
                        print(f"[TASK_QUEUE] Reset {n} stale task(s)",
                              flush=True)
                except Exception as exc:
                    print(f"[TASK_QUEUE] Stale reset error: {exc}",
                          file=sys.stderr)

                # ── Claim ───────────────────────────────────────────
                task = self._claim_next(worker)
                if task is None:
                    time.sleep(POLL_INTERVAL_S)
                    continue

                tid = task["task_id"]
                goal = task["goal"]
                task_type = task["task_type"]
                budget = task["budget"]
                repo_path = task["repo_path"]

                print(
                    f"[TASK_QUEUE] Running task {tid[:8]}… "
                    f"goal={goal[:80]!r}  type={task_type}",
                    flush=True,
                )

                # ── Heartbeat callback ──────────────────────────────
                # Wrap update_heartbeat so orchestrator can call it as
                # on_update(event_dict) without knowing about TaskQueue.
                def _heartbeat_cb(event_dict: dict, _tid=tid) -> None:
                    try:
                        self.update_heartbeat(_tid)
                    except Exception:
                        pass  # heartbeat failures are non-fatal

                # ── Execute ─────────────────────────────────────────
                t0 = time.time()
                try:
                    result = orchestrator.run_task(
                        goal=goal,
                        mode="autonomous",
                        task_type=task_type,
                        per_task_budget=budget,
                        on_update=_heartbeat_cb,
                        skip_plan=False,
                        auto_confirm=True,
                    )
                    wall_time = time.time() - t0

                    # Map TaskResult.outcome to queue status.
                    if result.outcome in ("done",):
                        q_status = "done"
                    elif result.outcome in ("cancelled",):
                        q_status = "cancelled"
                    else:
                        # abort, budget_exceeded, iteration_exceeded,
                        # unrecoverable — all map to "failed" at queue level.
                        q_status = "failed"

                    # Read model from orchestrator internals — TaskResult
                    # does not carry it.
                    model = getattr(orchestrator, "_model", None)

                    self._write_result(
                        tid,
                        status=q_status,
                        result_outcome=result.outcome,
                        result_cost_usd=result.cost_usd,
                        result_iterations=result.iterations,
                        result_wall_time_s=round(wall_time, 2),
                        model=model,
                    )
                    print(
                        f"[TASK_QUEUE] Task {tid[:8]} → {q_status} "
                        f"({result.outcome})  "
                        f"cost=${result.cost_usd:.4f}  "
                        f"iters={result.iterations}  "
                        f"wall={wall_time:.1f}s",
                        flush=True,
                    )

                except KeyboardInterrupt:
                    # Propagate — outer handler will catch.
                    raise
                except Exception as exc:
                    wall_time = time.time() - t0
                    tb = traceback.format_exc()
                    self._write_result(
                        tid,
                        status="failed",
                        result_wall_time_s=round(wall_time, 2),
                        error_message=f"{type(exc).__name__}: {exc}\n{tb}",
                    )
                    print(
                        f"[TASK_QUEUE] Task {tid[:8]} EXCEPTION: {exc}",
                        file=sys.stderr, flush=True,
                    )

        except KeyboardInterrupt:
            print("\n[TASK_QUEUE] KeyboardInterrupt — exiting.", flush=True)

        print("[TASK_QUEUE] Daemon stopped.", flush=True)


# =========================================================================
# CLI
# =========================================================================

def _print_status_table(tasks: list[dict]) -> None:
    """Print a human-readable table of tasks for operational use."""
    if not tasks:
        print("No tasks in queue.")
        return

    # Columns: task_id (8 chars), queued_at, status, task_type, priority,
    # outcome, cost, iterations, wall_time, error (truncated)
    header = (
        f"{'ID':>8}  {'QUEUED':>19}  {'STATUS':<10} {'TYPE':<18} "
        f"{'PRI':>3}  {'OUTCOME':<20} {'COST':>8}  {'ITER':>4}  "
        f"{'WALL(s)':>8}  {'ERROR'}"
    )
    print(header)
    print("─" * len(header))

    for t in tasks:
        tid = (t.get("task_id") or "")[:8]
        queued = t.get("queued_at") or ""
        status = t.get("status") or ""
        ttype = t.get("task_type") or ""
        pri = t.get("priority", 0)
        outcome = t.get("result_outcome") or ""
        cost = t.get("result_cost_usd")
        cost_s = f"${cost:.4f}" if cost is not None else ""
        iters = t.get("result_iterations")
        iters_s = str(iters) if iters is not None else ""
        wall = t.get("result_wall_time_s")
        wall_s = f"{wall:.1f}" if wall is not None else ""
        err = (t.get("error_message") or "")[:60]

        print(
            f"{tid:>8}  {queued:>19}  {status:<10} {ttype:<18} "
            f"{pri:>3}  {outcome:<20} {cost_s:>8}  {iters_s:>4}  "
            f"{wall_s:>8}  {err}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genie Task Queue — enqueue, run, and inspect tasks.",
    )
    sub = parser.add_subparsers(dest="command")

    # -- add --
    add_p = sub.add_parser("add", help="Enqueue a new task")
    add_p.add_argument("goal", help="Task goal description")
    add_p.add_argument("--type", dest="task_type", default="default",
                       help="Task type (default: 'default')")
    add_p.add_argument("--budget", type=float, default=None,
                       help="Per-task budget override (USD)")
    add_p.add_argument("--source", default=None,
                       help="Source label (e.g. 'cli', 'api', 'telegram')")
    add_p.add_argument("--repo", dest="repo_path", default=None,
                       help="Repository path for code tasks")
    add_p.add_argument("--priority", type=int, default=0,
                       help="Priority (higher = sooner, default 0)")

    # -- run --
    sub.add_parser("run", help="Start daemon runner (blocks until interrupted)")

    # -- cancel --
    cancel_p = sub.add_parser("cancel", help="Cancel a pending task")
    cancel_p.add_argument("task_id", help="Task ID or unique prefix")

    # -- status --
    sub.add_parser("status", help="Show task queue status")

    args = parser.parse_args()

    if args.command == "add":
        tq = TaskQueue()
        # Default source to "cli" when invoked from this entry point.
        source = args.source if args.source is not None else "cli"
        tid = tq.enqueue(
            goal=args.goal,
            task_type=args.task_type,
            budget_override=args.budget,
            source=source,
            repo_path=args.repo_path,
            priority=args.priority,
        )
        print(f"Enqueued task {tid}")

    elif args.command == "cancel":
        tq = TaskQueue()
        tid_prefix = args.task_id
        # Support short prefixes — find the matching row.
        tasks = tq.get_status()
        matches = [t for t in tasks if t["task_id"].startswith(tid_prefix)]
        if len(matches) == 0:
            print(f"No task matching '{tid_prefix}'.", file=sys.stderr)
            sys.exit(1)
        elif len(matches) > 1:
            print(f"Ambiguous prefix '{tid_prefix}' — matches {len(matches)} tasks.",
                  file=sys.stderr)
            sys.exit(1)
        row = matches[0]
        if row["status"] not in ("pending",):
            print(f"Task {row['task_id'][:8]} is '{row['status']}', "
                  f"can only cancel 'pending' tasks.", file=sys.stderr)
            sys.exit(1)
        conn = tq._conn()
        try:
            conn.execute(
                "UPDATE tasks SET status = 'cancelled', completed_at = ? WHERE task_id = ?",
                (TaskQueue._now_iso(), row["task_id"]),
            )
            conn.commit()
        finally:
            conn.close()
        print(f"Cancelled task {row['task_id'][:8]}")

    elif args.command == "run":
        # Construct GenieOrchestrator exactly as genie.py and genie_suite.py
        # do: XdotoolController() → GenieOrchestrator(controller).
        from xdotool_controller import XdotoolController
        from orchestrator import GenieOrchestrator

        controller = XdotoolController()
        orchestrator = GenieOrchestrator(controller)

        tq = TaskQueue()
        tq.run_queue(orchestrator)

    elif args.command == "status":
        tq = TaskQueue()
        tasks = tq.get_status()
        _print_status_table(tasks)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
