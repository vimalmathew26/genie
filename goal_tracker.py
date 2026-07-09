"""GoalTracker task decomposition and continuation-plan helpers. Extracted from planner.py."""
from __future__ import annotations

import ast
import config
import dataclasses
import json
import os
import re
import time
from typing import TYPE_CHECKING

import httpx

from config import (
    LLM_SERVICE_ERROR_CODES,
    MAX_GOALTRACKER_SUBTASKS,
    MAX_LLM_RETRIES,
    MODEL_ROSTER,
    RETRY_BACKOFF_SECONDS,
    SUBTASK_INTERFACE_RECENT_WINDOW,
    TASK_MODEL_MAP,
)
from exceptions import ResponseTruncatedError

if TYPE_CHECKING:
    from orchestrator import GenieOrchestrator


# =========================================================================
# Continuation-plan scoping helpers
# =========================================================================

# Regex to extract file references from subtask descriptions.
# Matches paths like dag_scheduler/executor.py, cli.py, jobs/cycle.yaml, etc.
_FILE_REF_RE = re.compile(r'[\w/._-]+\.(?:py|yaml|toml|json|cfg|txt|md|sh)\b')


def _extract_file_refs(description: str) -> set[str]:
    """Extract file basenames from a subtask description.

    >>> _extract_file_refs("Modify dag_scheduler/executor.py: add scheduler param")
    {'executor.py'}
    """
    return {os.path.basename(m) for m in _FILE_REF_RE.findall(description)}


def _partition_pending(
    failed_st: "Subtask",
    pending_subtasks: list["Subtask"],
) -> tuple[list["Subtask"], list["Subtask"]]:
    """Split pending subtasks into (affected, unaffected) by file overlap.

    A pending subtask is 'affected' if it references any of the same files
    as the failed subtask.  Unaffected subtasks are preserved verbatim —
    the continuation LLM never touches them.

    Returns (affected, unaffected) where each list preserves original order.
    """
    failed_files = _extract_file_refs(failed_st.description)
    if not failed_files:
        # Can't determine scope — treat all as affected (full replan)
        return list(pending_subtasks), []

    affected: list[Subtask] = []
    unaffected: list[Subtask] = []
    for st in pending_subtasks:
        st_files = _extract_file_refs(st.description)
        if not st_files:
            # No file refs (e.g. verification subtask) — preserve it
            unaffected.append(st)
        elif st_files & failed_files:
            affected.append(st)
        else:
            unaffected.append(st)
    return affected, unaffected


# Tier ordering for _highest_tier (higher index = stronger model)
_TIER_ORDER = ["tier_0", "tier_1", "tier_2", "tier_3", "tier_4"]


def _highest_tier(tiers: list[str]) -> str:
    """Return the highest model tier from a list of tier strings."""
    best = 0
    for t in tiers:
        idx = _TIER_ORDER.index(t) if t in _TIER_ORDER else 0
        best = max(best, idx)
    return _TIER_ORDER[best]


def _merge_same_file_subtasks(subtasks: list["Subtask"]) -> list["Subtask"]:
    """Merge subtasks that touch the SAME single file into one.

    Deterministic enforcement of the SINGLE-FILE RULE on LLM output.
    Only merges when both subtasks reference exactly one file and it's the
    same file — multi-file subtasks are left alone.

    Returns a new list with merged subtasks renumbered starting from 1.
    """
    if len(subtasks) <= 1:
        return subtasks

    # Map: primary file basename → list of indices (only for single-file subtasks)
    file_to_indices: dict[str, list[int]] = {}
    for i, st in enumerate(subtasks):
        files = _extract_file_refs(st.description)
        if len(files) == 1:
            fname = next(iter(files))
            file_to_indices.setdefault(fname, []).append(i)

    # Identify files with >1 subtask
    merge_targets = {f: idxs for f, idxs in file_to_indices.items() if len(idxs) > 1}
    if not merge_targets:
        return subtasks

    merged: list[Subtask] = []
    skip: set[int] = set()
    merge_log: list[str] = []

    for i, st in enumerate(subtasks):
        if i in skip:
            continue
        files = _extract_file_refs(st.description)
        primary = next(iter(files)) if len(files) == 1 else None
        if primary and primary in merge_targets and i == merge_targets[primary][0]:
            # First subtask for this file → absorb all siblings
            sibling_indices = merge_targets[primary]
            siblings = [subtasks[j] for j in sibling_indices]
            combined_desc = " THEN ".join(s.description for s in siblings)
            best_tier = _highest_tier([s.model_tier for s in siblings])
            merged.append(Subtask(
                n=len(merged) + 1,
                description=combined_desc,
                model_tier=best_tier,
            ))
            skip.update(sibling_indices[1:])
            merge_log.append(f"{primary}: merged {len(siblings)} subtasks into 1")
        else:
            merged.append(Subtask(
                n=len(merged) + 1,
                description=st.description,
                model_tier=st.model_tier,
            ))

    if merge_log:
        print(
            f"  [goaltracker] Same-file merge: {'; '.join(merge_log)} "
            f"(plan {len(subtasks)} → {len(merged)} subtasks)",
            flush=True,
        )
    return merged


# =========================================================================
# GoalTracker — Task Decomposition (Phase 5.3)
# =========================================================================

@dataclasses.dataclass
class Subtask:
    """Single subtask within a GoalTracker decomposition."""
    n: int
    description: str
    status: str = "pending"          # "pending" | "running" | "done" | "failed"
    # result_summary is used for failure diagnosis and final_summary() display.
    # It is NOT injected into subsequent subtask goals — interfaces{} is used instead.
    result_summary: str = ""         # filled on done/failed
    model_tier: str = "tier_0"       # assigned by router during decompose()
    files_written: list[str] = None  # paths written by this subtask (set on done)
    interfaces: dict = None          # filename → list of public symbol signatures (AST-extracted on done)

    def __post_init__(self):
        if self.files_written is None:
            self.files_written = []
        if self.interfaces is None:
            self.interfaces = {}


# =========================================================================
# AST-based interface extraction
# =========================================================================

def _format_arg(arg: ast.arg) -> str:
    """Format a single function argument with optional type annotation."""
    name = arg.arg
    if arg.annotation:
        try:
            name += ": " + ast.unparse(arg.annotation)
        except Exception:
            pass
    return name


def _format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a readable Python signature string from a FunctionDef node.

    Handles positional-only, regular, *args, keyword-only, **kwargs,
    default values, and return annotation.
    """
    args = node.args
    parts: list[str] = []

    # --- positional-only args ---
    num_posonlyargs = len(args.posonlyargs)
    # defaults are right-aligned: last N defaults correspond to last N args
    # For posonlyargs, defaults come from args.defaults shared pool.
    # Total positional args = posonlyargs + args.args
    total_positional = num_posonlyargs + len(args.args)
    num_defaults = len(args.defaults)
    # defaults offset: first (total_positional - num_defaults) args have no default
    for i, arg in enumerate(args.posonlyargs):
        s = _format_arg(arg)
        # default index: i - (total_positional - num_defaults)
        def_idx = i - (total_positional - num_defaults)
        if def_idx >= 0 and def_idx < num_defaults:
            try:
                s += "=" + ast.unparse(args.defaults[def_idx])
            except Exception:
                pass
        parts.append(s)

    if num_posonlyargs > 0:
        parts.append("/")

    # --- regular positional args ---
    for i, arg in enumerate(args.args):
        s = _format_arg(arg)
        def_idx = (num_posonlyargs + i) - (total_positional - num_defaults)
        if def_idx >= 0 and def_idx < num_defaults:
            try:
                s += "=" + ast.unparse(args.defaults[def_idx])
            except Exception:
                pass
        parts.append(s)

    # --- *args ---
    if args.vararg:
        parts.append("*" + _format_arg(args.vararg))
    elif args.kwonlyargs:
        # bare * separator when there are keyword-only args but no *args
        parts.append("*")

    # --- keyword-only args ---
    for i, arg in enumerate(args.kwonlyargs):
        s = _format_arg(arg)
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            try:
                s += "=" + ast.unparse(args.kw_defaults[i])
            except Exception:
                pass
        parts.append(s)

    # --- **kwargs ---
    if args.kwarg:
        parts.append("**" + _format_arg(args.kwarg))

    sig = f"({', '.join(parts)})"

    # --- return annotation ---
    if node.returns:
        try:
            sig += " -> " + ast.unparse(node.returns)
        except Exception:
            pass

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}{sig}"


def _is_private(name: str) -> bool:
    """Return True for private names (leading underscore), except __init__."""
    if name == "__init__":
        return False
    return name.startswith("_")


def extract_interfaces(file_paths: list[str]) -> dict[str, list[str]]:
    """Extract public symbol signatures from Python files via AST.

    Args:
        file_paths: List of absolute file paths to inspect.

    Returns:
        Dict mapping filename (basename) to list of signature strings.
        Files with no public symbols or non-.py files are omitted.
    """
    result: dict[str, list[str]] = {}

    for fpath in file_paths:
        if not fpath.endswith(".py"):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=fpath)
        except Exception:
            continue

        signatures: list[str] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _is_private(node.name):
                    continue
                signatures.append(_format_signature(node))

            elif isinstance(node, ast.ClassDef):
                if _is_private(node.name):
                    continue
                class_sigs: list[str] = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if _is_private(item.name):
                            continue
                        class_sigs.append("  " + _format_signature(item))
                if class_sigs:
                    signatures.append(f"class {node.name}:")
                    signatures.extend(class_sigs)
                else:
                    signatures.append(f"class {node.name}")

        if signatures:
            basename = os.path.basename(fpath)
            result[basename] = signatures

    return result


class GoalTracker:
    """Tracks a decomposed goal as a list of subtasks.

    Wraps the existing brain loop — each subtask gets its own
    ``_brain_loop()`` iteration.  The task lifecycle (observer, cost
    tracking, cancel event, Telegram thread) stays shared across all
    subtasks.
    """

    def __init__(
        self,
        original_goal: str,
        subtasks: list[Subtask],
        current_index: int = 0,
        min_subtasks: int = 0,
        max_subtasks: int | None = None,
        continuation_count: int = 0,
    ) -> None:
        self.original_goal = original_goal
        self.subtasks = subtasks
        self.current_index = current_index
        self.min_subtasks = min_subtasks
        self.max_subtasks = max_subtasks
        self.continuation_count = continuation_count

    # -- iteration helpers --------------------------------------------------

    def next_subtask(self) -> Subtask | None:
        """Return the next pending subtask, or None if all done/failed."""
        for i in range(self.current_index, len(self.subtasks)):
            if self.subtasks[i].status == "pending":
                self.current_index = i
                return self.subtasks[i]
        return None

    def mark_running(self, subtask: Subtask) -> None:
        """Set a subtask to running."""
        subtask.status = "running"

    def mark_done(self, summary: str) -> None:
        """Mark the current subtask as done with a result summary."""
        st = self.subtasks[self.current_index]
        st.status = "done"
        st.result_summary = summary
        if st.files_written:
            st.interfaces = extract_interfaces(st.files_written)
        self.current_index += 1

    def mark_failed(self, reason: str) -> None:
        """Mark the current subtask as failed with a reason."""
        st = self.subtasks[self.current_index]
        st.status = "failed"
        st.result_summary = reason

    def all_done(self) -> bool:
        """Return True when every subtask has status 'done'.

        Safety net: if min_subtasks was set during initial decomposition
        and fewer subtasks have completed than that floor, return False.
        This catches cases where a continuation plan underestimates the
        remaining work and produces too few replacement subtasks.
        """
        if not all(s.status == "done" for s in self.subtasks):
            return False
        # If a continuation plan replaced the remaining subtasks with fewer
        # than min_subtasks requires, the tracker should not declare victory.
        done_count = sum(1 for s in self.subtasks if s.status == "done")
        if self.min_subtasks > 0 and done_count < self.min_subtasks:
            return False
        return True

    def completed_context(self) -> str:
        """Rich summary of already-done subtasks for goal injection.

        Includes per-subtask file lists and tiered interface blocks.
        Recent subtasks (within SUBTASK_INTERFACE_RECENT_WINDOW) get full
        AST-extracted signatures.  Older subtasks get file-list only.
        Recent subtasks also get result_summary for tactical context.
        """
        done_subtasks = [s for s in self.subtasks if s.status == "done"]
        if not done_subtasks:
            return "(no prior subtasks completed)"

        # Determine the recent window boundary
        recent_window = SUBTASK_INTERFACE_RECENT_WINDOW
        recent_start = max(0, len(done_subtasks) - recent_window)

        parts: list[str] = []
        all_files: list[str] = []

        for idx, s in enumerate(done_subtasks):
            is_recent = idx >= recent_start
            entry = f"Subtask {s.n} (done): {s.description}"

            if s.files_written:
                display_files = s.files_written[:15]
                file_list = ", ".join(display_files)
                if len(s.files_written) > 15:
                    file_list += f" ... (+{len(s.files_written) - 15} more)"
                entry += f"\n  Files: {file_list}"
                all_files.extend(s.files_written)
            else:
                entry += "\n  (no files written)"

            # Include result_summary for recent subtasks so the next
            # subtask has tactical context (what was done, not just
            # that it was "done").
            if is_recent and s.result_summary:
                entry += f"\n  Summary: {s.result_summary[:200]}"

            # Full interface block only for recent subtasks with interfaces
            if is_recent and s.interfaces:
                iface_lines: list[str] = []
                for fname, sigs in s.interfaces.items():
                    iface_lines.append(f"    {fname}:")
                    for sig in sigs:
                        iface_lines.append(f"      {sig}")
                entry += "\n  Interfaces:\n" + "\n".join(iface_lines)

            parts.append(entry)

        ctx = "\n".join(parts)
        if all_files:
            ctx += (
                "\n\n⚠ EXISTING FILES (DO NOT recreate or overwrite from scratch "
                "— only MODIFY if needed):\n  "
                + "\n  ".join(sorted(set(all_files)))
            )
        return ctx

    def final_summary(self) -> str:
        """Join all subtask result_summaries into one string."""
        parts: list[str] = []
        for s in self.subtasks:
            label = "✓" if s.status == "done" else "✗"
            parts.append(f"[{label}] Subtask {s.n}: {s.result_summary}")
        return "\n".join(parts)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise for checkpoint storage."""
        return {
            "original_goal": self.original_goal,
            "current_index": self.current_index,
            "min_subtasks": self.min_subtasks,
            "max_subtasks": self.max_subtasks,
            "continuation_count": self.continuation_count,
            "subtasks": [
                {
                    "n": s.n,
                    "description": s.description,
                    "status": s.status,
                    "result_summary": s.result_summary,
                    "model_tier": s.model_tier,
                    "files_written": s.files_written or [],
                    "interfaces": s.interfaces or {},
                }
                for s in self.subtasks
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "GoalTracker":
        """Reconstruct from checkpoint dict."""
        subtasks = [
            Subtask(
                n=s["n"],
                description=s["description"],
                status=s.get("status", "pending"),
                result_summary=s.get("result_summary", ""),
                model_tier=s.get("model_tier", "tier_0"),
                files_written=s.get("files_written", []),
                interfaces=s.get("interfaces", {}),
            )
            for s in d["subtasks"]
        ]
        return GoalTracker(
            original_goal=d["original_goal"],
            subtasks=subtasks,
            current_index=d.get("current_index", 0),
            min_subtasks=d.get("min_subtasks", 0),
            max_subtasks=d.get("max_subtasks"),
            continuation_count=d.get("continuation_count", 0),
        )


# =========================================================================
# decompose() — LLM-powered goal → subtask list
# =========================================================================


def _build_roster_block() -> str:
    """Build the specialist model roster description for the decompose prompt."""
    lines = []
    for tier_name, info in MODEL_ROSTER.items():
        lines.append(f"  {tier_name}: {info['description']}")
    return "\n".join(lines)


# Injected into the decompose system prompt so the planning model understands
# what it is planning FOR — not just an abstract task description.
_GENIE_CONTEXT_BLOCK = f"""\
You are a PLANNER for Genie — an autonomous desktop automation agent running on
Ubuntu GNOME (Xorg), Python 3.12.  The user's workspace is ~/genie_workspace/.

Your job is to output a JSON PLAN (subtasks list).  You are NOT the executor.
Do NOT output shell commands or action objects.  Output ONLY the subtasks JSON.

How Genie executes a subtask:
  Each subtask is handed to an executor that runs a ReAct loop — it writes files,
  runs shell commands, reads output, and fixes errors autonomously.
  The loop is HARD-CAPPED at {config.MAX_ITERATIONS_PER_SUBTASK} executor steps.
  If a subtask is scoped too broadly, the executor hits the cap mid-task and FAILS.

Subtask granularity guidance:
  A well-scoped subtask implements ONE concern: one algorithm, one class's public
  interface, one feature boundary, or one integration wire-up.
  A subtask that implements multiple unrelated modules is TOO BROAD — split it.
  Rule of thumb: if you can name two distinct responsibilities in a subtask
  description, it should be two subtasks.
  The hard cap is 50 subtasks. Use it when the project warrants it:
    - Single-file script: 2–4 subtasks
    - 5-module project: 8–12 subtasks
    - 15-module project: 20–35 subtasks
    - 30+ module platform: 40–50 subtasks
"""


_DECOMPOSE_SYSTEM_PROMPT = (
    _GENIE_CONTEXT_BLOCK
    + "\n"
    + "Your job: given a user's goal, produce a sequence of subtasks that\n"
    + "Genie's executor can accomplish within the iteration budget above.\n"
    + "Return ONLY a JSON object — no markdown, no explanation:\n"
    + '{"subtasks": [{"n": 1, "description": "...", "model_tier": "tier_0"}, ...]}\n'
    + "\n"
    + "=== RULES ===\n"
    + "0. SCALE SUBTASK COUNT TO PROJECT SIZE:\n"
    + "   - Small task (1–3 output files, single concern): 2–5 subtasks\n"
    + "   - Medium project (4–10 files, a few distinct concerns): 6–12 subtasks\n"
    + "   - Large project (10+ files, multiple layers/modules): 15–30 subtasks\n"
    + f"   The hard cap ({MAX_GOALTRACKER_SUBTASKS}) exists as a safety ceiling, not a target.\n"
    + "   Always end with a verification subtask.\n"
    + "   A single subtask is NEVER acceptable regardless of goal size.\n"
    + "1. SEQUENTIAL BUILD-UP: subtasks run in order, each building on the previous.\n"
    + "2. ONE CONCERN PER SUBTASK: a concern is a single algorithm, a single module's\n"
    + "   public interface, or a single feature boundary.  Multiple subtasks may touch\n"
    + "   the same file if they implement genuinely distinct concerns within it\n"
    + "   (e.g. 'implement topological sort in dag.py' and 'implement cycle detection\n"
    + "   in dag.py' are two subtasks, both touching dag.py).\n"
    + "3. DESCRIPTIONS are plain-English work orders naming specific files and what\n"
    + "   to implement.  Never paste the user's goal text as a description.\n"
    + "4. FIRST SUBTASK — create the project skeleton ONLY: directory structure,\n"
    + "   pyproject.toml / setup files, __init__.py files, empty module stubs.\n"
    + "   No logic, no implementations in the first subtask.  Logic begins in subtask 2.\n"
    + "   Reason: a skeleton subtask that fails is cheap to retry.\n"
    + "4b. INTERFACE-FIRST for large from-scratch builds (10+ modules):\n"
    + "   After the skeleton subtask, create a 'contracts' subtask that writes a\n"
    + "   contracts.md (or similar) file documenting the public API signatures\n"
    + "   (class names, method signatures with argument types) for EVERY module.\n"
    + "   All subsequent implementation subtasks MUST reference contracts.md to\n"
    + "   ensure cross-module calls use the correct method signatures.\n"
    + "   This dramatically reduces integration bugs between independently-coded\n"
    + "   modules. Skip this rule for small projects (<10 files) or debugging tasks.\n"
    + "5. LAST SUBTASK — execute the deliverable and verify output:\n"
    + "   - CLI tool: run it with several subcommands, check stdout\n"
    + "   - Script: execute it and assert correct output\n"
    + "   - Web app: start server, hit endpoints, check responses\n"
    + "   NEVER just read a file or list a directory — that does not verify behaviour.\n"
    + "   VERIFICATION CAP: combine ALL end-to-end verification steps into ONE\n"
    + "   final subtask. Do NOT split verification across multiple subtasks.\n"
    + "   Include test fixture creation (writing test data files, sample configs,\n"
    + "   etc.) inside this same verification subtask — do NOT make a separate\n"
    + "   'prepare test environment' subtask.\n"
    + "   This applies to EVERY goal type — from-scratch builds, debugging, etc.\n"
    + "6. Assign \"model_tier\" per subtask from the specialist roster below.\n"
    + "7. DEBUGGING EXCEPTION: If the goal is to fix bugs in EXISTING code (goal\n"
    + "   says 'fix', 'patch', 'debug', or 'Do NOT delete') — override Rule 4.\n"
    + "   Do NOT create a 'read and understand' or 'skeleton' first subtask.\n"
    + "   Instead: FIRST subtask fixes Bug 1 (name the specific files to patch).\n"
    + "   ONE BUG PER SUBTASK — never group multiple independent bugs into one\n"
    + "   subtask. Each subtask must state the exact file(s) it modifies.\n"
    + "   After fixing each bug, include a quick smoke-test (e.g. python -c\n"
    + "   'import module' or python -m py_compile file.py) within the SAME\n"
    + "   subtask to confirm the fix before moving on.\n"
    + "8. SINGLE-FILE RULE: NEVER split implementing a single file across\n"
    + "   multiple subtasks (e.g. 'begin cli.py' + 'complete cli.py'). If a file\n"
    + "   needs new code, allocate ONE subtask for the COMPLETE implementation.\n"
    + "   This applies to EVERY goal type — from-scratch builds and debugging alike.\n"
    + "\n"
    + "=== EXAMPLE OUTPUT (URL shortener microservice, 6 subtasks) ===\n"
    + '{"subtasks": [\n'
    + '  {"n": 1, "model_tier": "tier_1", "description": "Create url_shortener/ project skeleton (overwrite any existing files as needed): pyproject.toml (with fastapi, uvicorn, pytest dependencies), url_shortener/__init__.py, url_shortener/main.py (empty), url_shortener/models.py (empty), url_shortener/storage.py (empty), url_shortener/utils.py (empty), tests/__init__.py, tests/test_api.py (empty)."},\n'
    + '  {"n": 2, "model_tier": "tier_2", "description": "Implement url_shortener/models.py: Pydantic models ShortenRequest(url: str, custom_alias: str | None) and ShortenResponse(short_url: str, original_url: str, created_at: datetime). Implement url_shortener/storage.py: InMemoryStore class with create(url, alias) -> str, resolve(short_code) -> str | None, list_all() -> list[dict]."},\n'
    + '  {"n": 3, "model_tier": "tier_2", "description": "Implement url_shortener/utils.py: generate_short_code(length=6) -> str using base62 encoding. Implement url_shortener/main.py: FastAPI app with POST /shorten, GET /{code} (redirect), GET /api/links (list all). Wire models, storage, and utils together."},\n'
    + '  {"n": 4, "model_tier": "tier_1", "description": "Write tests/test_api.py: pytest tests using FastAPI TestClient — test shorten with random alias, shorten with custom alias, resolve valid code (expect 307), resolve invalid code (expect 404), list links. At least 5 test functions."},\n'
    + '  {"n": 5, "model_tier": "tier_1", "description": "Install the package in editable mode (pip install -e .) and run pytest. Fix any import errors or test failures until all tests pass with exit code 0."},\n'
    + '  {"n": 6, "model_tier": "tier_0", "description": "Start the server with uvicorn in background, then exercise the API end-to-end: POST /shorten two URLs, GET the short codes to confirm redirects, GET /api/links to confirm both entries. Stop server. Confirm all curl commands return expected HTTP status codes."}\n'
    + ']}\n'
    + "\n"
    + "=== EXAMPLE OUTPUT (async job processor, 10 modules, 15 subtasks) ===\n"
    + '{"subtasks": [\n'
    + '  {"n": 1, "model_tier": "tier_1", "description": "Create job_processor/ skeleton (overwrite any existing files as needed): pyproject.toml (fastapi, uvicorn, aiosqlite, click, pydantic, pytest deps), job_processor/__init__.py, job_processor/config.py (empty), job_processor/models.py (empty), job_processor/database.py (empty), job_processor/queue.py (empty), job_processor/executor.py (empty), job_processor/scheduler.py (empty), job_processor/retry.py (empty), job_processor/api.py (empty), job_processor/cli.py (empty), job_processor/__main__.py (empty), tests/__init__.py, tests/test_queue.py (empty), tests/test_api.py (empty)."},\n'
    + '  {"n": 2, "model_tier": "tier_2", "description": "Implement job_processor/config.py: Settings dataclass (DB_PATH, MAX_CONCURRENT=4, DEFAULT_RETRY=3, DEFAULT_TIMEOUT=60, API_PORT=8000). Implement job_processor/models.py: Pydantic schemas JobDefinition, JobRun, JobState enum (queued/running/done/failed/timed_out)."},\n'
    + '  {"n": 3, "model_tier": "tier_2", "description": "Implement job_processor/database.py: async init_db() creating jobs and job_runs tables in SQLite WAL mode via aiosqlite. Functions: insert_job, get_job, list_jobs, insert_run, update_run, list_runs_for_job."},\n'
    + '  {"n": 4, "model_tier": "tier_2", "description": "Implement job_processor/queue.py: asyncio priority queue backed by the jobs table. Functions: enqueue(job_def, priority), dequeue() -> JobRun | None, mark_done(run_id), mark_failed(run_id, reason). Persist queue state so it survives restart."},\n'
    + '  {"n": 5, "model_tier": "tier_2", "description": "Implement job_processor/retry.py: RetryPolicy dataclass (max_attempts, backoff_base, jitter, retry_on_exit_codes). Function should_retry(policy, attempt, exit_code) -> bool. Function backoff_delay(policy, attempt) -> float using exponential backoff with ±20%% jitter."},\n'
    + '  {"n": 6, "model_tier": "tier_2", "description": "Implement job_processor/executor.py: async run_job(run: JobRun) -> int. Uses asyncio.create_subprocess_shell. Enforces timeout with asyncio.wait_for — on timeout kill process, return exit_code=-1. Captures stdout/stderr and stores in database log_chunks table."},\n'
    + '  {"n": 7, "model_tier": "tier_2", "description": "Implement job_processor/scheduler.py: SchedulerLoop class. Runs asyncio loop: dequeue job, acquire asyncio.Semaphore(MAX_CONCURRENT), run via executor, pass exit_code to retry.py, re-enqueue or mark final state. Exposes start() and stop() coroutines."},\n'
    + '  {"n": 8, "model_tier": "tier_2", "description": "Implement job_processor/api.py: FastAPI app on port 8000. Endpoints: GET /jobs (list, filter by state), GET /jobs/{id} (detail), GET /jobs/{id}/runs (history), GET /jobs/{id}/logs (stdout/stderr), POST /jobs/{id}/trigger (force-queue), GET /stats (total_runs, pass_rate, avg_duration), GET /health."},\n'
    + '  {"n": 9, "model_tier": "tier_1", "description": "Implement job_processor/cli.py: click CLI with commands: status (list all jobs), trigger <name> (force-queue), logs <name> (tail last run), runs <name> (run history), stats (aggregate stats), cancel <name>."},\n'
    + '  {"n": 10, "model_tier": "tier_1", "description": "Implement job_processor/__main__.py: asyncio entry point. Starts uvicorn (API), scheduler loop, handles SIGTERM/SIGINT for clean shutdown. Run with python -m job_processor."},\n'
    + '  {"n": 11, "model_tier": "tier_1", "description": "Write tests/test_queue.py: pytest tests for queue enqueue/dequeue order, priority ordering, persistence across restart. At least 4 test functions."},\n'
    + '  {"n": 12, "model_tier": "tier_1", "description": "Write tests/test_api.py: pytest tests for all FastAPI endpoints using TestClient. Test list jobs, trigger job, get run history, health check, stats. At least 6 test functions."},\n'
    + '  {"n": 13, "model_tier": "tier_1", "description": "Install package with pip install -e . and run pytest. Fix all import errors and test failures until pytest exits 0."},\n'
    + '  {"n": 14, "model_tier": "tier_0", "description": "Start daemon with python -m job_processor in background. Trigger 2 jobs via CLI. Wait 5s. Run CLI status — confirm jobs appear. Run CLI logs for a completed job — confirm non-empty output. Hit GET /health, GET /jobs, GET /stats via curl — confirm all return 2xx JSON. Then: stop daemon, define one job with exit 1 and retry.max_attempts=3, restart daemon, trigger it, wait 15s, run CLI runs — confirm exactly 3 run entries all with exit_code=1 and final state=failed."}\n'
    + ']}\n'
    + "\n"
    + "SPECIALIST MODEL ROSTER:\n"
    + _build_roster_block()
)


# ---------------------------------------------------------------------------
# _compute_min_subtasks — Fix 2 (A+B): heuristic minimum subtask floor
# ---------------------------------------------------------------------------

def _compute_min_subtasks(goal: str, complexity: str = "medium") -> tuple[int, int | None]:
    """Compute a minimum subtask count from complexity tier + goal size heuristics.

    Strategy A: map the clarify-phase complexity tier to a floor.
    Strategy B: goal size heuristics — character count and enumeration density.
    Returns max(A, B) so both signals raise the floor independently.

    For debug/patch/fix tasks, the floor is capped low because each bug
    only needs 1-2 subtasks, not an entire project build-up.
    """
    # Detect debug/patch task
    goal_lower = goal.lower()
    _DEBUG_MARKERS = ("fix ", "patch ", "debug ", "do not delete", "repair ", "bugfix")
    is_debug = any(m in goal_lower for m in _DEBUG_MARKERS)

    # A: complexity-tier floor (from clarify phase)
    complexity_floor = {
        "simple":  2,
        "medium":  6,
        "large":  15,
        "massive": 25,
    }.get(complexity, 6)

    # B: goal size heuristics
    goal_len = len(goal)
    # Count enumerated list items: lines beginning with digit+dot/paren, dash, asterisk, bullet
    enum_items = len(re.findall(r'(?m)^\s*(?:\d+[.):]|[-*\u2022])\s+\S', goal))
    # Count explicit .py file references (proxy for module count)
    py_refs = len(re.findall(r'\b\w+\.py\b', goal))

    heuristic_floor = 2
    if goal_len > 4000:
        heuristic_floor = max(heuristic_floor, 25)
    elif goal_len > 2000:
        heuristic_floor = max(heuristic_floor, 18)
    elif goal_len > 800:
        heuristic_floor = max(heuristic_floor, 10)
    elif goal_len > 300:
        heuristic_floor = max(heuristic_floor, 5)

    if enum_items > 25:
        heuristic_floor = max(heuristic_floor, 25)
    elif enum_items > 15:
        heuristic_floor = max(heuristic_floor, 18)
    elif enum_items > 8:
        heuristic_floor = max(heuristic_floor, 12)
    elif enum_items > 4:
        heuristic_floor = max(heuristic_floor, 7)

    if py_refs >= 10:
        heuristic_floor = max(heuristic_floor, py_refs)
    elif py_refs >= 5:
        heuristic_floor = max(heuristic_floor, 12)

    result = max(complexity_floor, heuristic_floor)
    # Debug/patch tasks: cap the floor low — each bug = 1-2 subtasks + 1 verify.
    # The generic heuristic over-counts because long prompts with many .py refs
    # describe the codebase, not modules to CREATE.
    max_subtasks: int | None = None
    if is_debug:
        # Count distinct bugs mentioned (e.g. "Bug 1", "Bug 2", etc.)
        bug_count = max(1, len(re.findall(r'(?i)\bbug\s*\d', goal)))
        debug_floor = bug_count * 2 + 1  # 2 subtasks per bug + 1 verify
        result = min(result, max(debug_floor, 4))  # never below 4 for debug tasks
        max_subtasks = max(debug_floor + 2, 6)  # hard cap: a little slack above floor
    print(
        f"  [goaltracker] _compute_min_subtasks: complexity={complexity} "
        f"floor={complexity_floor}, heuristic floor={heuristic_floor} "
        f"(len={goal_len}, enums={enum_items}, py_refs={py_refs}) "
        f"is_debug={is_debug} → min={result} max={max_subtasks}",
        flush=True,
    )
    return result, max_subtasks


_VERIFICATION_RE = re.compile(
    r'(?:^|\b)(?:verif|test\s|check\s|assert\s|confirm\s|validate\s|ensure\s'
    r'|step\s*\d|prepare\s+(?:verification|test)|run\s+(?:all|end.to.end|final)'
    r'|final\s+(?:audit|check|test|verif))',
    re.IGNORECASE,
)


def _merge_verification_tail(
    goal: str, subtasks: list["Subtask"]
) -> list["Subtask"]:
    """Merge multiple verification-only tail subtasks into one.

    Scans backwards from the last subtask.  Any trailing subtask whose
    description matches a verification pattern (Verify, Test, Step N,
    Prepare verification, etc.) is considered verification-only.
    If 2+ consecutive verification subtasks sit at the tail, merge them
    into a single subtask so the agent runs verification in one pass.
    Applies to ALL goal types (from-scratch builds and debugging alike).
    """
    if len(subtasks) < 3:
        return subtasks

    # Scan backwards: find the contiguous tail of verification subtasks
    first_verify_idx = len(subtasks)  # exclusive upper bound
    for i in range(len(subtasks) - 1, -1, -1):
        if _VERIFICATION_RE.search(subtasks[i].description):
            first_verify_idx = i
        else:
            break

    tail = subtasks[first_verify_idx:]
    if len(tail) <= 1:
        return subtasks

    merged_desc = " Then: ".join(st.description for st in tail)
    merged_tier = tail[0].model_tier
    result = subtasks[:first_verify_idx] + [
        Subtask(n=first_verify_idx + 1, description=merged_desc, model_tier=merged_tier)
    ]
    for i, st in enumerate(result):
        st.n = i + 1
    print(
        f"  [goaltracker] Merged {len(tail)} verification subtasks into 1 "
        f"(plan now {len(result)} subtasks)",
        flush=True,
    )
    return result


def _validate_debugging_plan(
    goal: str,
    subtasks: list["Subtask"],
    messages: list[dict],
    response_text: str,
    model: str,
    orch: "GenieOrchestrator",
) -> list["Subtask"]:
    """Post-plan validator for debugging tasks.

    Detects and forces re-decomposition when the LLM:
    1. Creates a read-only/exploration first subtask (wastes iterations before
       any file is touched).
    2. Bundles multiple distinct bugs into a single subtask (causes cap hits).

    Returns the original or corrected subtask list.
    """
    # Only applies when the goal contains numbered bug sections
    bug_sections = re.findall(r'BUG\s+\d+', goal, re.IGNORECASE)
    if len(set(bug_sections)) < 2:
        return subtasks  # not a multi-bug debugging goal

    _EXPLORE_RE = re.compile(
        r'\b(read|list|inspect|identify|understand|explore|examine|review)\b',
        re.IGNORECASE,
    )
    _ACTION_RE = re.compile(
        r'\b(fix|write|patch|implement|add|modify|create|install|update|change|edit|replace)\b',
        re.IGNORECASE,
    )

    issues: list[str] = []

    # Check 1: first subtask must modify at least one file
    if subtasks:
        first_desc = subtasks[0].description
        if _EXPLORE_RE.search(first_desc) and not _ACTION_RE.search(first_desc):
            issues.append(
                "Subtask 1 is a pure read/exploration step — it modifies nothing. "
                "For debugging, the FIRST subtask must patch a file. "
                "The bug descriptions already tell you exactly which files to change."
            )

    # Check 2: no subtask may reference more than one distinct bug
    for st in subtasks:
        found_bugs = set(
            b.upper() for b in re.findall(r'BUG\s+\d+', st.description, re.IGNORECASE)
        )
        if len(found_bugs) > 1:
            issues.append(
                f"Subtask {st.n} bundles {sorted(found_bugs)} — "
                "each bug must be its own subtask."
            )

    # Check 3: merge verification-only tail into at most one subtask.
    subtasks = _merge_verification_tail(goal, subtasks)

    if not issues:
        return subtasks

    issue_lines = "\n".join(f"  - {iss}" for iss in issues)
    print(
        f"  [goaltracker] Debugging plan validation failed:\n{issue_lines}\n"
        "  Retrying decompose with corrections…",
        flush=True,
    )

    constraint_msg = (
        "Your previous plan has these problems:\n"
        f"{issue_lines}\n\n"
        "REQUIRED CORRECTIONS:\n"
        "1. Never start with a read-only subtask. Patch files immediately — "
        "the bug descriptions name the files.\n"
        "2. Each 'BUG N' must be a separate subtask. Never put two bugs in one.\n"
        "Re-decompose now applying these corrections."
    )
    retry_messages = messages + [
        {"role": "assistant", "content": response_text},
        {"role": "user", "content": constraint_msg},
    ]
    try:
        retry_resp = orch._llm_call(retry_messages, model=model, normalize=False)
        retry_parsed = _parse_decompose_json(retry_resp)
        if (
            retry_parsed is not None
            and isinstance(retry_parsed.get("subtasks"), list)
            and retry_parsed["subtasks"]
        ):
            raw = retry_parsed["subtasks"][:MAX_GOALTRACKER_SUBTASKS]
            corrected: list[Subtask] = []
            for i, s in enumerate(raw):
                if not isinstance(s, dict):
                    continue
                desc = s.get("description", "")
                if not desc:
                    continue
                tier = s.get("model_tier", "tier_0")
                if tier not in MODEL_ROSTER:
                    tier = "tier_0"
                corrected.append(Subtask(n=i + 1, description=desc, model_tier=tier))
            if corrected:
                # Re-run the same checks on the corrected plan
                retry_issues: list[str] = []
                first_desc = corrected[0].description if corrected else ""
                if _EXPLORE_RE.search(first_desc) and not _ACTION_RE.search(first_desc):
                    retry_issues.append("ST1 is still read-only")
                for st in corrected:
                    found = set(
                        b.upper()
                        for b in re.findall(r"BUG\s+\d+", st.description, re.IGNORECASE)
                    )
                    if len(found) > 1:
                        retry_issues.append(f"ST{st.n} still bundles {sorted(found)}")
                if not retry_issues:
                    print(
                        f"  [goaltracker] Validation retry => {len(corrected)} subtask(s) (PASS)",
                        flush=True,
                    )
                    return corrected
                print(
                    f"  [goaltracker] Retry still failed validation: {retry_issues} — using original plan",
                    flush=True,
                )
    except Exception as _exc:
        print(f"  [goaltracker] Validation retry failed: {_exc}", flush=True)

    return subtasks


def decompose(orch: "GenieOrchestrator") -> GoalTracker:
    """Decompose ``orch._goal`` into subtasks via an LLM call.

    On parse failure, returns a single-subtask GoalTracker wrapping the
    original goal — graceful degradation so the brain loop runs once as
    before.
    """
    _plan_model    = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[0]
    _plan_fallback = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[1]

    # Build workspace snapshot so the LLM knows what already exists
    workspace_snapshot = ""
    try:
        _ws = os.path.expanduser("~/genie_workspace")
        _entries = []
        for root, dirs, files in os.walk(_ws):
            # Skip hidden dirs and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            rel = os.path.relpath(root, _ws)
            for f in files:
                _entries.append(os.path.join(rel, f) if rel != "." else f)
            if len(_entries) > 60:
                break
        if _entries:
            workspace_snapshot = (
                "\n\nEXISTING WORKSPACE (~/genie_workspace/):\n  "
                + "\n  ".join(_entries[:60])
                + ("\n  ... (truncated)" if len(_entries) > 60 else "")
                + "\n\nWARNING \u2014 existing files detected in the workspace:\n"
                "- FRESH BUILD goal (build/create/write from scratch): your FIRST subtask MUST "
                "'rm -rf' the old project directory before creating anything new.\n"
                "- FIX/MODIFY/IMPROVE goal (fix/add/update/debug/improve): DO NOT delete any files. "
                "These files ARE the project. Work on top of them."
            )
    except OSError:
        pass

    min_subtasks, max_subtasks = _compute_min_subtasks(
        orch._goal, getattr(orch, "_ask_user_complexity", "medium")
    )
    if max_subtasks:
        budget_hint = (
            f"\n\nEXECUTOR BUDGET: {config.MAX_ITERATIONS_PER_SUBTASK} iterations per subtask. "
            "Split the work so each subtask comfortably fits within that budget. "
            f"This goal requires EXACTLY {min_subtasks} subtasks — no more, no fewer."
        )
    else:
        budget_hint = (
            f"\n\nEXECUTOR BUDGET: {config.MAX_ITERATIONS_PER_SUBTASK} iterations per subtask. "
            "Split the work so each subtask comfortably fits within that budget. "
            f"This goal requires AT LEAST {min_subtasks} subtasks — do not produce fewer."
        )

    messages = [
        {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user",   "content": f"GOAL: {orch._goal}{workspace_snapshot}{budget_hint}"},
    ]

    response_text = None
    model = _plan_model
    print(f"  [goaltracker] Decomposing goal via {model}… (min_subtasks={min_subtasks}, max_subtasks={max_subtasks})", flush=True)

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            response_text = orch._llm_call(messages, model=model, normalize=False)
            break
        except ResponseTruncatedError as rte:
            response_text = rte.partial_content
            break
        except (httpx.TimeoutException, httpx.RequestError):
            if attempt == MAX_LLM_RETRIES:
                print("  [goaltracker] LLM network failure — single-subtask fallback", flush=True)
                return _single_subtask_fallback(orch._goal)
            time.sleep(RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in LLM_SERVICE_ERROR_CODES:
                model = _plan_fallback
                try:
                    response_text = orch._llm_call(messages, model=model, normalize=False)
                except Exception:
                    return _single_subtask_fallback(orch._goal)
                break
            return _single_subtask_fallback(orch._goal)

    if not response_text:
        return _single_subtask_fallback(orch._goal)

    print(f"  [goaltracker] Raw decompose response ({len(response_text)} chars):\n{response_text[:1000]}", flush=True)

    parsed = _parse_decompose_json(response_text)
    if parsed is None or not isinstance(parsed.get("subtasks"), list) or not parsed["subtasks"]:
        print("  [goaltracker] Parse failure — single-subtask fallback", flush=True)
        return _single_subtask_fallback(orch._goal)

    _subtask_cap = max_subtasks if max_subtasks else MAX_GOALTRACKER_SUBTASKS
    raw_subtasks = parsed["subtasks"][:_subtask_cap]
    subtasks = []
    for i, s in enumerate(raw_subtasks):
        if not isinstance(s, dict):
            continue
        desc = s.get("description", "")
        if not desc:
            continue
        tier = s.get("model_tier", "tier_0")
        # Validate tier — fall back to tier_0 if LLM hallucinated a bad tier
        if tier not in MODEL_ROSTER:
            tier = "tier_0"
        subtasks.append(Subtask(n=i + 1, description=desc, model_tier=tier))

    if not subtasks:
        return _single_subtask_fallback(orch._goal)

    # -- Enforce minimum 2 subtasks ------------------------------------------
    # Safety net: if the model still collapsed to 1 subtask despite grounding,
    # auto-append a Genie-aware verification subtask.
    if len(subtasks) == 1:
        print("  [goaltracker] Only 1 subtask returned — auto-appending verify step", flush=True)
        subtasks.append(Subtask(
            n=2,
            description=(
                "Verify the deliverable by executing it with run_command — do NOT just read_file. "
                "Run every command or entry point described in the original goal, inspect stdout/stderr, "
                "and confirm correct output. If a run_command returns a non-zero exit_code, "
                "use write_file to fix the code and re-run until all commands succeed. "
                "Only call done when every verification run_command exits 0."
            ),
            model_tier="tier_0",
        ))

    # -- Enforce computed minimum subtask count (Fix 2 A+B) -------------------
    if len(subtasks) < min_subtasks:
        print(
            f"  [goaltracker] Only {len(subtasks)} subtask(s) but min={min_subtasks} "
            f"— retrying decompose with hard constraint",
            flush=True,
        )
        constraint_msg = (
            f"Your previous decomposition produced only {len(subtasks)} subtask(s). "
            f"This goal requires AT LEAST {min_subtasks} subtasks based on its size and complexity. "
            "Re-decompose and produce AT LEAST that many subtasks. "
            "Split each large concern into a separate subtask — "
            "aim for one module or one feature boundary per subtask."
        )
        retry_messages = messages + [
            {"role": "assistant", "content": response_text},
            {"role": "user", "content": constraint_msg},
        ]
        try:
            retry_response = orch._llm_call(retry_messages, model=model, normalize=False)
            retry_parsed = _parse_decompose_json(retry_response)
            if (
                retry_parsed is not None
                and isinstance(retry_parsed.get("subtasks"), list)
                and retry_parsed["subtasks"]
            ):
                raw_retry = retry_parsed["subtasks"][:MAX_GOALTRACKER_SUBTASKS]
                retry_subtasks: list[Subtask] = []
                for i, s in enumerate(raw_retry):
                    if not isinstance(s, dict):
                        continue
                    desc = s.get("description", "")
                    if not desc:
                        continue
                    tier = s.get("model_tier", "tier_0")
                    if tier not in MODEL_ROSTER:
                        tier = "tier_0"
                    retry_subtasks.append(Subtask(n=i + 1, description=desc, model_tier=tier))
                if len(retry_subtasks) > len(subtasks):
                    print(
                        f"  [goaltracker] Retry produced {len(retry_subtasks)} subtask(s)",
                        flush=True,
                    )
                    subtasks = retry_subtasks
        except Exception as _retry_exc:
            print(f"  [goaltracker] Retry decompose failed: {_retry_exc}", flush=True)

    subtasks = _validate_debugging_plan(
        orch._goal, subtasks, messages, response_text, model, orch
    )
    # Enforce VERIFICATION CAP for ALL goal types: merge trailing verify subtasks
    subtasks = _merge_verification_tail(orch._goal, subtasks)
    # Enforce SINGLE-FILE RULE: merge subtasks that touch the same file
    subtasks = _merge_same_file_subtasks(subtasks)
    # Enforce max_subtasks cap after validation (which may have re-decomposed)
    if max_subtasks and len(subtasks) > max_subtasks:
        print(f"  [goaltracker] Trimming {len(subtasks)} subtasks to max={max_subtasks}", flush=True)
        subtasks = subtasks[:max_subtasks]

    tier_summary = ", ".join(f"S{s.n}={s.model_tier}" for s in subtasks)
    print(f"  [goaltracker] Decomposed into {len(subtasks)} subtask(s) [{tier_summary}]", flush=True)
    return GoalTracker(original_goal=orch._goal, subtasks=subtasks, min_subtasks=min_subtasks, max_subtasks=max_subtasks)


def _single_subtask_fallback(goal: str) -> GoalTracker:
    """Fallback: wrap the original goal as a single subtask."""
    return GoalTracker(
        original_goal=goal,
        subtasks=[Subtask(n=1, description=goal)],
    )


def _parse_decompose_json(text: str) -> dict | None:
    """Extract and parse the decompose JSON object."""
    # Strip <think>...</think> blocks (DeepSeek R1 and similar reasoning models
    # emit chain-of-thought before the JSON — remove it first so the JSON extractor
    # doesn't grab content from inside the think block.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None
