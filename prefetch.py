"""Deterministic content pre-fetch and injection. Extracted from orchestrator.py."""
from __future__ import annotations

import os
import re

import actions
import context_builder
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator import GenieOrchestrator


# Regex matching navigation-intent phrases immediately before a URL.
_NAV_INTENT_RE = re.compile(
    r"(?:open|navigate|visit|go to|launch|open in chrome|open in browser)"
    r"\s+(?:the\s+)?(?:url\s+)?(?:at\s+)?https?://",
    re.IGNORECASE,
)
# file:// and localhost URLs should not be pre-fetched.
_SKIP_URL_RE = re.compile(r"^(?:file://|https?://(?:localhost|127\.0\.0\.1)\b)")

# Phrases immediately before a URL that signal it is a *runtime* target
# the script should call, not a documentation source to read.
_RUNTIME_INTENT_RE = re.compile(
    r"(?:"
    r"(?:GET|POST|PUT|DELETE|PATCH|HEAD)\s+request\s+to"
    r"|make\s+a\s+(?:synchronous\s+|async(?:hronous)?\s+)?(?:GET|POST|PUT|DELETE|HTTP)\s+request\s+to"
    r"|sends?\s+a\s+(?:GET|POST|HTTP)\s+request\s+to"
    r"|make\s+a\s+request\s+to"
    r"|(?:call|hit|query|ping)\s+the\s+(?:URL|endpoint|API)"
    r"|(?:request\s+to|fetch\s+from|send\s+(?:a\s+)?request\s+to)"
    r")\s+['\"]?https?://",
    re.IGNORECASE,
)

# Known test/mock/API-echo hostnames that are never documentation sources.
_RUNTIME_API_HOSTS_RE = re.compile(
    r"^https?://(?:"
    r"httpbin\.org"
    r"|postman-echo\.com"
    r"|jsonplaceholder\.typicode\.com"
    r"|reqres\.in"
    r"|httpstat\.us"
    r"|mockbin\.org"
    r"|api\.github\.com"
    r"|api\."          # any subdomain starting with "api."
    r")",
    re.IGNORECASE,
)

# =====================================================================
# Project file pre-fetch for existing-project tasks
# =====================================================================

_PREFETCH_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".json", ".md"}
_PREFETCH_SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", ".mypy_cache", ".pytest_cache", "egg-info"}
_PREFETCH_MAX_BYTES = 120_000  # ~30k tokens — generous but bounded

# ------------------------------------------------------------------
# Green-field draft pre-generation constants
# ------------------------------------------------------------------
_STUB_PATTERN = re.compile(
    r'\b(TODO|FIXME|HACK|XXX)\b'
    r'|#\s*placeholder'
    r'|raise\s+NotImplementedError'
    r'|\bpass\b\s*$',
    re.I | re.M,
)
_IMPL_VERBS = re.compile(
    r'\b(implement|create|write|build|add|develop|code)\b',
    re.I,
)


def prefetch_project_files(orch) -> None:
    """Smart prefetch: load project files into private cache + generate index.

    Phase 1 (runs once at task start):
    - Scans ``orch._original_goal`` for a ``~/genie_workspace/<project>`` path
    - Reads ALL source files into ``_project_file_cache`` (private — NOT rendered)
    - Generates CODEBASE_INDEX.md via ``actions.index_codebase()`` and stores
      it in ``_workspace_cache`` (rendered in every LLM call — ~5KB)

    Phase 2 (``_inject_subtask_files`` runs per-subtask):
    - Extracts filenames from subtask goal
    - Copies ONLY matching files from ``_project_file_cache`` → ``_workspace_cache``
    - Result: LLM sees index (~5KB) + 2-4 relevant files (~10-15KB) instead
      of the entire project (~70KB+)
    """
    goal = getattr(orch, "_original_goal", "") or orch._goal or ""
    m = re.search(r"(?:~/|/home/\w+/)genie_workspace/\S+", goal)
    if not m:
        return

    project_dir = os.path.expanduser(m.group(0).rstrip("/. "))
    if not os.path.isdir(project_dir):
        return

    # Skip if already loaded
    if orch._project_file_cache:
        return

    orch._project_dir = project_dir

    # -- Phase 1a: Read all source files into private cache ----------------
    total_bytes = 0
    loaded = 0
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [
            d for d in dirs
            if d not in _PREFETCH_SKIP_DIRS and not d.startswith(".")
        ]
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _PREFETCH_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue
            if size > 50_000:
                continue
            if total_bytes + size > _PREFETCH_MAX_BYTES:
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            orch._project_file_cache[fpath] = content
            total_bytes += len(content.encode("utf-8", errors="replace"))
            loaded += 1

    if not loaded:
        return

    # -- Phase 1b: Generate CODEBASE_INDEX.md and store in workspace_cache -
    try:
        result = actions.index_codebase(project_dir)
        index_path = result["index_path"]
        with open(index_path, "r", encoding="utf-8", errors="replace") as fh:
            orch._workspace_cache[index_path] = fh.read()
        print(
            f"  [prefetch] Indexed {result['file_count']} files, "
            f"{result['symbol_count']} symbols → CODEBASE_INDEX.md",
            flush=True,
        )
    except Exception as exc:
        print(f"  [prefetch] index_codebase failed: {exc}", flush=True)

    orch._prefetched_content = (
        f"[project-prefetch] {loaded} files cached from {project_dir}"
    )
    print(
        f"  [prefetch] Cached {loaded} source files "
        f"({total_bytes:,} bytes) from {project_dir} (smart mode — "
        f"injected per-subtask via _inject_subtask_files)",
        flush=True,
    )

def inject_subtask_files(orch, subtask_goal: str) -> None:
    """Inject only goal-relevant project files into workspace_cache.

    Called at the start of each subtask.  Parses the subtask goal for
    filename references, then copies ONLY those files from the private
    ``_project_file_cache`` into ``_workspace_cache`` (which gets
    rendered in every LLM call).

    Also re-sets ``_prefetched_content`` (cleared by ``_reset_for_subtask``).
    """
    if not orch._project_file_cache or not orch._project_dir:
        return

    # -- Remove previously injected project files from workspace_cache -----
    for fpath in orch._injected_project_files:
        orch._workspace_cache.pop(fpath, None)
    orch._injected_project_files = set()

    # -- Extract filenames from subtask goal text --------------------------
    # Match patterns like: executor.py, file_watcher.py, dag/models.py,
    # src/utils/helpers.ts, etc.
    file_refs = set(re.findall(
        r"[\w./\-]+\.(?:py|js|ts|jsx|tsx|yaml|yml|toml|json|cfg|ini|md)\b",
        subtask_goal,
    ))

    if not file_refs:
        # No explicit file references — inject nothing beyond the index.
        # The LLM has CODEBASE_INDEX.md and can call read_file as needed.
        orch._prefetched_content = (
            f"[project-prefetch] index only — no files matched in subtask goal"
        )
        return

    # -- Match extracted references against cached files -------------------
    injected = 0
    injected_bytes = 0
    for fpath, content in orch._project_file_cache.items():
        rel = os.path.relpath(fpath, orch._project_dir)
        basename = os.path.basename(fpath)
        # Match if any extracted reference matches the basename or is a
        # suffix of the relative path (e.g. "dag/models.py" matches
        # "dag_scheduler/dag/models.py")
        matched = any(
            basename == ref or rel == ref or rel.endswith("/" + ref)
            for ref in file_refs
        )
        if matched:
            orch._workspace_cache[fpath] = content
            orch._injected_project_files.add(fpath)
            injected += 1
            injected_bytes += len(content.encode("utf-8", errors="replace"))

    orch._prefetched_content = (
        f"[project-prefetch] {injected} files injected for subtask"
    )
    if injected:
        print(
            f"  [smart-prefetch] Injected {injected} files "
            f"({injected_bytes:,} bytes) matching: "
            f"{', '.join(sorted(file_refs)[:8])}",
            flush=True,
        )


def pre_generate_greenfield_draft(
    orch,
    subtask: "GoalTrackerSubtask",  # noqa: F821
) -> None:
    """Detect stub-heavy files in workspace_cache and pre-generate drafts.

    Called at subtask start, AFTER ``_inject_subtask_files``.  If the
    subtask description contains implementation verbs and a referenced
    file in ``_workspace_cache`` is dominated by TODO stubs, we make a
    single focused LLM call to generate a complete implementation draft.

    The draft is written to disk and injected into ``_workspace_cache``
    so the brain model sees a mostly-implemented file and enters
    "review + fix" mode (which it excels at) instead of "generate from
    scratch" mode (which it fails at under long context).
    """
    # Gate: only fire when subtask description sounds like implementation
    if not _IMPL_VERBS.search(subtask.description):
        return

    # Scan workspace_cache for stub-heavy files mentioned in the subtask
    stub_files: list[tuple[str, str]] = []  # (abs_path, content)
    for fpath, content in list(orch._workspace_cache.items()):
        basename = os.path.basename(fpath)
        if basename not in subtask.description:
            continue
        if not fpath.endswith(".py"):
            continue
        # Count stub vs total function defs
        func_defs = re.findall(r'^\s*def ', content, re.M)
        if len(func_defs) < 2:
            continue  # too few functions to be a green-field candidate
        stub_hits = _STUB_PATTERN.findall(content)
        stub_ratio = len(stub_hits) / max(len(func_defs), 1)
        if stub_ratio >= 0.4:
            stub_files.append((fpath, content))

    if not stub_files:
        return

    # Extract the relevant spec section from the original goal
    # for the file(s) we need to implement
    spec_text = orch._original_goal or ""

    for fpath, stub_content in stub_files:
        basename = os.path.basename(fpath)
        print(
            f"  [greenfield] Detected stub-heavy file: {basename} "
            f"— pre-generating implementation draft",
            flush=True,
        )

        prompt = (
            f"You are given a Python source file that contains function stubs "
            f"(TODO comments, `pass` bodies, `raise NotImplementedError`). "
            f"Your job is to rewrite this file with COMPLETE, WORKING "
            f"implementations for every function.\n\n"
            f"RULES:\n"
            f"1. Keep all existing imports, decorators, function signatures, "
            f"and class structure EXACTLY as-is.\n"
            f"2. Replace EVERY stub body with a full implementation.\n"
            f"3. Use the IMPLEMENTATION SPEC below to determine what each "
            f"function should do.\n"
            f"4. Do NOT add TODO, FIXME, or placeholder comments.\n"
            f"5. Output ONLY the complete Python file — no explanation, "
            f"no markdown fences, no commentary.\n\n"
            f"=== IMPLEMENTATION SPEC ===\n{spec_text}\n\n"
            f"=== CURRENT FILE ({basename}) ===\n{stub_content}\n\n"
            f"=== COMPLETE REWRITTEN FILE ==="
        )

        messages = [
            {"role": "system", "content": "You are a senior Python developer. Output only valid Python code."},
            {"role": "user", "content": prompt},
        ]

        # Use tier_2 model for good code quality at reasonable cost
        from config import MODEL_ROSTER
        draft_model = MODEL_ROSTER.get("tier_2", {}).get(
            "model_id", orch._model
        )

        try:
            raw_text, cost = orch._llm.call(
                messages, model=draft_model, max_tokens=4096,
            )
            orch._task_cost_usd += cost
            orch._monthly_cost_usd += cost
        except Exception as exc:
            print(
                f"  [greenfield] LLM call failed for {basename}: {exc}",
                flush=True,
            )
            continue

        # Clean up: strip markdown fences if the model wrapped them
        draft = raw_text.strip()
        if draft.startswith("```"):
            # Remove opening fence
            first_newline = draft.index("\n") if "\n" in draft else len(draft)
            draft = draft[first_newline + 1:]
        if draft.endswith("```"):
            draft = draft[:-3].rstrip()

        # Validate: must be valid Python and must have fewer stubs
        try:
            compile(draft, basename, "exec")
        except SyntaxError:
            print(
                f"  [greenfield] Draft for {basename} has syntax errors — skipping",
                flush=True,
            )
            continue

        new_stub_hits = _STUB_PATTERN.findall(draft)
        old_stub_hits = _STUB_PATTERN.findall(stub_content)
        if len(new_stub_hits) >= len(old_stub_hits):
            print(
                f"  [greenfield] Draft for {basename} still has "
                f"{len(new_stub_hits)} stubs — skipping",
                flush=True,
            )
            continue

        # Write draft to disk and update workspace_cache
        try:
            actions.write_file(fpath, draft)
            orch._workspace_cache[fpath] = draft
            # Also update the project_file_cache so future subtasks see it
            if fpath in orch._project_file_cache:
                orch._project_file_cache[fpath] = draft
            print(
                f"  [greenfield] Wrote draft for {basename} "
                f"({len(draft)} bytes, {len(old_stub_hits)}→"
                f"{len(new_stub_hits)} stubs)",
                flush=True,
            )
        except OSError as exc:
            print(
                f"  [greenfield] Failed to write {basename}: {exc}",
                flush=True,
            )

def prefetch_goal_urls(orch) -> None:
    """Regex-scan goal for https:// URLs, pre-execute fetch_url for each.

    Instead of injecting a text block the LLM can ignore, this method
    executes fetch_url through the full pipeline: the result is recorded
    in ``_history``, ``_last_obs``, and ``_fire_on_update`` — so the LLM
    sees LAST OBSERVATION with the fetched content and a history entry
    showing ``fetch_url`` already succeeded.

    Also sets ``_prefetched_content`` to a truthy flag so that the
    Chrome-block guard in open_app still works.

    Only fetches external http(s) URLs not preceded by navigation-intent
    keywords.  Silent on failure — LLM can still use fetch_url manually.
    """
    goal = orch._goal
    urls = re.findall(r"https?://[^\s\)\'\"<>]+", goal)
    if not urls:
        # Only clear if not already set by _inject_subtask_files (project prefetch)
        if not orch._prefetched_content:
            orch._prefetched_content = ""
        return

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique_urls: list[str] = []
    for u in urls:
        u = u.rstrip(".,;:!?")
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    # Filter out file://, localhost, navigation-intent, and runtime-target URLs
    filtered: list[str] = []
    for u in unique_urls:
        if _SKIP_URL_RE.match(u):
            continue
        # Known test/API servers are never documentation sources
        if _RUNTIME_API_HOSTS_RE.match(u):
            continue
        idx = goal.find(u)
        if idx > 0:
            prefix = goal[max(0, idx - 120):idx]
            if _NAV_INTENT_RE.search(prefix):
                continue
            # Goal says "make a GET request to <url>" → runtime target
            if _RUNTIME_INTENT_RE.search(prefix):
                continue
        filtered.append(u)

    if not filtered:
        # Only clear if not already set by _inject_subtask_files (project prefetch)
        if not orch._prefetched_content:
            orch._prefetched_content = ""
        return

    any_fetched = False
    for url in filtered:
        try:
            content = actions.fetch_url(url, query="")
            if not content or not content.strip():
                continue
        except Exception:
            # Silent failure — LLM can still call fetch_url itself
            continue

        # Cap content to 4000 chars
        if len(content) > 4000:
            content = content[:4000] + "\n... (truncated)"

        # Build synthetic act_dict + obs_entry matching real pipeline
        synth_act = {
            "action": "fetch_url",
            "args": {"url": url, "query": ""},
        }
        synth_obs = {
            "result": "success",
            "observation": {"content": content},
            "error": None,
            "action": "fetch_url",
            "args": {"url": url, "query": ""},
        }

        # Record in history so LLM sees it in compressed history
        orch._history.append(
            context_builder.make_history_entry(synth_act, synth_obs)
        )
        # Set as last observation so first LLM turn sees the content
        orch._last_obs = synth_obs
        # Bump iteration counter so brain loop starts after this
        orch._iteration += 1
        # Fire on_update so test harness trace records "fetch_url"
        orch._fire_on_update(orch._iteration, synth_act, synth_obs)
        any_fetched = True

    # Flag for Chrome-block guard (truthy when content was pre-fetched)
    # Preserve existing value from _inject_subtask_files (project prefetch)
    if any_fetched:
        orch._prefetched_content = "prefetched"
    elif not orch._prefetched_content:
        orch._prefetched_content = ""
    # NOTE: do NOT set _fetch_url_called here — that flag tracks only
    # LLM-initiated fetch_url calls so the write_file guard fires even
    # when a pre-fetch already ran (forces re-fetch rather than memory write).


def extract_api_summary(content: str, path: str) -> str:
    """P3: Extract public class/def signatures from written file content.

    Returns a compact API summary string for scratchpad storage.
    Only processes Python files (.py). Filters out private names.
    """
    if not path.endswith(".py"):
        return ""
    _classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
    _defs = re.findall(
        r'^[ \t]*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)',
        content, re.MULTILINE,
    )
    # Filter out private names (starting with _)
    _public_classes = [c for c in _classes if not c.startswith("_")]
    _public_defs = [
        f"{name}({args})" for name, args in _defs
        if not name.startswith("_")
    ]
    parts = []
    if _public_classes:
        parts.append("Classes: " + ", ".join(_public_classes))
    if _public_defs:
        parts.append("API: " + ", ".join(_public_defs[:12]))  # cap at 12
    return " | ".join(parts) if parts else ""

