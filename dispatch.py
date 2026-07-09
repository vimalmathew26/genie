"""Action dispatch — maps LLM action dicts to actions.py functions. Extracted from orchestrator.py."""
from __future__ import annotations

import os
import re
import urllib.parse

import actions
from config import APP_PROFILES, log
from exceptions import EnvironmentalError, UnrecoverableError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator import GenieOrchestrator


def dispatch(orch, act_dict: dict) -> tuple[dict, str | None, str | None]:
    """Execute action and return (wrapped_result, wid, tier).

    wid and tier are only set for open_app. All other actions return
    (wrapped_result, None, None).
    """
    action = act_dict["action"]
    args = act_dict["args"]
    wid = None
    tier = None

    if action == "open_app":
        app_label = args["app"]
        app_name = app_label.rsplit("_", 1)[0]

        # -- Fix: block any browser app unless the goal explicitly requires
        # opening a browser GUI (e.g. "Open the file in Chrome by navigating
        # to file://...").  Extend to all common browsers, not just Chrome.
        _BROWSER_NAMES = {
            "chrome", "chromium", "firefox", "browser", "edge", "safari",
            "opera", "brave", "vivaldi",
        }
        _is_browser_app = any(b in app_name for b in _BROWSER_NAMES)
        if _is_browser_app and (
            orch._prefetched_content or "chrome" not in orch._goal.lower()
        ):
            _goal_lower_oa = orch._goal.lower()
            _goal_needs_pr = any(
                p in _goal_lower_oa
                for p in ("pull request", "open_pr", "open a pr", "github")
            )
            if orch._prefetched_content:
                _browser_msg = (
                    "Content for the goal URL was already fetched via fetch_url "
                    "earlier in this conversation — scroll back through the "
                    "conversation history to find the fetch_url observation with "
                    "the page content, extract what you need, and write it with "
                    "write_file. Do NOT open a browser."
                )
            elif _goal_needs_pr:
                _browser_msg = (
                    "Do NOT open a browser for GitHub operations. "
                    "Use the open_pr action directly with: "
                    "repo, title, body, head, base. "
                    "No browser or GUI is needed."
                )
            else:
                _browser_msg = (
                    "Opening a browser GUI is not available. "
                    "You MUST use fetch_url to retrieve this information — "
                    "do NOT write from memory or pretrained knowledge."
                )
            raise EnvironmentalError(_browser_msg)

        # Block terminal emulators — the agent should use run_command instead.
        _TERMINAL_APPS = {
            "alacritty", "gnome-terminal", "xterm", "konsole",
            "terminator", "kitty", "tilix", "urxvt", "st", "sakura",
            "terminal", "xfce4-terminal", "mate-terminal",
        }
        if app_name in _TERMINAL_APPS:
            raise EnvironmentalError(
                f"Terminal app '{app_name}' is not available. "
                "Use run_command to execute shell commands directly. "
                "Use write_file to save files. "
                "Do NOT try to open a terminal app."
            )

        raw_wid = orch.registry.open_app(app_name, label=app_label)
        wid = raw_wid
        # Read tier from registry entry (handles unknown apps correctly)
        with orch.registry._registry_lock:
            reg_entry = orch.registry._registry.get(app_label, {})
        tier = reg_entry.get("tier") or APP_PROFILES.get(app_name, {}).get("tier", "atspi")

        # Fix 1 — dismiss Chrome's "Restore pages?" crash-recovery dialog
        # if present. The dialog is Chrome browser UI (not a webpage), so
        # it can only be dismissed via xdotool. Sending Escape to the
        # window is safe when the dialog is absent (just clears URL-bar
        # focus).  We wait 1.5 s first to let Chrome finish rendering.
        if "chrome" in app_name:
            orch._dismiss_chrome_restore_popup(raw_wid)

        return {"status": "ok", "wid": wid}, wid, tier

    if action == "focus_window":
        success = orch.registry.focus_window(args["app"])
        if not success:
            raise EnvironmentalError("focus_window returned False")
        return {"status": "ok"}, None, None

    if action == "click_element":
        # Optional index parameter: 0-based positional selection.
        # When role="link" and name="", index=N navigates to the Nth
        # organic search result via JS (bypasses AX tree on heavy pages).
        index = int(args.get("index", 0))
        raw = orch.element_resolver.click_element(
            app=args["app"], role=args["role"], name=args["name"],
            url_hint=args.get("url_hint"),
            index=index,
        )
        return raw, None, None

    if action == "type_element":
        raw = orch.element_resolver.type_element(
            app=args["app"], role=args["role"], name=args["name"],
            text=args["text"], url_hint=args.get("url_hint"),
        )
        return raw, None, None

    if action == "read_element":
        index = int(args.get("index", 0))
        raw = orch.element_resolver.read_element(
            app=args["app"], role=args["role"], name=args["name"],
            url_hint=args.get("url_hint"),
            index=index,
        )
        return raw, None, None

    if action == "look":
        raw = orch.element_resolver.look(
            app=args["app"],
            question=args.get("question", ""),
        )
        return raw, None, None

    # -- File operations --
    if action == "read_file":
        try:
            raw = actions.read_file(args["path"])
        except (FileNotFoundError, OSError) as exc:
            raise EnvironmentalError(str(exc))
        return {"content": raw}, None, None

    if action == "write_file":
        content = args["content"]
        if "${" in content or "<" in content:
            content = orch._resolve_placeholder_from_history(
                content, orch._history, None,
            )
        # Guard: reject empty writes so the LLM retries with actual data
        if not content or not content.strip():
            raise EnvironmentalError(
                "write_file called with empty content. "
                "Re-read the data you need (read_element / run_command) "
                "and provide the actual text in the content field."
            )
        # Guard: block memory writes on lookup tasks that haven't fetched yet.
        # If the goal is a web-lookup (find/look up/use fetch_url/extract)
        # and fetch_url has never been called in this task, the LLM is writing
        # from pretrained knowledge — block it and force a fetch first.
        _LOOKUP_MARKERS = (
            "use fetch_url", "fetch_url to read", "use fetch_url to",
            "find the current", "find what ", "find the latest",
            "look up", "research the",
        )
        _goal_lower = orch._goal.lower()
        _is_lookup = any(m in _goal_lower for m in _LOOKUP_MARKERS)
        # Only enforce when there was NO pre-fetch: tasks with a pre-fetch
        # already have fetch_url content in history, so the LLM-initiated
        # flag is irrelevant and the guard would cause loops.
        if not orch._prefetched_content and _is_lookup and not orch._fetch_url_called:
            raise EnvironmentalError(
                "You must call fetch_url BEFORE writing the result. "
                "Do NOT write from memory or pretrained knowledge — "
                "call fetch_url with the appropriate URL first, "
                "then write the content you found."
            )

        # -- Python syntax pre-validation: reject .py writes that
        #    contain syntax errors BEFORE corrupting the file on disk.
        #    This catches unicode artefacts (…, —, etc.), placeholder
        #    strings ("corrected content here"), and truncated batches. --
        _wpath = args["path"]
        if _wpath.endswith(".py"):
            try:
                compile(content, _wpath, "exec")
            except SyntaxError as _syn:
                raise EnvironmentalError(
                    f"write_file REJECTED — Python syntax error in "
                    f"{_wpath}: {_syn.msg} at line {_syn.lineno}. "
                    f"The file on disk has NOT been modified. Fix the "
                    f"syntax error in the content field before writing."
                )

        actions.write_file(args["path"], content)

        # Update workspace cache with the content just written.
        _resolved_wf = actions._resolve_path(args["path"])
        orch._workspace_cache[_resolved_wf] = content

        # Fix 2 — duplicate section detection for markdown files
        # Fix 3 — file length guard for markdown files
        _path = args["path"]
        _warnings: list[str] = []
        if _path.endswith(".md"):
            from config import WRITE_FILE_MAX_LINES_MD
            lines = content.split("\n")
            line_count = len(lines)

            # Length guard
            if WRITE_FILE_MAX_LINES_MD and line_count > WRITE_FILE_MAX_LINES_MD:
                _warnings.append(
                    f"WARNING: file is {line_count} lines — exceeds "
                    f"{WRITE_FILE_MAX_LINES_MD}-line limit for an A4-page doc. "
                    f"Consider rewriting it to be shorter and more concise."
                )

            # Duplicate heading detection
            headings = [l.strip() for l in lines if l.strip().startswith("#")]
            seen_h: dict[str, int] = {}
            dupes: list[str] = []
            for h in headings:
                h_lower = h.lower()
                seen_h[h_lower] = seen_h.get(h_lower, 0) + 1
                if seen_h[h_lower] == 2:
                    dupes.append(h)
            if dupes:
                _warnings.append(
                    f"WARNING: duplicate section headings detected: "
                    f"{dupes}. The file has redundant sections — "
                    f"rewrite it to remove duplicate content."
                )

        result = {"status": "ok"}
        if _warnings:
            result["warnings"] = _warnings
        return result, None, None

    if action == "append_file":
        content = args["content"]
        if "${" in content or "<" in content:
            content = orch._resolve_placeholder_from_history(
                content, orch._history, None,
            )
        if not content or not content.strip():
            raise EnvironmentalError(
                "append_file called with empty content. "
                "Re-read the data you need and provide actual text."
            )
        actions.append_file(args["path"], content)

        # Refresh workspace cache from disk — appends are additive and
        # the cache must reflect the final file state, not just the block.
        _resolved_af = actions._resolve_path(args["path"])
        try:
            with open(_resolved_af, "r", errors="replace") as _f:
                orch._workspace_cache[_resolved_af] = _f.read()
        except OSError:
            pass  # non-fatal — cache may be stale but file was written

        return {"status": "ok"}, None, None

    if action == "delete_file":
        actions.delete_file(args["path"])

        # Remove deleted file from workspace cache.
        _resolved_df = actions._resolve_path(args["path"])
        orch._workspace_cache.pop(_resolved_df, None)

        return {"status": "ok"}, None, None

    if action == "list_dir":
        actions.list_dir(args["path"])
        return {"status": "ok"}, None, None

    if action == "index_codebase":
        try:
            result = actions.index_codebase(args["path"])
        except (OSError, ValueError) as exc:
            raise EnvironmentalError(str(exc))
        return result, None, None

    if action == "search_codebase":
        result = actions.search_codebase(
            path=args["path"],
            pattern=args["pattern"],
            glob=args.get("glob"),
            max_results=int(args.get("max_results", 50)),
        )
        return result, None, None

    if action == "ast_search":
        result = actions.ast_search(
            path=args["path"],
            query_type=args["query_type"],
            name=args["name"],
            glob=args.get("glob", "*.py"),
            max_results=int(args.get("max_results", 50)),
        )
        return result, None, None

    # -- Shell --
    if action == "run_command":
        cmd = args["cmd"]
        # Guard: block attempts to build a handoff package via shell.
        # run_command "never fails" — it returns a dict even on non-zero
        # exit, so the batch engine sees "success" and continues to any
        # batched `done`. Raising EnvironmentalError here is the only way
        # to actually stop the batch and force a replan.
        _cmd_lower = cmd.lower()
        # Block curl/wget to external URLs — use fetch_url instead.
        # This prevents the model from bypassing the fetch_url pipeline
        # (retries, content extraction, context injection) and avoids
        # silent failures from shell pipelines (empty grep output, etc.).
        # Localhost and 127.x URLs are explicitly allowed (local server tests).
        if re.search(r'\b(curl|wget)\b', cmd, re.IGNORECASE):
            if re.search(r'https?://(?!localhost|127\.)', cmd, re.IGNORECASE):
                raise EnvironmentalError(
                    "Do NOT use curl or wget to fetch content from external URLs. "
                    "Use the fetch_url action instead — it handles retries, "
                    "content extraction, and context injection automatically. "
                    "fetch_url also makes the content available in your history "
                    "for subsequent actions."
                )
        # Block rm -rf / rm -r outside ~/genie_workspace.
        # Targets under /tmp, /var, /usr, etc. are outside the workspace
        # sandbox and will be rejected by the blocklist anyway — but by
        # catching it here we can give a directive message immediately
        # rather than a generic "command rejected" error that the model
        # ignores and retries with variations.
        _workspace_dir = os.path.expanduser("~/genie_workspace")
        if re.search(r'\brm\s+-[rf]+\b', cmd, re.IGNORECASE):
            # Rebuild the target path: strip 'rm -rf ' prefix variants
            # Use a path-safe pattern: stop at whitespace or shell metacharacters
            _rm_match = re.search(
                r'\brm\s+-[rf]+\s+([^\s;&|]+)', cmd, re.IGNORECASE
            )
            if _rm_match:
                _rm_target = os.path.expanduser(
                    _rm_match.group(1).rstrip("/;'\"")
                )
                _rm_abs = os.path.abspath(_rm_target)
                if not (_rm_abs.startswith(_workspace_dir) or
                        _rm_abs == _workspace_dir):
                    raise EnvironmentalError(
                        f"rm -rf is blocked for paths outside "
                        f"~/genie_workspace (target: {_rm_target}). "
                        f"Use mkdir -p to create a fresh directory instead: "
                        f"run_command('mkdir -p {_rm_target}') — this ensures "
                        f"the path exists as a clean starting point without "
                        f"requiring delete permission. Do NOT retry rm -rf."
                    )
        _ZIP_HANDOFF_MARKERS = (
            "assemble_handoff",
            "handoff.zip", "handoff_zip",
            "import zipfile", "zipfile.",
            "make_archive", "-m zipfile",
            "zip -r ", "zip -rq ", "zip -qr ",
        )
        if any(kw in _cmd_lower for kw in _ZIP_HANDOFF_MARKERS):
            raise EnvironmentalError(
                "Do NOT use run_command to create a handoff zip. "
                "Call the assemble_handoff action directly with: "
                "repo_path, task_summary, output_dir, endpoints."
            )
        # Block `pip install --user` inside the active virtualenv.
        # Virtualenvs hide user site-packages so --user always fails with
        # "Can not perform a '--user' install". Drop --user and install
        # directly into the active env instead.
        if re.search(r'\bpip[23]?\s+install\b', cmd, re.IGNORECASE):
            if "--user" in _cmd_lower:
                raise EnvironmentalError(
                    "Cannot use 'pip install --user' inside a virtualenv — "
                    "user site-packages are disabled in this environment. "
                    "Remove the --user flag: use 'pip install <package>' "
                    "(or 'pip3 install <package>') to install into the "
                    "active virtualenv directly."
                )
        raw = actions.run_command(
            cmd, timeout=args.get("timeout"),
        )
        # Inject an environment hint when SSL cert verification fails.
        # This is an environment limitation (venv CA bundle), NOT a code
        # bug — the model should use verify=False in requests or urllib,
        # not install certifi or retry indefinitely.
        _combined_out = (raw.get("stdout", "") + raw.get("stderr", "")).lower()
        if ("certificate verify failed" in _combined_out or
                "ssl: certificate_verify_failed" in _combined_out or
                "sslcertverificationerror" in _combined_out):
            raw = dict(raw)  # make a mutable copy
            raw["_env_hint"] = (
                "SSL certificate verification failed — this is an environment "
                "limitation (the CA bundle in this venv cannot verify the host). "
                "If using the `requests` library, add verify=False to your "
                "request call: requests.get(url, verify=False). "
                "If using urllib, wrap with ssl.create_default_context() with "
                "check_hostname=False. Do NOT try to install certifi with --user."
            )
        return raw, None, None

    if action == "run_background":
        raw = actions.run_background(args["cmd"])
        return raw, None, None

    if action == "kill_process":
        actions.kill_process(int(args["pid"]))
        return {"status": "ok"}, None, None

    # -- Raw input --
    if action == "click":
        actions.click(int(args["x"]), int(args["y"]))
        return {"status": "ok"}, None, None

    if action == "press_key":
        actions.press_key(args["key"])
        return {"status": "ok"}, None, None

    if action == "type_text":
        actions.type_text(args["text"])
        return {"status": "ok"}, None, None

    if action == "wait":
        actions.wait(float(args["seconds"]))
        return {"status": "ok"}, None, None

    if action == "checkpoint":
        # No-op in dispatch — checkpoint is handled by _execute_batch
        # and the brain loop directly. If it reaches here (single-action
        # path), just return success.
        return {"status": "ok"}, None, None

    # -- Clipboard (GPaste) --
    if action == "list_clipboard_history":
        raw = actions.list_clipboard_history()
        return {"history": raw}, None, None

    if action == "get_clipboard_item":
        raw = actions.get_clipboard_item(int(args["index"]))
        return {"content": raw}, None, None

    if action == "paste_clipboard_item":
        actions.paste_clipboard_item(int(args["index"]))
        return {"status": "ok"}, None, None

    if action == "fetch_url":
        # Fix 3 — multi-tier URL content fetcher.
        # Cascade: llms.txt → known APIs → curl+html2text → Jina → Playwright.
        # Returns cleaned plain text windowed to the most relevant chunk.
        # Raises FetchError when all tiers are exhausted.
        from exceptions import FetchError as _FetchError
        from config import MAX_FETCH_PER_DOMAIN
        url   = args["url"]
        query = args.get("query", "")

        # Fix 4 — per-domain fetch rate limiter
        domain = urllib.parse.urlparse(url).netloc
        domain_count = orch._fetch_domain_counts.get(domain, 0)
        if domain_count >= MAX_FETCH_PER_DOMAIN:
            raise EnvironmentalError(
                f"fetch_url: domain '{domain}' already fetched {domain_count} times "
                f"in this task (limit {MAX_FETCH_PER_DOMAIN}). "
                f"Use the content already in your conversation history, "
                f"or try a DIFFERENT domain/URL. Do NOT re-fetch the same site."
            )

        try:
            text = actions.fetch_url(url, query)
            orch._fetch_url_called = True
            orch._fetch_domain_counts[domain] = domain_count + 1
            return {"content": text}, None, None
        except _FetchError as exc:
            orch._fetch_domain_counts[domain] = domain_count + 1
            # Surface as environmental_failure so the brain loop can decide
            # whether to fall back to pretrained knowledge or abort.
            raise EnvironmentalError(str(exc)) from exc

    if action == "open_pr":
        try:
            result = actions.open_pr(
                repo=args["repo"],
                title=args["title"],
                body=args["body"],
                head=args["head"],
                base=args.get("base", "main"),
            )
        except Exception as exc:
            # open_pr failures (missing token, wrong repo, HTTP error)
            # are environmental — the model should skip the PR step,
            # not halt the entire subtask.
            raise EnvironmentalError(str(exc)) from exc
        orch._telegram_notify(f"PR opened: {result['pr_url']}")
        return {
            "status":    "ok",
            "pr_url":    result["pr_url"],
            "pr_number": result["pr_number"],
        }, None, None

    # ------------------------------------------------------------------
    # GitHub API actions — all raise EnvironmentalError on failure so
    # the brain loop can replan rather than halting.
    # ------------------------------------------------------------------

    if action == "create_repo":
        try:
            result = actions.create_repo(
                name=args["name"],
                private=bool(args.get("private", True)),
                description=str(args.get("description", "")),
                auto_init=bool(args.get("auto_init", True)),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "delete_repo":
        try:
            result = actions.delete_repo(repo=args["repo"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "list_repos":
        try:
            result = actions.list_repos(user=str(args.get("user", "")))
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"repos": result}, None, None

    if action == "fork_repo":
        try:
            result = actions.fork_repo(repo=args["repo"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "list_branches":
        try:
            result = actions.list_branches(repo=args["repo"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"branches": result}, None, None

    if action == "create_branch":
        try:
            result = actions.create_branch(
                repo=args["repo"],
                branch=args["branch"],
                from_ref=str(args.get("from_ref", "main")),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "delete_branch":
        try:
            result = actions.delete_branch(repo=args["repo"], branch=args["branch"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "list_prs":
        try:
            result = actions.list_prs(
                repo=args["repo"],
                state=str(args.get("state", "open")),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"prs": result}, None, None

    if action == "merge_pr":
        try:
            result = actions.merge_pr(
                repo=args["repo"],
                pr_number=int(args["pr_number"]),
                merge_method=str(args.get("merge_method", "merge")),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "close_pr":
        try:
            result = actions.close_pr(repo=args["repo"], pr_number=int(args["pr_number"]))
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "create_issue":
        try:
            result = actions.create_issue(
                repo=args["repo"],
                title=args["title"],
                body=str(args.get("body", "")),
                labels=args.get("labels"),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "close_issue":
        try:
            result = actions.close_issue(
                repo=args["repo"],
                issue_number=int(args["issue_number"]),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "list_issues":
        try:
            result = actions.list_issues(
                repo=args["repo"],
                state=str(args.get("state", "open")),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"issues": result}, None, None

    if action == "create_release":
        try:
            result = actions.create_release(
                repo=args["repo"],
                tag=args["tag"],
                name=args["name"],
                body=str(args.get("body", "")),
                draft=bool(args.get("draft", False)),
                prerelease=bool(args.get("prerelease", False)),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "add_collaborator":
        try:
            result = actions.add_collaborator(
                repo=args["repo"],
                username=args["username"],
                permission=str(args.get("permission", "push")),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "get_file_contents":
        try:
            result = actions.get_file_contents(
                repo=args["repo"],
                path=args["path"],
                ref=str(args.get("ref", "main")),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "put_file":
        try:
            result = actions.put_file(
                repo=args["repo"],
                path=args["path"],
                content=args["content"],
                message=args["message"],
                sha=str(args.get("sha", "")),
                branch=str(args.get("branch", "main")),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    # ------------------------------------------------------------------
    # GitHub extended actions
    # ------------------------------------------------------------------

    if action == "get_authenticated_user":
        try:
            result = actions.get_authenticated_user()
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "update_repo":
        try:
            result = actions.update_repo(
                repo=args["repo"],
                name=str(args.get("name", "")),
                description=str(args.get("description", "")),
                private=args.get("private"),
                homepage=str(args.get("homepage", "")),
                has_issues=args.get("has_issues"),
                has_wiki=args.get("has_wiki"),
                archived=args.get("archived"),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "set_repo_topics":
        try:
            result = actions.set_repo_topics(repo=args["repo"], topics=args["topics"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "search_repos":
        try:
            result = actions.search_repos(
                query=args["query"],
                sort=str(args.get("sort", "best-match")),
                per_page=int(args.get("per_page", 10)),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"results": result}, None, None

    if action == "list_labels":
        try:
            result = actions.list_labels(repo=args["repo"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"labels": result}, None, None

    if action == "create_label":
        try:
            result = actions.create_label(
                repo=args["repo"],
                name=args["name"],
                color=args["color"],
                description=str(args.get("description", "")),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "protect_branch":
        try:
            result = actions.protect_branch(
                repo=args["repo"],
                branch=args["branch"],
                required_approvals=int(args.get("required_approvals", 1)),
                dismiss_stale_reviews=bool(args.get("dismiss_stale_reviews", False)),
                require_code_owner_reviews=bool(args.get("require_code_owner_reviews", False)),
                require_status_checks=args.get("require_status_checks"),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "list_webhooks":
        try:
            result = actions.list_webhooks(repo=args["repo"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"webhooks": result}, None, None

    if action == "create_webhook":
        try:
            result = actions.create_webhook(
                repo=args["repo"],
                url=args["url"],
                events=args.get("events"),
                secret=str(args.get("secret", "")),
                active=bool(args.get("active", True)),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "delete_webhook":
        try:
            result = actions.delete_webhook(repo=args["repo"], hook_id=int(args["hook_id"]))
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "list_workflows":
        try:
            result = actions.list_workflows(repo=args["repo"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"workflows": result}, None, None

    if action == "trigger_workflow":
        try:
            result = actions.trigger_workflow(
                repo=args["repo"],
                workflow=args["workflow"],
                ref=str(args.get("ref", "main")),
                inputs=args.get("inputs"),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "list_workflow_runs":
        try:
            result = actions.list_workflow_runs(
                repo=args["repo"],
                workflow=str(args.get("workflow", "")),
                status=str(args.get("status", "")),
                per_page=int(args.get("per_page", 10)),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"runs": result}, None, None

    if action == "create_gist":
        try:
            result = actions.create_gist(
                description=args["description"],
                files=args["files"],
                public=bool(args.get("public", False)),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "list_gists":
        try:
            result = actions.list_gists(user=str(args.get("user", "")))
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"gists": result}, None, None

    if action == "star_repo":
        try:
            result = actions.star_repo(repo=args["repo"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "unstar_repo":
        try:
            result = actions.unstar_repo(repo=args["repo"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "create_org_repo":
        try:
            result = actions.create_org_repo(
                org=args["org"],
                name=args["name"],
                private=bool(args.get("private", True)),
                description=str(args.get("description", "")),
                auto_init=bool(args.get("auto_init", True)),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "list_org_members":
        try:
            result = actions.list_org_members(org=args["org"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"members": result}, None, None

    if action == "list_teams":
        try:
            result = actions.list_teams(org=args["org"])
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"teams": result}, None, None

    if action == "list_packages":
        try:
            result = actions.list_packages(
                package_type=str(args.get("package_type", "container")),
                user=str(args.get("user", "")),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"packages": result}, None, None

    if action == "delete_package_version":
        try:
            result = actions.delete_package_version(
                package_type=args["package_type"],
                package_name=args["package_name"],
                version_id=int(args["version_id"]),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "list_notifications":
        try:
            result = actions.list_notifications(all_=bool(args.get("all", False)))
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"notifications": result}, None, None

    if action == "mark_notifications_read":
        try:
            result = actions.mark_notifications_read(repo=str(args.get("repo", "")))
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return result, None, None

    if action == "get_audit_log":
        try:
            result = actions.get_audit_log(
                org=args["org"],
                phrase=str(args.get("phrase", "")),
                per_page=int(args.get("per_page", 25)),
            )
        except Exception as exc:
            raise EnvironmentalError(str(exc)) from exc
        return {"events": result}, None, None

    if action == "assemble_handoff":
        try:
            result = actions.assemble_handoff(
                repo_path=args["repo_path"],
                task_summary=args["task_summary"],
                output_dir=args.get("output_dir", ""),
                endpoints=args.get("endpoints"),
            )
        except RuntimeError as exc:
            raise EnvironmentalError(str(exc)) from exc
        except OSError as exc:
            raise EnvironmentalError(str(exc)) from exc
        orch._telegram_notify(f"Handoff package ready: {result['package_path']}")
        return result, None, None

    # Fallback — should not reach here after validation
    raise UnrecoverableError(f"unhandled action: {action}")
