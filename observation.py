"""
Genie — Layer 3: Observation Engine

Captures passive system state after every Layer 4 action and writes a
structured JSONL entry.  Never calls the LLM.  Never executes actions.
observe() has an unconditional no-propagation contract — no exception
may exit it under any circumstances.
"""

import datetime
import json
import os
import subprocess
import sys
import time
import urllib.request

import psutil
import pyatspi
from websockets.sync.client import connect as ws_connect

from config import (
    ARGS_TRUNCATION_CHARS,
    ARGS_TRUNCATION_MARKER,
    OBSERVATION_OUTPUT_TRUNCATION_CHARS,
    OBSERVATION_OUTPUT_TRUNCATION_MARKER,
    MAX_READ_ELEMENT_CHARS,
    ATSPI_FOCUSED_SEARCH_MAX_NODES,
    CDP_RECV_TIMEOUT_SECONDS,
    CDP_EVENT_DISCARD_CAP,
    SUCCESS_LOG_PATH,
    INCOMPLETE_LOG_PATH,
)
from exceptions import (
    ElementResolverError,
    TransientError,
    EnvironmentalError,
    ResourceError,
    UnrecoverableError,
)


# =========================================================================
# Module-level functions
# =========================================================================

def _obs_cdp_send(ws, cmd_id_ref: list, method: str, params: dict) -> dict:
    """Send one CDP command and return response['result'].

    Simpler than ElementResolver._cdp_send — no error field classification,
    no exceptionDetails check.  All failures propagate to caller; the caller
    catches everything and maps to element_state_cdp=None, observation_partial=True.

    Args:
        ws: Active websocket connection.
        cmd_id_ref: Mutable single-element list [n] — incremented on each call.
        method: CDP method string (e.g. "DOM.resolveNode").
        params: CDP method params dict.

    Returns:
        response["result"] dict on successful matched response.
    """
    cmd_id_ref[0] += 1
    cmd_id = cmd_id_ref[0]

    payload = json.dumps({"id": cmd_id, "method": method, "params": params})
    ws.send(payload)

    discard_count = 0
    while True:
        raw = ws.recv(timeout=CDP_RECV_TIMEOUT_SECONDS)
        response = json.loads(raw)
        if response.get("id") == cmd_id:
            return response["result"]
        # Unsolicited event — discard
        discard_count += 1
        if discard_count >= CDP_EVENT_DISCARD_CAP:
            raise RuntimeError(
                f"_obs_cdp_send: discarded {discard_count} events waiting "
                f"for id={cmd_id} method={method}"
            )


def _make_focused_predicate(max_nodes: int):
    """Return a stateful closure suitable for pyatspi.findDescendant().

    The closure increments an internal counter on every call.  When the
    counter exceeds *max_nodes* it raises StopIteration — this propagates
    out of findDescendant() and must be caught by the outer try/except.
    """
    count = [0]

    def predicate(node):
        count[0] += 1
        if count[0] > max_nodes:
            raise StopIteration
        try:
            return pyatspi.STATE_FOCUSED in node.getState().getStates()
        except Exception:
            return False

    return predicate


# =========================================================================
# Observer class
# =========================================================================

class Observer:
    """Layer 3 observation engine.

    Captures passive system state after every Layer 4 action and writes a
    structured JSONL entry.  One instance per process, shared across all
    task runs.
    """

    # Args fields that receive truncation before logging.
    _TRUNCATE_ARG_KEYS = frozenset({"content", "text", "cmd"})

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)

        self._incomplete_fh = open(INCOMPLETE_LOG_PATH, "a", encoding="utf-8")
        self._success_fh = open(SUCCESS_LOG_PATH, "a", encoding="utf-8")

        self.task_id: str | None = None
        self.sequence: int = 0
        self.is_running: bool = False
        self.log_write_failed: bool = False
        self.observe_guard_fired_count: int = 0
        self.observe_guard_last_action: str | None = None
        self.last_entry: dict | None = None

    # -----------------------------------------------------------------
    # Task lifecycle
    # -----------------------------------------------------------------

    def start_task(self, task_id: str, initial_sequence: int = 0) -> None:
        """Begin a new task.  Raises RuntimeError on double-call."""
        if self.is_running:
            raise RuntimeError(
                "Observer.start_task() called while task already running"
            )

        # Surface guard fires from previous task before wiping
        if self.observe_guard_fired_count > 0:
            print(
                f"[OBSERVER] previous task had {self.observe_guard_fired_count} "
                f"observe() guard fires (last_action="
                f"{self.observe_guard_last_action})",
                file=sys.stderr,
            )

        self.task_id = task_id
        self.sequence = initial_sequence
        self.is_running = True
        self.log_write_failed = False
        self.observe_guard_fired_count = 0
        self.observe_guard_last_action = None

    def end_task(self) -> None:
        """End the current task.  No-op if already stopped."""
        if not self.is_running:
            return
        self.is_running = False

    # -----------------------------------------------------------------
    # observe_think() — THINK entry logging
    # -----------------------------------------------------------------

    def observe_think(self, think_content: str) -> None:
        """Log a THINK entry to genie_incomplete.jsonl.

        NEVER raises — unconditional no-propagation contract.
        """
        if not self.is_running:
            return

        self.sequence += 1

        timestamp = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="milliseconds")
        )

        entry = {
            "task_id": self.task_id,
            "sequence": self.sequence,
            "timestamp": timestamp,
            "action": "think",
            "args": {},
            "result": None,
            "observation": {},
            "error": None,
            "think": think_content,
            "duration_ms": None,
        }

        try:
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            self._incomplete_fh.write(line)
            self._incomplete_fh.flush()
        except Exception as exc:
            if not self.log_write_failed:
                print(
                    f"[OBSERVER] JSONL write failed in observe_think "
                    f"(task_id={self.task_id}, seq={self.sequence}): {exc}",
                    file=sys.stderr,
                )
            self.log_write_failed = True

    # -----------------------------------------------------------------
    # observe_plan() — PLAN entry logging
    # -----------------------------------------------------------------

    def observe_plan(self, plan: dict) -> None:
        """Log a PLAN entry to genie_incomplete.jsonl.

        NEVER raises — unconditional no-propagation contract.
        """
        if not self.is_running:
            return

        timestamp = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="milliseconds")
        )

        entry = {
            "task_id": self.task_id,
            "sequence": 0,
            "timestamp": timestamp,
            "action": "plan",
            "args": {},
            "result": None,
            "observation": {},
            "error": None,
            "plan": plan,
            "duration_ms": None,
        }

        try:
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            self._incomplete_fh.write(line)
            self._incomplete_fh.flush()
        except Exception as exc:
            if not self.log_write_failed:
                print(
                    f"[OBSERVER] JSONL write failed in observe_plan "
                    f"(task_id={self.task_id}, seq=0): {exc}",
                    file=sys.stderr,
                )
            self.log_write_failed = True

    # -----------------------------------------------------------------
    # observe() — main entry point
    # -----------------------------------------------------------------

    def observe(
        self,
        action_dict: dict | None,
        result=None,
        error=None,
        attempt: int = 1,
        t_start: float | None = None,
        tier: str | None = None,
        wid: str | None = None,
        llm_response: str | None = None,
        llm_messages: list | None = None,
    ) -> None:
        """Capture observation and write JSONL entry.

        NEVER raises — unconditional no-propagation contract.
        """
        try:
            self._observe_inner(
                action_dict, result, error, attempt, t_start, tier, wid,
                llm_response=llm_response, llm_messages=llm_messages,
            )
        except Exception:
            # Absolute last-resort catch — should never fire if all inner
            # handlers are correctly guarded, but the contract is unconditional.
            try:
                print(
                    "[OBSERVER] unhandled exception escaped _observe_inner",
                    file=sys.stderr,
                )
            except Exception:
                pass

    def _observe_inner(
        self,
        action_dict, result, error, attempt, t_start, tier, wid,
        llm_response=None, llm_messages=None,
    ) -> None:
        """Inner observe implementation.  May raise — caller catches."""

        # -- 6a: Guard check --
        if not self.is_running:
            self.observe_guard_fired_count += 1
            self.observe_guard_last_action = (
                action_dict.get("action", "unknown")
                if action_dict else "unknown"
            )
            print(
                f"[OBSERVER] observe() called outside active task — skipping "
                f"(count={self.observe_guard_fired_count}, "
                f"last_action={self.observe_guard_last_action})",
                file=sys.stderr,
            )
            return

        # -- 6b: Entry computations --
        self.sequence += 1

        if t_start is not None:
            duration_ms = int((time.time() - t_start) * 1000)
        else:
            duration_ms = None

        timestamp = (
            datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        )

        action_name = (
            action_dict.get("action", "unknown") if action_dict else "unknown"
        )
        raw_args = action_dict.get("args", {}) if action_dict else {}

        # Args truncation (Section 7)
        logged_args = self._truncate_args(raw_args)

        # -- 6c: Result derivation (Section 8) --
        result_str, error_dict = self._derive_result(
            action_name, result, error,
        )

        # -- 6d: Observation capture (Section 9) --
        observation_dict = self._capture_observation(
            action_name, action_dict, result, error, tier, wid,
        )

        # -- 6e: Assemble and write --
        entry = {
            "task_id": self.task_id,
            "sequence": self.sequence,
            "attempt": attempt,
            "timestamp": timestamp,
            "action": action_name,
            "args": logged_args,
            "result": result_str,
            "observation": observation_dict,
            "error": error_dict,
            "duration_ms": duration_ms,
            "llm_response": llm_response,
            "llm_messages": llm_messages,
        }

        self.last_entry = entry

        # -- Section 11: JSONL write --
        try:
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            self._incomplete_fh.write(line)
            self._incomplete_fh.flush()
            if result_str == "success":
                self._success_fh.write(line)
                self._success_fh.flush()
        except Exception as exc:
            if not self.log_write_failed:
                print(
                    f"[OBSERVER] JSONL write failed: {exc}", file=sys.stderr,
                )
            self.log_write_failed = True

    # -----------------------------------------------------------------
    # Section 7 — Args truncation
    # -----------------------------------------------------------------

    def _truncate_args(self, raw_args: dict) -> dict:
        """Return a shallow copy of *raw_args* with large fields truncated."""
        logged = {}
        for key, value in raw_args.items():
            if key in self._TRUNCATE_ARG_KEYS:
                s = str(value)
                if len(s) > ARGS_TRUNCATION_CHARS:
                    marker = ARGS_TRUNCATION_MARKER.format(
                        limit=ARGS_TRUNCATION_CHARS,
                    )
                    logged[key] = s[:ARGS_TRUNCATION_CHARS] + marker
                else:
                    logged[key] = value
            else:
                logged[key] = value
        return logged

    # -----------------------------------------------------------------
    # Section 8 — Result derivation
    # -----------------------------------------------------------------

    def _derive_result(
        self, action_name: str, result, error,
    ) -> tuple:
        """Derive (result_str, error_dict) from result and error.

        Five cases evaluated strictly in order:
          Case 0 — run_command exit check
          Case 1 — Success
          Case 2 — Known ElementResolverError subclass
          Case 3 — Unknown Exception subclass
          Case 4 — Both-None safety net
        """

        # Case 0 — run_command exit check
        if action_name == "run_command" and result is not None:
            exit_code = result["exit_code"]
            timed_out = result["timed_out"]
            if timed_out is True:
                return (
                    "command_timeout",
                    {
                        "type": "CommandTimeout",
                        "message": (
                            f"exit_code={exit_code} timed_out={timed_out}"
                        ),
                    },
                )
            if exit_code != 0:
                return (
                    "command_failed",
                    {
                        "type": "CommandFailed",
                        "message": (
                            f"exit_code={exit_code} timed_out={timed_out}"
                        ),
                    },
                )
            # exit_code == 0 and timed_out == False → fall through to Case 1

        # Case 1 — Success
        if error is None and result is not None:
            return ("success", None)

        # Case 2 — Known ElementResolverError subclass
        if isinstance(error, ElementResolverError):
            if isinstance(error, TransientError):
                result_str = "transient_failure"
            elif isinstance(error, EnvironmentalError):
                result_str = "environmental_failure"
            elif isinstance(error, ResourceError):
                result_str = "environmental_failure"
            elif isinstance(error, UnrecoverableError):
                result_str = "unrecoverable"
            else:
                # Future subclass — safe fallback
                result_str = "unrecoverable"
            return (
                result_str,
                {"type": type(error).__name__, "message": str(error)},
            )

        # Case 3 — OS / standard Python exceptions → environmental by default
        if isinstance(error, OSError):
            return (
                "environmental_failure",
                {"type": type(error).__name__, "message": str(error)},
            )

        # Case 3b — Unknown Exception subclass
        if error is not None:
            return (
                "unrecoverable",
                {"type": type(error).__name__, "message": str(error)},
            )

        # Case 4 — Both-None safety net
        return (
            "unrecoverable",
            {
                "type": "ObservationInternalError",
                "message": (
                    "both result and error were None "
                    "— unclassified failure in call pattern"
                ),
            },
        )

    # -----------------------------------------------------------------
    # Section 9 — Per-action observation dispatch
    # -----------------------------------------------------------------

    def _capture_observation(
        self, action_name, action_dict, result, error, tier, wid,
    ) -> dict:
        """Dispatch to the correct per-action handler.  Never raises."""
        try:
            if action_name == "read_file":
                return self._obs_read_file(action_dict, result, error)
            if action_name == "write_file":
                return self._obs_write_file(action_dict, result, error)
            if action_name == "append_file":
                return self._obs_append_file(action_dict, result, error)
            if action_name == "delete_file":
                return self._obs_delete_file(action_dict, result, error)
            if action_name == "list_dir":
                return self._obs_list_dir(action_dict, result, error)
            if action_name == "run_command":
                return self._obs_run_command(action_dict, result, error)
            if action_name == "run_background":
                return self._obs_run_background(action_dict, result, error)
            if action_name == "kill_process":
                return self._obs_kill_process(action_dict, result, error)
            if action_name == "open_app":
                return self._obs_open_app(
                    action_dict, result, error, wid=wid, tier=tier,
                )
            if action_name == "focus_window":
                return self._obs_focus_window(action_dict, result, error)
            if action_name in ("click", "press_key", "type_text"):
                return self._obs_raw_input(action_dict, result, error)
            if action_name == "click_element":
                return self._obs_click_element(action_dict, result, error)
            if action_name == "type_element":
                return self._obs_type_element(action_dict, result, error)
            if action_name == "read_element":
                return self._obs_read_element(action_dict, result, error)
            if action_name == "look":
                return self._obs_look(action_dict, result, error)
            if action_name == "list_clipboard_history":
                return self._obs_clipboard_history(action_dict, result, error)
            if action_name == "get_clipboard_item":
                return self._obs_clipboard_get(action_dict, result, error)
            if action_name in ("paste_clipboard_item", "wait", "chat",
                               "checkpoint"):
                return {}
            # Unknown action
            return {"observation_partial": True}
        except Exception:
            return {"observation_partial": True}

    # -----------------------------------------------------------------
    # 9.1 — read_file
    # -----------------------------------------------------------------

    def _obs_read_file(self, action_dict, result, error) -> dict:
        if error is not None:
            return {
                "file_exists": None,
                "file_size_bytes": None,
                "mtime": None,
            }
        path = action_dict["args"]["path"]
        content = result.get("content", "") if result else ""
        if len(content) > OBSERVATION_OUTPUT_TRUNCATION_CHARS:
            marker = OBSERVATION_OUTPUT_TRUNCATION_MARKER.format(
                limit=OBSERVATION_OUTPUT_TRUNCATION_CHARS,
            )
            content = content[:OBSERVATION_OUTPUT_TRUNCATION_CHARS] + marker
        try:
            stat = os.stat(path)
            return {
                "file_exists": True,
                "file_size_bytes": stat.st_size,
                "mtime": (
                    datetime.datetime.utcfromtimestamp(stat.st_mtime)
                    .isoformat(timespec="milliseconds") + "Z"
                ),
                "content": content,
            }
        except OSError:
            return {
                "file_exists": None,
                "file_size_bytes": None,
                "mtime": None,
                "content": content,
                "observation_partial": True,
            }

    # -----------------------------------------------------------------
    # 9.2 — write_file
    # -----------------------------------------------------------------

    def _obs_write_file(self, action_dict, result, error) -> dict:
        if error is not None:
            return {"file_exists": None, "file_size_bytes": None}
        path = action_dict["args"]["path"]
        try:
            stat = os.stat(path)
            # Read content to provide word/line counts — helps LLM verify
            # size constraints (e.g. "at least 200 words") without a read-back.
            try:
                with open(path, "r", errors="replace") as f:
                    content = f.read(262144)  # cap at 256KB
                word_count = len(content.split())
                line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            except OSError:
                word_count = None
                line_count = None
            return {
                "file_exists": True,
                "file_size_bytes": stat.st_size,
                "word_count": word_count,
                "line_count": line_count,
            }
        except OSError:
            return {
                "file_exists": None,
                "file_size_bytes": None,
                "observation_partial": True,
            }

    # -----------------------------------------------------------------
    # 9.3 — append_file
    # -----------------------------------------------------------------

    def _obs_append_file(self, action_dict, result, error) -> dict:
        if error is not None:
            return {"file_exists": None, "file_size_bytes": None}
        path = action_dict["args"]["path"]
        try:
            stat = os.stat(path)
            return {"file_exists": True, "file_size_bytes": stat.st_size}
        except OSError:
            return {
                "file_exists": None,
                "file_size_bytes": None,
                "observation_partial": True,
            }

    # -----------------------------------------------------------------
    # 9.4 — delete_file
    # -----------------------------------------------------------------

    def _obs_delete_file(self, action_dict, result, error) -> dict:
        if error is not None:
            return {"file_exists": None}
        path = action_dict["args"]["path"]
        try:
            os.stat(path)
            # stat succeeded — file still exists, unexpected on successful delete
            return {"file_exists": True, "observation_partial": True}
        except FileNotFoundError:
            return {"file_exists": False}
        except OSError:
            return {"file_exists": None, "observation_partial": True}

    # -----------------------------------------------------------------
    # 9.5 — list_dir
    # -----------------------------------------------------------------

    def _obs_list_dir(self, action_dict, result, error) -> dict:
        if error is not None:
            return {"path_exists": None, "entry_count": None}
        path = action_dict["args"]["path"]
        try:
            entries = sorted(os.listdir(path))
            return {"path_exists": True, "entry_count": len(entries), "entries": entries}
        except OSError:
            return {
                "path_exists": None,
                "entry_count": None,
                "entries": None,
                "observation_partial": True,
            }

    # -----------------------------------------------------------------
    # 9.6 — run_command
    # -----------------------------------------------------------------

    def _obs_run_command(self, action_dict, result, error) -> dict:
        if result is None:
            return {
                "exit_code": None,
                "stdout": None,
                "stderr": None,
                "timed_out": None,
                "observation_partial": True,
            }
        stdout = result["stdout"]
        stderr = result["stderr"]
        if len(stdout) > OBSERVATION_OUTPUT_TRUNCATION_CHARS:
            marker = OBSERVATION_OUTPUT_TRUNCATION_MARKER.format(
                limit=OBSERVATION_OUTPUT_TRUNCATION_CHARS,
            )
            stdout = stdout[:OBSERVATION_OUTPUT_TRUNCATION_CHARS] + marker
        if len(stderr) > OBSERVATION_OUTPUT_TRUNCATION_CHARS:
            marker = OBSERVATION_OUTPUT_TRUNCATION_MARKER.format(
                limit=OBSERVATION_OUTPUT_TRUNCATION_CHARS,
            )
            stderr = stderr[:OBSERVATION_OUTPUT_TRUNCATION_CHARS] + marker
        return {
            "exit_code": result["exit_code"],
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": result["timed_out"],
            **( {"env_hint": result["_env_hint"]}
                if result.get("_env_hint") else {} ),
        }

    # -----------------------------------------------------------------
    # 9.7 — run_background
    # -----------------------------------------------------------------

    def _obs_run_background(self, action_dict, result, error) -> dict:
        if result is None:
            return {"process_alive": None, "observation_partial": True}
        pid = result["pid"]
        try:
            status = psutil.Process(pid).status()
            if status == psutil.STATUS_ZOMBIE:
                return {"process_alive": False}
            return {"process_alive": True}
        except psutil.NoSuchProcess:
            return {"process_alive": False}
        except psutil.AccessDenied:
            print(
                f"[OBSERVER] psutil.AccessDenied for pid={pid} "
                f"in run_background",
                file=sys.stderr,
            )
            return {"process_alive": None, "observation_partial": True}

    # -----------------------------------------------------------------
    # 9.8 — kill_process
    # -----------------------------------------------------------------

    def _obs_kill_process(self, action_dict, result, error) -> dict:
        pid = action_dict["args"]["pid"]
        try:
            status = psutil.Process(pid).status()
            if status == psutil.STATUS_ZOMBIE:
                return {"process_alive": False}
            return {"process_alive": True}
        except psutil.NoSuchProcess:
            return {"process_alive": False}
        except psutil.AccessDenied:
            return {"process_alive": None, "observation_partial": True}

    # -----------------------------------------------------------------
    # 9.9 — open_app
    # -----------------------------------------------------------------

    def _obs_open_app(
        self, action_dict, result, error,
        wid: str | None = None, tier: str | None = None,
    ) -> dict:
        app_label = action_dict.get("args", {}).get("app") if action_dict else None
        _null_obs = {
            "window_title": None,
            "top_level_children": None,
            "observation_partial": True,
            "registered_as": app_label,
        }
        try:
            if wid is None:
                return _null_obs

            # Tier-aware polling budget
            if tier is None:
                print(
                    "[OBSERVER] open_app tier is None — using 6×500ms "
                    "safe fallback",
                    file=sys.stderr,
                )
                polls = 6
            elif tier == "cdp_primary":
                polls = 6
            else:
                polls = 3
            interval = 0.5

            # Step 1 — xdotool geometry for the WID
            try:
                geo_result = subprocess.run(
                    ["xdotool", "getwindowgeometry", "--shell", wid],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except Exception:
                return _null_obs

            if geo_result.returncode != 0:
                return _null_obs

            geo = {}
            for line in geo_result.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    geo[k.strip()] = v.strip()

            try:
                target_x = int(geo["X"])
                target_y = int(geo["Y"])
                target_w = int(geo["WIDTH"])
                target_h = int(geo["HEIGHT"])
            except (KeyError, ValueError):
                return _null_obs

            # Step 2 — Poll for AT-SPI window matching geometry
            active_window = None
            for _poll in range(polls):
                desktop = pyatspi.Registry.getDesktop(0)
                for i in range(desktop.childCount):
                    child = desktop.getChildAtIndex(i)
                    if child is None:
                        continue
                    try:
                        comp = child.queryComponent()
                        ext = comp.getExtents(pyatspi.DESKTOP_COORDS)
                        if (
                            abs(ext.x - target_x) <= 20
                            and abs(ext.y - target_y) <= 20
                            and abs(ext.width - target_w) <= 20
                            and abs(ext.height - target_h) <= 20
                        ):
                            active_window = child
                            break
                    except Exception:
                        continue
                if active_window is not None:
                    break
                time.sleep(interval)

            if active_window is None:
                return _null_obs

            # Step 3 — window title
            window_title = active_window.name

            # Step 4 — Poll for at least one AT-SPI child to appear
            remaining_polls = polls - (_poll + 1)
            for _ in range(remaining_polls):
                if active_window.childCount > 0:
                    break
                time.sleep(interval)

            if active_window.childCount == 0:
                return {
                    "window_title": window_title,
                    "top_level_children": [],
                    "observation_partial": True,
                    "registered_as": app_label,
                }

            # Step 5 — Collect top-level children
            top_level_children = []
            for i in range(active_window.childCount):
                child = active_window.getChildAtIndex(i)
                if child is None:
                    continue
                try:
                    top_level_children.append({
                        "role": child.getRoleName(),
                        "name": child.name,
                    })
                except Exception:
                    continue

            return {
                "window_title": window_title,
                "top_level_children": top_level_children,
                "registered_as": app_label,
            }

        except Exception:
            return {
                "window_title": None,
                "top_level_children": None,
                "observation_partial": True,
                "registered_as": app_label,
            }

    # -----------------------------------------------------------------
    # 9.10 — shared: active window identification
    # -----------------------------------------------------------------

    def _obs_get_active_window(self):
        """Return the first pyatspi desktop child with STATE_ACTIVE, or None."""
        try:
            desktop = pyatspi.Registry.getDesktop(0)
            for i in range(desktop.childCount):
                child = desktop.getChildAtIndex(i)
                if child is None:
                    continue
                try:
                    if pyatspi.STATE_ACTIVE in child.getState().getStates():
                        return child
                except Exception:
                    continue
            return None
        except Exception:
            return None

    # -----------------------------------------------------------------
    # 9.11 — shared: focused element
    # -----------------------------------------------------------------

    def _obs_get_focused_element(self, active_window):
        """Return (focused_role, focused_name) or (None, None)."""
        if active_window is None:
            return (None, None)
        try:
            predicate = _make_focused_predicate(ATSPI_FOCUSED_SEARCH_MAX_NODES)
            focused = pyatspi.findDescendant(
                active_window, predicate, breadthFirst=True,
            )
            if focused is None:
                return (None, None)
            return (focused.getRoleName(), focused.name)
        except Exception:
            return (None, None)

    # -----------------------------------------------------------------
    # 9.12 — focus_window
    # -----------------------------------------------------------------

    def _obs_focus_window(self, action_dict, result, error) -> dict:
        active_window = self._obs_get_active_window()
        if active_window is None:
            return {"window_title": None, "observation_partial": True}
        return {"window_title": active_window.name}

    # -----------------------------------------------------------------
    # 9.13 — raw input (click, press_key, type_text)
    # -----------------------------------------------------------------

    def _obs_raw_input(self, action_dict, result, error) -> dict:
        active_window = self._obs_get_active_window()
        if active_window is None:
            return {
                "window_title": None,
                "focused_role": None,
                "focused_name": None,
                "observation_partial": True,
            }
        window_title = active_window.name
        focused_role, focused_name = self._obs_get_focused_element(
            active_window,
        )
        obs = {
            "window_title": window_title,
            "focused_role": focused_role,
            "focused_name": focused_name,
        }
        if focused_role is None:
            obs["observation_partial"] = True
        return obs

    # -----------------------------------------------------------------
    # 9.14 — click_element
    # -----------------------------------------------------------------

    def _obs_click_element(self, action_dict, result, error) -> dict:
        # Failure path — result is None
        if result is None:
            return {"tier_used": None, "observation_partial": True}

        observation_partial = False

        # Step 1 — Derive tier_used
        tier_val = result.get("tier")
        if tier_val == "cdp":
            backend_node_id = result.get("backendDOMNodeId")
            if backend_node_id is not None:
                tier_used = "cdp"
            else:
                tier_used = "vision"
        elif tier_val == "cdp_vision":
            # Vision-discovery fallback path: element located by screenshot,
            # not by AX tree. Treat same as CDP vision disambiguation.
            tier_used = "vision"
        elif tier_val in ("atspi", "terminal"):
            tier_used = tier_val
        else:
            tier_used = None
            observation_partial = True
            print(
                f"[OBSERVER] unexpected tier value '{tier_val}' "
                f"in click_element result",
                file=sys.stderr,
            )

        # Step 2 — Active window and focused element
        active_window = self._obs_get_active_window()
        window_title = (
            active_window.name if active_window is not None else None
        )
        focused_role, focused_name = self._obs_get_focused_element(
            active_window,
        )
        if focused_role is None:
            observation_partial = True

        # Step 3 — element_state from result dict (AT-SPI only)
        element_state = result.get("element_state")

        # Step 4 — CDP sequence (only when tier_used == "cdp" or "vision")
        element_state_cdp = None
        run_cdp = tier_used in ("cdp", "vision")

        if run_cdp:
            if tier_used == "vision":
                # Step 0b — vision None guard
                element_state_cdp = None
                observation_partial = True
            else:
                # Steps 1–6 — single-match CDP path
                try:
                    cdp_port = result["cdp_port"]

                    # Step 2 — open ephemeral CDP session
                    url = f"http://localhost:{cdp_port}/json"
                    resp = urllib.request.urlopen(url)
                    targets = json.loads(resp.read().decode())
                    pages = [t for t in targets if t.get("type") == "page"]
                    if not pages:
                        raise RuntimeError("no page targets found")
                    ws_url = pages[0]["webSocketDebuggerUrl"]

                    ws = ws_connect(ws_url)
                    try:
                        cmd_id_ref = [0]

                        # Step 4 — pre-observe delay
                        time.sleep(0.1)

                        # Step 5 — DOM.resolveNode
                        resolve_result = _obs_cdp_send(
                            ws, cmd_id_ref, "DOM.resolveNode",
                            {"backendNodeId": result["backendDOMNodeId"]},
                        )
                        object_id = resolve_result["object"]["objectId"]

                        # Step 6 — Runtime.callFunctionOn
                        try:
                            call_result = _obs_cdp_send(
                                ws, cmd_id_ref, "Runtime.callFunctionOn",
                                {
                                    "objectId": object_id,
                                    "functionDeclaration": (
                                        "function(){ return this.checked "
                                        "?? this.getAttribute('aria-checked') }"
                                    ),
                                    "returnByValue": True,
                                },
                            )
                            raw_value = call_result["result"]["value"]

                            # Normalization
                            if isinstance(raw_value, bool):
                                element_state_cdp = raw_value
                            elif raw_value == "true":
                                element_state_cdp = True
                            elif raw_value == "false":
                                element_state_cdp = False
                            elif raw_value == "mixed":
                                element_state_cdp = "mixed"
                            else:
                                element_state_cdp = None
                        finally:
                            # Release object — swallow failure
                            try:
                                _obs_cdp_send(
                                    ws, cmd_id_ref,
                                    "Runtime.releaseObject",
                                    {"objectId": object_id},
                                )
                            except Exception:
                                pass
                    finally:
                        ws.close()

                except Exception:
                    element_state_cdp = None
                    observation_partial = True

        # Step 5 — Assemble observation
        obs = {
            "tier_used": tier_used,
            "window_title": window_title,
            "focused_role": focused_role,
            "focused_name": focused_name,
            "element_state": element_state,
        }

        if run_cdp:
            obs["element_state_cdp"] = element_state_cdp

        if observation_partial:
            obs["observation_partial"] = True

        return obs

    # -----------------------------------------------------------------
    # 9.15 — type_element
    # -----------------------------------------------------------------

    def _obs_type_element(self, action_dict, result, error) -> dict:
        # Failure path — result is None
        if result is None:
            return {
                "tier_used": None,
                "element_text_after": None,
                "observation_partial": True,
            }

        # Derive tier_used (same logic as click_element Step 1)
        tier_val = result.get("tier")
        if tier_val == "cdp":
            backend_node_id = result.get("backendDOMNodeId")
            if backend_node_id is not None:
                tier_used = "cdp"
            else:
                tier_used = "vision"
        elif tier_val == "cdp_vision":
            # Vision-discovery fallback path: element located by screenshot.
            tier_used = "vision"
        elif tier_val in ("atspi", "terminal"):
            tier_used = tier_val
        else:
            tier_used = None
            print(
                f"[OBSERVER] unexpected tier value '{tier_val}' "
                f"in type_element result",
                file=sys.stderr,
            )

        # Uniform policy — no readback attempted
        return {
            "tier_used": tier_used,
            "element_text_after": None,
            "observation_partial": True,
        }

    # -----------------------------------------------------------------
    # 9.16 — read_element
    # -----------------------------------------------------------------

    def _obs_read_element(self, action_dict, result, error) -> dict:
        # Failure path — result is None
        if result is None:
            return {"tier_used": None, "value": None}

        # Success path
        raw_value = result["content"]

        # Defensive truncation at MAX_READ_ELEMENT_CHARS
        if isinstance(raw_value, str) and len(raw_value) > MAX_READ_ELEMENT_CHARS:
            raw_value = raw_value[:MAX_READ_ELEMENT_CHARS]

        return {"tier_used": None, "value": raw_value}

    # -----------------------------------------------------------------
    # 9.16b — look (vision screenshot description)
    # -----------------------------------------------------------------

    def _obs_look(self, action_dict, result, error) -> dict:
        # Failure path
        if result is None:
            return {"description": None}

        # Success path — bounded text description goes into LLM context
        desc = result.get("description", "")
        # Cap at 500 chars to keep context tight
        if isinstance(desc, str) and len(desc) > 500:
            desc = desc[:500] + "..."
        return {"description": desc}

    # -----------------------------------------------------------------
    # 9.17 — list_clipboard_history (GPaste)
    # -----------------------------------------------------------------

    def _obs_clipboard_history(self, action_dict, result, error) -> dict:
        if error is not None:
            return {"history": [], "count": 0}
        history = result.get("history", []) if result else []
        return {"history": history, "count": len(history)}

    # -----------------------------------------------------------------
    # 9.18 — get_clipboard_item (GPaste)
    # -----------------------------------------------------------------

    def _obs_clipboard_get(self, action_dict, result, error) -> dict:
        if error is not None:
            return {"content": None}
        content = result.get("content", "") if result else ""
        # Truncate for context efficiency
        if isinstance(content, str) and len(content) > MAX_READ_ELEMENT_CHARS:
            content = content[:MAX_READ_ELEMENT_CHARS]
        return {"content": content}