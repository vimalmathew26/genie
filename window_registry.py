"""
Genie Window Registry (Layer 1)

Manages application lifecycle: launch, WID tracking, CDP/Electron port
assignment, and window state.  Built across three prompts — this file
contains the foundation segment (Prompt 1).

Prompt 1 — Foundation:
    LaunchFailureError, _get_display(), WindowRegistry.__init__(), _assign_port()

Prompt 2 — open_app, polling loop, WID validation, _cleanup_cdp_profile

Prompt 3 — background poller, start_poller / stop_poller
"""

import os
import time
import socket
import threading
import subprocess
import queue

from Xlib import display as xdisplay
import Xlib.X
import Xlib.error

import psutil

from config import (
    APP_PROFILES,
    LAUNCH_TIMEOUTS,
    WID_POLL_INTERVAL,
    RETRY_TIMEOUT_MULTIPLIERS,
    MAX_OPEN_APP_RETRIES,
    FOCUS_WINDOW_TIMEOUT,
    CDP_BASE_PORT,
    ELECTRON_BASE_PORT,
    CDP_PORT_SCAN_MAX,
    LOCK_CLEANUP_WAIT_TIMEOUT,
    PROCESS_WAIT_TIMEOUT,
    log,
)


# =============================================================================
# Custom Exceptions
# =============================================================================

class LaunchFailureError(Exception):
    """Raised when an application fails to launch after all retry attempts.

    Attributes:
        app_name:    Binary or label of the application that failed.
        attempts:    Total launch attempts made (original + retries).
        last_reason: Human-readable description of the final failure.
    """

    def __init__(self, app_name: str, attempts: int, last_reason: str):
        self.app_name = app_name
        self.attempts = attempts
        self.last_reason = last_reason
        message = (
            f"Failed to launch '{app_name}' after {attempts} attempt(s): "
            f"{last_reason}"
        )
        super().__init__(message)


# =============================================================================
# Thread-local X11 Display Management
# =============================================================================

_thread_local = threading.local()


def _get_display() -> xdisplay.Display:
    """Return the thread-local X11 Display, creating it on first call.

    On first invocation per thread, opens a new ``Display()`` connection
    and interns the three atoms used throughout this module.  Subsequent
    calls on the same thread return the cached instance without any
    round-trips to the X server.

    Interned atoms cached on ``_thread_local``:
        atom_net_wm_type          — ``_NET_WM_WINDOW_TYPE``
        atom_net_wm_type_normal   — ``_NET_WM_WINDOW_TYPE_NORMAL``
        atom_net_client_list      — ``_NET_CLIENT_LIST``
    """
    if not hasattr(_thread_local, "display"):
        _thread_local.display = xdisplay.Display()
        _thread_local.atom_net_wm_type = _thread_local.display.intern_atom(
            "_NET_WM_WINDOW_TYPE"
        )
        _thread_local.atom_net_wm_type_normal = _thread_local.display.intern_atom(
            "_NET_WM_WINDOW_TYPE_NORMAL"
        )
        _thread_local.atom_net_client_list = _thread_local.display.intern_atom(
            "_NET_CLIENT_LIST"
        )
    return _thread_local.display


# =============================================================================
# Window Registry
# =============================================================================

class WindowRegistry:
    """Tracks Genie-launched application windows and their debug ports.

    Registry entry schema (created by ``open_app`` in Prompt 2)::

        {
            "wid": str,              # validated X11 window ID string
            "process": object,       # subprocess.Popen object
            "wm_class": str,         # from xprop WM_CLASS
            "wm_name": str,          # from xprop WM_NAME
            "origin": "genie",       # always "genie" for Layer 1
            "cdp_port": int | None,  # runtime-assigned debug port, or None
            "tier": str,             # "cdp_primary" or "atspi"
            "time_to_window_ms": int # ms from Popen to validated WID
        }
    """

    def __init__(self) -> None:
        # Application registry — label → entry dict
        self._registry: dict = {}
        self._registry_lock: threading.Lock = threading.Lock()

        # Port ledger — tier string → last assigned port
        # -1 offset so first _assign_port() call returns the exact base port.
        self._port_ledger: dict = {
            "cdp": CDP_BASE_PORT - 1,
            "electron": ELECTRON_BASE_PORT - 1,
        }
        self._port_lock: threading.Lock = threading.Lock()

        # Background poller infrastructure (built in Prompt 3)
        self._event_queue: queue.Queue = queue.Queue()
        self._stop_event: threading.Event = threading.Event()
        self._poller_thread: threading.Thread | None = None

    # ----- Port Assignment ---------------------------------------------------

    def clear(self) -> None:
        """Remove all registered apps from the registry.

        Called by the test runner between tests to avoid stale WID / CDP port
        entries from a previous test polluting the next one.
        Port ledger is intentionally preserved so ports keep incrementing and
        never collide across tests in the same process.
        """
        with self._registry_lock:
            self._registry.clear()

    def _assign_port(self, tier: str) -> int:
        """Assign the next free debug port for *tier*.

        Args:
            tier: ``"cdp"`` or ``"electron"``.

        Returns:
            An available TCP port number.

        Raises:
            ValueError: If *tier* is not ``"cdp"`` or ``"electron"``.
            LaunchFailureError: If no free port exists within the scan range.
        """
        if tier not in self._port_ledger:
            raise ValueError(
                f"Unknown port tier '{tier}'. "
                f"Expected one of: {sorted(self._port_ledger.keys())}"
            )

        with self._port_lock:
            candidate = self._port_ledger[tier] + 1

            while candidate <= CDP_PORT_SCAN_MAX:
                occupied = False
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock.settimeout(0.2)
                    result = sock.connect_ex(("127.0.0.1", candidate))
                    if result == 0:
                        # Port is in use — try the next one.
                        occupied = True
                finally:
                    sock.close()

                if occupied:
                    candidate += 1
                    continue

                # Found a free port — record and return.
                self._port_ledger[tier] = candidate
                return candidate

            # Exhausted the scan range.
            raise LaunchFailureError(
                app_name="port_ledger",
                attempts=1,
                last_reason=f"no free port found in range for tier {tier}",
            )

    # ----- CDP Profile Lock Cleanup ------------------------------------------

    def _cleanup_cdp_profile(self, app_name: str, profile_path: str) -> None:
        """Kill stale processes holding *profile_path* and remove lock files.

        Called by ``open_app`` before every CDP/Electron launch to ensure
        a clean profile directory.  The sequence is:

        1. Scan running processes for one whose cmdline contains *profile_path*.
        2. Collect the full process tree rooted at the match.
        3. SIGKILL children first (leaves→root), then the parent.
        4. Wait for all processes to exit (bounded by
           ``LOCK_CLEANUP_WAIT_TIMEOUT``).
        5. Delete browser/electron lock files from *profile_path*.

        Args:
            app_name:     Binary name (e.g. ``"firefox"``, ``"code"``).
            profile_path: Absolute path to the profile directory.
        """

        # -- Step 1: find stale process holding the profile open --------------
        matched_pid: int | None = None

        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"] or []
                if profile_path in cmdline:
                    matched_pid = proc.info["pid"]
                    break
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                continue
            except psutil.ZombieProcess:
                continue

        if matched_pid is None:
            log(
                f"CDP cleanup: no stale process found for profile "
                f"{profile_path} — proceeding"
            )
            return

        # -- Step 2: collect full process tree --------------------------------
        try:
            matched_process = psutil.Process(matched_pid)
            procs_to_kill = [matched_process] + matched_process.children(
                recursive=True
            )
        except psutil.NoSuchProcess:
            log(
                f"CDP cleanup: matched process {matched_pid} exited before "
                f"tree collection — proceeding"
            )
            return

        # -- Step 3: SIGKILL children first, then parent ----------------------
        for child in procs_to_kill[1:]:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        try:
            procs_to_kill[0].kill()
        except psutil.NoSuchProcess:
            log(
                f"CDP cleanup: parent {matched_pid} already dead before "
                f"SIGKILL"
            )

        # -- Step 4: wait for processes to die --------------------------------
        gone, alive = psutil.wait_procs(
            procs_to_kill, timeout=LOCK_CLEANUP_WAIT_TIMEOUT
        )
        if alive:
            log(
                f"CDP cleanup: {len(alive)} process(es) still alive after "
                f"{LOCK_CLEANUP_WAIT_TIMEOUT}s — proceeding anyway. "
                f"PIDs: {[p.pid for p in alive]}"
            )

        # -- Step 5: delete lock files ----------------------------------------
        if app_name == "firefox":
            lock_files = ["lock", ".parentlock"]
        elif app_name in {"chromium", "chrome", "code", "slack", "discord"}:
            lock_files = ["SingletonLock", "SingletonCookie", "SingletonSocket"]
        else:
            lock_files = []

        for filename in lock_files:
            fpath = os.path.join(profile_path, filename)
            if not os.path.exists(fpath):
                continue
            try:
                os.remove(fpath)
            except OSError as exc:
                log(
                    f"CDP cleanup: failed to remove {fpath}: {exc}"
                )

        log(
            f"CDP cleanup: lock file cleanup complete for {app_name} "
            f"at {profile_path}"
        )

    # ----- _NET_CLIENT_LIST Snapshot -----------------------------------------

    def _snapshot_client_list(self) -> set[int]:
        """Return the current ``_NET_CLIENT_LIST`` as a set of WID ints.

        Used by ``open_app`` to detect new windows for apps not in
        ``APP_PROFILES`` (dynamic wm_class discovery).
        """
        try:
            d = _get_display()
            root = d.screen().root
            prop = root.get_full_property(
                _thread_local.atom_net_client_list,
                Xlib.X.AnyPropertyType,
            )
            if prop is not None:
                return set(prop.value)
        except Exception:
            pass
        return set()

    # ----- WID Disambiguation ------------------------------------------------

    def _disambiguate_wid(
        self, wids: list[str], start_time: float, timeout: float
    ) -> str | None:
        """Select the NORMAL-type window from a list of WID strings.

        Returns the first WID whose ``_NET_WM_WINDOW_TYPE`` contains
        ``_NET_WM_WINDOW_TYPE_NORMAL``.  If no candidate has the atom
        set yet (compositor lag), applies a time-based fallback:

        - Before 50% of *timeout* has elapsed: return ``None`` so the
          caller retries on the next poll cycle.
        - After 50%: fall back to largest-area heuristic via
          ``xdotool getwindowgeometry``.

        Args:
            wids:       WID strings from ``xdotool search --pid``.
            start_time: ``time.monotonic()`` captured at Popen time.
            timeout:    Total polling budget for this attempt (seconds).

        Returns:
            A WID string, or ``None`` if no suitable window is ready yet.
        """
        if not wids:
            return None

        display = _get_display()

        for wid_str in wids:
            try:
                wid_int = int(wid_str)
                win = display.create_resource_object("window", wid_int)
                prop = win.get_full_property(
                    _thread_local.atom_net_wm_type, Xlib.X.AnyPropertyType
                )
                if (
                    prop is not None
                    and _thread_local.atom_net_wm_type_normal in prop.value
                ):
                    return wid_str
            except (Xlib.error.XError, Xlib.error.BadWindow, Exception):
                continue

        # No candidate had _NET_WM_WINDOW_TYPE_NORMAL set yet.
        elapsed = time.monotonic() - start_time
        half_timeout = timeout / 2

        if elapsed < half_timeout:
            return None

        # Fallback: largest area heuristic.
        best_wid: str | None = None
        best_area = -1

        for wid_str in wids:
            try:
                result = subprocess.run(
                    ["xdotool", "getwindowgeometry", wid_str],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode != 0:
                    continue
                # Output: "Window <wid>\n  Position: ...\n  Geometry: WxH"
                for line in result.stdout.splitlines():
                    line_stripped = line.strip()
                    if line_stripped.startswith("Geometry:"):
                        geom = line_stripped.split(":", 1)[1].strip()
                        w_str, h_str = geom.split("x")
                        area = int(w_str) * int(h_str)
                        if area > best_area:
                            best_area = area
                            best_wid = wid_str
            except Exception:
                continue

        return best_wid if best_area > 0 else None

    # ----- Application Launch ------------------------------------------------

    def _try_adopt_existing(self, app_name: str, label: str) -> str | None:
        """Attempt to adopt an already-running window for app_name.

        Used when a singleton app exits immediately on launch (code 0),
        indicating an existing instance handled the request.

        Searches by wm_class from APP_PROFILES. Validates via xprop,
        registers under label, and returns the WID. Returns None if no
        adoptable window found.
        """
        wm_class = APP_PROFILES.get(app_name, {}).get("wm_class")
        if not wm_class:
            return None

        try:
            result = subprocess.run(
                ["xdotool", "search", "--class", wm_class],
                capture_output=True, text=True, timeout=5,
            )
            wids = [w.strip() for w in result.stdout.splitlines() if w.strip()]
        except Exception:
            return None

        if not wids:
            return None

        candidate_wid = wids[0]

        # Validate via xprop
        try:
            xprop_result = subprocess.run(
                ["xprop", "-id", candidate_wid, "WM_CLASS", "WM_NAME"],
                capture_output=True, text=True, timeout=5,
            )
            if xprop_result.returncode != 0:
                return None
        except Exception:
            return None

        wm_class_found = ""
        wm_name = ""
        for xline in xprop_result.stdout.splitlines():
            if "WM_CLASS" in xline and "=" in xline:
                parts = xline.split("=", 1)[1].strip()
                quoted = [s.strip().strip('"') for s in parts.split(",")]
                wm_class_found = quoted[-1] if quoted else ""
            elif "WM_NAME" in xline and "=" in xline:
                raw_name = xline.split("=", 1)[1].strip()
                wm_name = raw_name.strip('"')

        entry = {
            "wid": candidate_wid,
            "process": None,
            "wm_class": wm_class_found,
            "wm_name": wm_name,
            "origin": "adopted",
            "cdp_port": APP_PROFILES.get(app_name, {}).get("cdp_port"),
            "tier": APP_PROFILES.get(app_name, {}).get("tier", "atspi"),
            "time_to_window_ms": 0,
        }

        with self._registry_lock:
            self._registry[label] = entry

        log(
            f"open_app: adopted existing window for '{app_name}' as '{label}', "
            f"WID={candidate_wid}, wm_class='{wm_class_found}'"
        )
        return candidate_wid

    def _try_adopt_unknown(
        self, app_name: str, label: str, pre_launch_wids: set[int]
    ) -> str | None:
        """Adopt a window for an unknown app not in ``APP_PROFILES``.

        Fallback for ``_try_adopt_existing`` when ``wm_class`` is not
        pre-configured.  Two strategies, tried in order:

        1. ``_NET_CLIENT_LIST`` diff — detect windows that appeared after
           the pre-launch snapshot.
        2. Name-based search — ``xdotool search --name`` using *app_name*
           as substring. Covers the singleton/DBus-activation case where
           the window existed before launch.

        Returns:
            WID string on success, ``None`` on failure.
        """
        # Strategy 1: client list diff
        current_wids = self._snapshot_client_list()
        new_wids = current_wids - pre_launch_wids
        candidates = [str(w) for w in new_wids]

        # Strategy 2: name substring search (singleton activation)
        if not candidates:
            try:
                result = subprocess.run(
                    ["xdotool", "search", "--name", app_name],
                    capture_output=True, text=True, timeout=5,
                )
                candidates = [
                    w.strip() for w in result.stdout.splitlines()
                    if w.strip()
                ]
            except Exception:
                pass

        if not candidates:
            return None

        candidate_wid = candidates[0]

        # Validate via xprop
        try:
            xprop_result = subprocess.run(
                ["xprop", "-id", candidate_wid, "WM_CLASS", "WM_NAME"],
                capture_output=True, text=True, timeout=5,
            )
            if xprop_result.returncode != 0:
                return None
        except Exception:
            return None

        wm_class_found = ""
        wm_name = ""
        for xline in xprop_result.stdout.splitlines():
            if "WM_CLASS" in xline and "=" in xline:
                parts = xline.split("=", 1)[1].strip()
                quoted = [s.strip().strip('"') for s in parts.split(",")]
                wm_class_found = quoted[-1] if quoted else ""
            elif "WM_NAME" in xline and "=" in xline:
                raw_name = xline.split("=", 1)[1].strip()
                wm_name = raw_name.strip('"')

        entry = {
            "wid": candidate_wid,
            "process": None,
            "wm_class": wm_class_found,
            "wm_name": wm_name,
            "origin": "adopted",
            "cdp_port": None,
            "tier": "atspi",
            "time_to_window_ms": 0,
        }

        with self._registry_lock:
            self._registry[label] = entry

        log(
            f"open_app: adopted unknown app '{app_name}' as '{label}', "
            f"WID={candidate_wid}, wm_class='{wm_class_found}'"
        )
        return candidate_wid

    def open_app(self, app_name: str, label: str | None = None) -> str:
        """Launch an application, poll for its WID, validate, and register.

        Blocks until a validated WID is registered or all retry attempts
        are exhausted.

        Args:
            app_name: Binary name (must be on ``$PATH`` or in
                      ``APP_PROFILES``).
            label:    Registry label.  Auto-derived as ``<app_name>_N``
                      if ``None``.

        Returns:
            The validated WID string.

        Raises:
            LaunchFailureError: After ``MAX_OPEN_APP_RETRIES`` failures.
        """

        # -- Label assignment -------------------------------------------------
        if label is None:
            with self._registry_lock:
                n = 1
                while f"{app_name}_{n}" in self._registry:
                    n += 1
                label = f"{app_name}_{n}"

        # -- CDP port assignment (once, before retry loop) --------------------
        profile = APP_PROFILES.get(app_name)
        launch_tier = profile["launch_tier"] if profile else None
        is_cdp = launch_tier in ("cdp", "electron")
        assigned_port: int | None = None

        if is_cdp:
            port_tier = "cdp" if launch_tier == "cdp" else "electron"
            assigned_port = self._assign_port(port_tier)

        # -- CDP lock cleanup (once, before retry loop) ----------------------
        if is_cdp and profile is not None:
            flags = profile["launch_flags"]
            profile_path: str | None = None
            for flag in flags:
                if flag.startswith("--user-data-dir="):
                    profile_path = flag.split("=", 1)[1]
                    break
                if flag.startswith("--profile="):
                    profile_path = flag.split("=", 1)[1]
                    break
            if profile_path is not None:
                os.makedirs(profile_path, exist_ok=True)
                self._cleanup_cdp_profile(app_name, profile_path)

        # -- Build base command -----------------------------------------------
        base_cmd: list[str] = [profile["binary"] if profile else app_name]
        if profile is not None:
            base_cmd.extend(profile["launch_flags"])
        if is_cdp and assigned_port is not None:
            base_cmd.append(f"--remote-debugging-port={assigned_port}")

        # -- Outer retry loop -------------------------------------------------
        last_reason = "unknown"
        pre_launch_wids: set[int] = set()

        for attempt in range(MAX_OPEN_APP_RETRIES):
            # Snapshot _NET_CLIENT_LIST before launch (unknown app fallback)
            if profile is None:
                pre_launch_wids = self._snapshot_client_list()

            # Fresh launch on every attempt
            try:
                process = subprocess.Popen(
                    base_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except FileNotFoundError:
                raise LaunchFailureError(
                    app_name, attempt + 1, f"binary not found: {base_cmd[0]}"
                )

            pid = process.pid
            start_time = time.monotonic()

            tier_key = (
                APP_PROFILES.get(app_name, {})
                .get("launch_tier", "default")
            )
            base_timeout = LAUNCH_TIMEOUTS.get(
                tier_key, LAUNCH_TIMEOUTS["default"]
            )
            timeout = base_timeout * RETRY_TIMEOUT_MULTIPLIERS[attempt]

            log(
                f"open_app: attempt {attempt + 1}/{MAX_OPEN_APP_RETRIES} "
                f"for {app_name} (PID {pid}, timeout {timeout:.1f}s)"
            )

            validated_wid: str | None = None

            # -- Inner polling loop -------------------------------------------
            while time.monotonic() - start_time < timeout:
                # Fast-fail: process already exited?
                if process.poll() is not None:
                    last_reason = (
                        f"process exited immediately "
                        f"(code {process.poll()})"
                    )
                    # Singleton adoption: code 0 on attempt 0 means an existing
                    # instance handled the launch. Try to adopt its window.
                    if process.poll() == 0 and attempt == 0:
                        adopted_wid = self._try_adopt_existing(app_name, label)
                        if adopted_wid is not None:
                            return adopted_wid
                        # Unknown app fallback: client-list diff + name search
                        if profile is None:
                            adopted_wid = self._try_adopt_unknown(
                                app_name, label, pre_launch_wids
                            )
                            if adopted_wid is not None:
                                return adopted_wid
                    break

                # Parent-first WID search via xdotool search --pid
                wids: list[str] = []
                try:
                    result = subprocess.run(
                        ["xdotool", "search", "--pid", str(pid)],
                        capture_output=True, text=True, timeout=5,
                    )
                    wids = [
                        w.strip() for w in result.stdout.splitlines()
                        if w.strip()
                    ]
                except Exception:
                    pass

                # If no WIDs from parent, search children
                if not wids:
                    try:
                        children = psutil.Process(pid).children(recursive=True)
                    except psutil.NoSuchProcess:
                        children = []
                    except psutil.AccessDenied:
                        children = []

                    for child in children:
                        try:
                            cresult = subprocess.run(
                                [
                                    "xdotool", "search", "--pid",
                                    str(child.pid),
                                ],
                                capture_output=True, text=True, timeout=5,
                            )
                            child_wids = [
                                w.strip()
                                for w in cresult.stdout.splitlines()
                                if w.strip()
                            ]
                            if child_wids:
                                wids = child_wids
                                break
                        except psutil.NoSuchProcess:
                            continue
                        except psutil.AccessDenied:
                            continue
                        except Exception:
                            continue

                # Fallback for unknown apps: _NET_CLIENT_LIST diff
                if not wids and profile is None and pre_launch_wids:
                    current_client_wids = self._snapshot_client_list()
                    diff_wids = current_client_wids - pre_launch_wids
                    if diff_wids:
                        wids = [str(w) for w in diff_wids]
                        log(
                            f"WID poll: _NET_CLIENT_LIST diff found "
                            f"{len(diff_wids)} new window(s) for "
                            f"unknown app {app_name}"
                        )

                if not wids:
                    log(
                        f"WID poll: xdotool search --pid returned no WIDs "
                        f"for PID {pid} — app may not set _NET_WM_PID"
                    )
                    time.sleep(WID_POLL_INTERVAL)
                    continue

                # Disambiguate among candidate WIDs
                candidate_wid = self._disambiguate_wid(
                    wids, start_time, timeout
                )
                if candidate_wid is None:
                    time.sleep(WID_POLL_INTERVAL)
                    continue

                # -- WID Validation Step 1: _NET_CLIENT_LIST zombie guard -----
                try:
                    display = _get_display()
                    root = display.screen().root
                    client_prop = root.get_full_property(
                        _thread_local.atom_net_client_list,
                        Xlib.X.AnyPropertyType,
                    )
                    if client_prop is None:
                        time.sleep(WID_POLL_INTERVAL)
                        continue
                    if int(candidate_wid) not in client_prop.value:
                        time.sleep(WID_POLL_INTERVAL)
                        continue
                except Exception:
                    time.sleep(WID_POLL_INTERVAL)
                    continue

                # -- WID Validation Step 2: non-zero geometry -----------------
                try:
                    geom_result = subprocess.run(
                        ["xdotool", "getwindowgeometry", candidate_wid],
                        capture_output=True, text=True, timeout=5,
                    )
                    if geom_result.returncode != 0:
                        time.sleep(WID_POLL_INTERVAL)
                        continue
                    geom_ok = False
                    for gline in geom_result.stdout.splitlines():
                        gline_s = gline.strip()
                        if gline_s.startswith("Geometry:"):
                            geom_str = gline_s.split(":", 1)[1].strip()
                            gw, gh = geom_str.split("x")
                            if int(gw) > 0 and int(gh) > 0:
                                geom_ok = True
                    if not geom_ok:
                        time.sleep(WID_POLL_INTERVAL)
                        continue
                except Exception:
                    time.sleep(WID_POLL_INTERVAL)
                    continue

                # -- WID Validation Step 3: xprop wm_class / wm_name ---------
                try:
                    xprop_result = subprocess.run(
                        ["xprop", "-id", candidate_wid, "WM_CLASS", "WM_NAME"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if xprop_result.returncode != 0:
                        # Window died after geometry confirmed — crash.
                        last_reason = (
                            f"xprop failed after geometry confirmed "
                            f"(WID {candidate_wid})"
                        )
                        break  # trigger kill-and-retry
                except Exception as exc:
                    last_reason = (
                        f"xprop error after geometry confirmed: {exc}"
                    )
                    break  # trigger kill-and-retry

                wm_class = ""
                wm_name = ""
                for xline in xprop_result.stdout.splitlines():
                    if "WM_CLASS" in xline and "=" in xline:
                        # WM_CLASS(STRING) = "instance", "class"
                        parts = xline.split("=", 1)[1].strip()
                        # Extract the quoted strings
                        quoted = [
                            s.strip().strip('"')
                            for s in parts.split(",")
                        ]
                        wm_class = quoted[-1] if quoted else ""
                    elif "WM_NAME" in xline and "=" in xline:
                        raw_name = xline.split("=", 1)[1].strip()
                        wm_name = raw_name.strip('"')

                # -- Registration ---------------------------------------------
                time_to_window_ms = int(
                    (time.monotonic() - start_time) * 1000
                )
                entry = {
                    "wid": candidate_wid,
                    "process": process,
                    "wm_class": wm_class,
                    "wm_name": wm_name,
                    "origin": "genie",
                    "cdp_port": assigned_port,
                    "tier": APP_PROFILES.get(app_name, {}).get(
                        "tier", "atspi"
                    ),
                    "time_to_window_ms": time_to_window_ms,
                }

                with self._registry_lock:
                    self._registry[label] = entry

                log(
                    f"open_app: {app_name} registered as '{label}', "
                    f"WID={candidate_wid}, "
                    f"time_to_window={time_to_window_ms}ms"
                )
                # Raise Chrome (or any app) window to foreground so that
                # subsequent press_key / type_text go to the right window.
                # Without this, the terminal/IDE that launched Genie keeps
                # keyboard focus and every keypress lands in the wrong app.
                try:
                    subprocess.run(
                        ["xdotool", "windowactivate", "--sync",
                         candidate_wid],
                        capture_output=True,
                        timeout=5,
                    )
                except Exception as _wa_exc:
                    log(f"open_app: windowactivate warning: {_wa_exc}")
                return candidate_wid

            # -- Inner loop exited without success: kill-and-retry ------------
            if last_reason == "unknown":
                last_reason = f"polling timeout after {timeout:.1f}s"
            process.kill()
            try:
                process.wait(timeout=PROCESS_WAIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                log(
                    f"kill-and-retry: process {pid} did not exit within "
                    f"{PROCESS_WAIT_TIMEOUT}s after SIGKILL (D-state) "
                    f"— proceeding"
                )

            with self._registry_lock:
                self._registry.pop(label, None)

            log(
                f"open_app: attempt {attempt + 1} failed for {app_name}: "
                f"{last_reason}"
            )

        # All retries exhausted.
        raise LaunchFailureError(app_name, MAX_OPEN_APP_RETRIES, last_reason)

    # ----- Registry Lookup ---------------------------------------------------

    def get_wid(self, label: str) -> str | None:
        """Return the WID for *label*, or ``None`` if not registered."""
        with self._registry_lock:
            entry = self._registry.get(label)
        return entry["wid"] if entry else None

    # ----- Window Focus ------------------------------------------------------

    def focus_window(self, label: str) -> bool:
        """Bring the window registered under *label* to front.

        Returns ``True`` on success, ``False`` if the label is unknown
        or the xdotool command fails.  On timeout, attempts WID
        re-discovery via wm_class search before giving up.
        """
        wid = self.get_wid(label)
        if wid is None:
            return False
        try:
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", wid],
                capture_output=True, timeout=FOCUS_WINDOW_TIMEOUT,
            )
            return True
        except subprocess.TimeoutExpired:
            # WID may be stale — try to re-discover via wm_class
            new_wid = self._rediscover_wid(label)
            if new_wid and new_wid != wid:
                try:
                    subprocess.run(
                        ["xdotool", "windowactivate", "--sync", new_wid],
                        capture_output=True, timeout=FOCUS_WINDOW_TIMEOUT,
                    )
                    return True
                except Exception:
                    pass
            # Last resort: try without --sync (fire-and-forget)
            try:
                subprocess.run(
                    ["xdotool", "windowactivate", wid],
                    capture_output=True, timeout=3,
                )
                time.sleep(0.3)
                return True
            except Exception:
                pass
            log(f"focus_window: failed for '{label}' (WID {wid}): timeout after re-discovery")
            return False
        except Exception as exc:
            log(f"focus_window: failed for '{label}' (WID {wid}): {exc}")
            return False

    def _rediscover_wid(self, label: str) -> str | None:
        """Try to find the WID for a registered app using wm_class search.

        On success, updates the registry entry with the new WID.
        Returns the new WID string or None.
        """
        with self._registry_lock:
            entry = self._registry.get(label)
        if entry is None:
            return None
        wm_class = entry.get("wm_class", "")
        if not wm_class:
            return None
        try:
            result = subprocess.run(
                ["xdotool", "search", "--class", wm_class],
                capture_output=True, text=True, timeout=3,
            )
            wids = result.stdout.strip().split()
            if wids:
                new_wid = wids[0]
                if new_wid != entry["wid"]:
                    log(f"_rediscover_wid: found new WID {new_wid} for "
                        f"'{label}' (was {entry['wid']})")
                    with self._registry_lock:
                        if label in self._registry:
                            self._registry[label]["wid"] = new_wid
                return new_wid
        except Exception:
            pass
        return None

    # ----- Background Poller -------------------------------------------------

    def start_poller(self) -> None:
        """Start the background window-state poller thread.

        Guards against double-start: if the poller thread is already
        running, logs a warning and returns immediately.
        """
        if self._poller_thread is not None and self._poller_thread.is_alive():
            log("start_poller: poller thread is already running — skipping")
            return

        self._stop_event.clear()
        self._poller_thread = threading.Thread(
            target=self._poll_loop, daemon=True
        )
        self._poller_thread.start()
        log("start_poller: background poller started")

    def stop_poller(self) -> None:
        """Signal the poller thread to stop and wait for it to exit."""
        self._stop_event.set()

        if self._poller_thread is not None:
            self._poller_thread.join(timeout=5)
            if self._poller_thread.is_alive():
                log(
                    "stop_poller: poller thread did not exit within 5s "
                    "— proceeding regardless"
                )

        self._poller_thread = None
        log("stop_poller: background poller stopped")

    def _poll_loop(self) -> None:
        """Background loop that detects closed/crashed windows.

        Runs on a daemon thread with its own thread-local X11 Display
        connection (via ``_get_display()``).  Each cycle:

        1. Check ``_stop_event``.
        2. Sleep ``WID_POLL_INTERVAL``.
        3. Snapshot the registry (lock held only during copy).
        4. Fetch ``_NET_CLIENT_LIST`` once via python-xlib.
        5. Emit events for windows no longer in the client list.
        """
        while True:
            try:
                # Step 1: clean exit check
                if self._stop_event.is_set():
                    return

                # Step 2: sleep
                time.sleep(WID_POLL_INTERVAL)

                # Step 3: snapshot registry
                with self._registry_lock:
                    snapshot = [
                        (label, entry["wid"], entry["process"])
                        for label, entry in self._registry.items()
                    ]

                # Step 4: empty snapshot — nothing to check
                if not snapshot:
                    continue

                # Step 5: fetch _NET_CLIENT_LIST once
                try:
                    display = _get_display()
                    root = display.screen().root
                    client_prop = root.get_full_property(
                        _thread_local.atom_net_client_list,
                        Xlib.X.AnyPropertyType,
                    )
                    if client_prop is None:
                        log(
                            "_poll_loop: _NET_CLIENT_LIST returned None "
                            "— skipping cycle"
                        )
                        continue
                except Exception as exc:
                    log(
                        f"_poll_loop: failed to fetch _NET_CLIENT_LIST: "
                        f"{exc} — skipping cycle"
                    )
                    continue

                # Step 6: build set of live WID ints
                client_wids: set[int] = set(client_prop.value)

                # Step 7: check each registered window
                for label, wid_str, process in snapshot:
                    wid_int = int(wid_str)
                    if wid_int in client_wids:
                        continue

                    # Window is gone — determine status from exit code
                    rc = process.poll()
                    if rc is None:
                        status = "closed"
                        exit_code = None
                    elif rc == 0:
                        status = "closed"
                        exit_code = 0
                    else:
                        status = "crashed"
                        exit_code = rc

                    event = {
                        "label": label,
                        "wid": wid_str,
                        "status": status,
                        "exit_code": exit_code,
                    }
                    self._event_queue.put(event)

                    # Remove from registry immediately to prevent
                    # duplicate events on subsequent poll cycles.
                    with self._registry_lock:
                        self._registry.pop(label, None)

                    log(
                        f"_poll_loop: event — label={label}, "
                        f"status={status}, exit_code={exit_code}"
                    )

            except Exception as exc:
                log(f"_poll_loop: unhandled exception: {exc} — continuing")
                continue

    def pump_events(self) -> list[dict]:
        """Non-blocking drain of the event queue.

        For each consumed event, removes the corresponding label from
        the registry (if still present).  Returns the list of event
        dicts consumed.
        """
        consumed: list[dict] = []
        while True:
            try:
                event = self._event_queue.get_nowait()
                consumed.append(event)
            except queue.Empty:
                break

        for event in consumed:
            with self._registry_lock:
                removed = self._registry.pop(event["label"], None)
            log(
                f"pump_events: removed '{event['label']}' from registry "
                f"(found={removed is not None})"
            )

        return consumed


# =============================================================================
# Phase A Validation Harness
# =============================================================================

if __name__ == "__main__":
    import sys

    reg = WindowRegistry()

    # -- Scenario 4: Clean profile no-op ------------------------------------
    log("--- Scenario 4: CDP cleanup with no stale process ---")
    reg._cleanup_cdp_profile("firefox", "/tmp/genie_fx_profile")
    log("Scenario 4 passed.\n")

    # -- Scenario 1: Firefox launch ----------------------------------------
    log("--- Scenario 1: Firefox launch ---")
    try:
        wid_fx = reg.open_app("firefox", label="test_firefox")
        entry_fx = reg._registry["test_firefox"]
        log(f"  WID:               {wid_fx}")
        log(f"  wm_class:          {entry_fx['wm_class']}")
        log(f"  wm_name:           {entry_fx['wm_name']}")
        log(f"  cdp_port:          {entry_fx['cdp_port']}")
        log(f"  tier:              {entry_fx['tier']}")
        log(f"  time_to_window_ms: {entry_fx['time_to_window_ms']}")
        assert isinstance(wid_fx, str) and wid_fx, "WID must be a non-empty string"
        assert entry_fx["cdp_port"] == CDP_BASE_PORT, (
            f"Expected port {CDP_BASE_PORT}, got {entry_fx['cdp_port']}"
        )
        assert entry_fx["tier"] == "cdp_primary"
        log("  focus_window test...")
        assert reg.focus_window("test_firefox"), "focus_window failed"
        log("Scenario 1 passed.\n")
    except LaunchFailureError as exc:
        log(f"Scenario 1 FAILED: {exc}")
        sys.exit(1)

    # -- Scenario 2: Electron tier (VS Code) --------------------------------
    log("--- Scenario 2: VS Code Electron launch ---")
    try:
        wid_obs = reg.open_app("obsidian", label="test_obsidian")
        entry_obs = reg._registry["test_obsidian"]
        log(f"  WID:               {wid_obs}")
        log(f"  wm_class:          {entry_obs['wm_class']}")
        log(f"  cdp_port:          {entry_obs['cdp_port']}")
        log(f"  tier:              {entry_obs['tier']}")
        log(f"  time_to_window_ms: {entry_obs['time_to_window_ms']}")
        assert isinstance(wid_obs, str) and wid_obs
        assert entry_obs["cdp_port"] >= ELECTRON_BASE_PORT
        assert entry_obs["tier"] == "cdp_primary"
        log("Scenario 2 passed.\n")
    except LaunchFailureError as exc:
        log(f"Scenario 2 FAILED: {exc}")
        sys.exit(1)

    # -- Scenario 5: atspi-tier app (gedit) WID-targeted input ---------------
    log("--- Scenario 5: gedit WID-targeted input ---")
    try:
        wid_gt = reg.open_app("gedit", label="test_gedit")
        entry_gt = reg._registry["test_gedit"]
        log(f"  WID:               {wid_gt}")
        log(f"  wm_class:          {entry_gt['wm_class']}")
        log(f"  cdp_port:          {entry_gt['cdp_port']}")
        log(f"  tier:              {entry_gt['tier']}")
        log(f"  time_to_window_ms: {entry_gt['time_to_window_ms']}")
        assert entry_gt["cdp_port"] is None
        assert entry_gt["tier"] == "atspi"
        # Type into gedit WITHOUT focusing it first
        subprocess.run(
            ["xdotool", "type", "--window", wid_gt, "--",
             "hello from genie"],
            timeout=5,
        )
        log("  Typed 'hello from genie' into gedit via WID-targeted input.")
        log("Scenario 5 passed.\n")
    except LaunchFailureError as exc:
        log(f"Scenario 5 FAILED: {exc}")
        sys.exit(1)

    # -- Start background poller for Scenario 3 -----------------------------
    reg.start_poller()

    # -- Scenario 3: Closed window detection --------------------------------
    log("--- Scenario 3: Closed window detection ---")
    try:
        process = reg._registry["test_gedit"]["process"]
        process.kill()
        process.wait(timeout=5)
        time.sleep(1.0)
        events = reg.pump_events()
        assert len(events) == 1, (
            f"Expected 1 event, got {len(events)}: {events}"
        )
        assert events[0]["label"] == "test_gedit", (
            f"Expected label 'test_gedit', got {events[0]['label']}"
        )
        assert events[0]["status"] == "crashed", (
            f"Expected status 'crashed', got {events[0]['status']}"
        )
        assert events[0]["exit_code"] is not None and events[0]["exit_code"] != 0, (
            f"Expected non-zero exit_code, got {events[0]['exit_code']}"
        )
        assert "test_gedit" not in reg._registry, (
            "test_gedit still in registry after pump_events"
        )
        log("Scenario 3 passed.\n")
    except LaunchFailureError as exc:
        log(f"Scenario 3 FAILED: {exc}")
        sys.exit(1)

    reg.stop_poller()

    log("Phase A+B validation complete.\n")

    # -- Scenario 6: Unknown app launch (dynamic wm_class discovery) ---------
    log("--- Scenario 6: Unknown app launch (eog — not in APP_PROFILES) ---")
    try:
        wid_calc = reg.open_app("eog", label="test_eog")
        entry_calc = reg._registry["test_eog"]
        log(f"  WID:               {wid_calc}")
        log(f"  wm_class:          {entry_calc['wm_class']}")
        log(f"  wm_name:           {entry_calc['wm_name']}")
        log(f"  cdp_port:          {entry_calc['cdp_port']}")
        log(f"  tier:              {entry_calc['tier']}")
        log(f"  time_to_window_ms: {entry_calc['time_to_window_ms']}")
        assert isinstance(wid_calc, str) and wid_calc, "WID must be a non-empty string"
        assert entry_calc["cdp_port"] is None, (
            f"Expected cdp_port None for unknown app, got {entry_calc['cdp_port']}"
        )
        assert entry_calc["tier"] == "atspi", (
            f"Expected tier 'atspi', got {entry_calc['tier']}"
        )
        assert entry_calc["wm_class"], "wm_class must be dynamically discovered (non-empty)"
        log("Scenario 6 passed.\n")
    except LaunchFailureError as exc:
        log(f"Scenario 6 FAILED: {exc}")
        sys.exit(1)

    log("Phase A+B+C validation complete.")
