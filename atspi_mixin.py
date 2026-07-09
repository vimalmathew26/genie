"""AT-SPI resolution tier mixin for ElementResolver.

Extracted from element_resolver.py.  Mixed into ElementResolver
so that all self._atspi_* calls work unchanged.
"""

import subprocess
import time
from collections import deque

import pyatspi

from config import (
    ROLE_ALIASES,
    MAX_READ_ELEMENT_CHARS,
    READ_ELEMENT_TRUNCATION_MARKER,
)
from exceptions import (
    EnvironmentalError,
    TransientError,
)


class ATSpiMixin:
    """AT-SPI element resolution methods."""

    # -----------------------------------------------------------------
    # Private helpers — AT-SPI resolution
    # -----------------------------------------------------------------

    def _find_atspi_window(self, label: str):
        """Three-filter PID-to-AT-SPI window bridge.

        Filter 1: PID match against desktop application nodes.
        Filter 2: Full bounding-box correlation (±20px tolerance).
        Filter 3: wm_name substring tiebreaker (only if Filter 2 ambiguous).
        """
        reg_entry = self.registry._registry[label]
        target_pid = reg_entry["process"].pid
        wid = reg_entry["wid"]

        # --- Filter 1: PID match with retry ---
        app_node = None
        for attempt in range(3):
            try:
                desktop = pyatspi.Registry.getDesktop(0)
                for i in range(desktop.childCount):
                    child = desktop.getChildAtIndex(i)
                    if child is not None and child.get_process_id() == target_pid:
                        app_node = child
                        break
            except Exception:
                pass  # bus hiccup — retry
            if app_node is not None:
                break
            time.sleep(0.5)

        if app_node is None:
            raise TransientError("atspi tree not ready")

        # --- Filter 2: Full bounding-box correlation ---
        try:
            result = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", str(wid)],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                raise EnvironmentalError(
                    f"xdotool getwindowgeometry failed for wid={wid}"
                )
            wid_x = wid_y = wid_w = wid_h = None
            for line in result.stdout.splitlines():
                if line.startswith("X="):
                    wid_x = int(line.split("=", 1)[1])
                elif line.startswith("Y="):
                    wid_y = int(line.split("=", 1)[1])
                elif line.startswith("WIDTH="):
                    wid_w = int(line.split("=", 1)[1])
                elif line.startswith("HEIGHT="):
                    wid_h = int(line.split("=", 1)[1])
            if None in (wid_x, wid_y, wid_w, wid_h):
                raise EnvironmentalError(
                    f"failed to parse xdotool geometry for wid={wid}"
                )
        except subprocess.TimeoutExpired:
            raise EnvironmentalError(
                f"xdotool getwindowgeometry timed out for wid={wid}"
            )

        candidates = []
        for i in range(app_node.childCount):
            win = app_node.getChildAtIndex(i)
            if win is None:
                continue
            try:
                ext = win.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                atspi_x, atspi_y = ext.x, ext.y
                atspi_w, atspi_h = ext.width, ext.height
            except Exception:
                continue
            if (abs(wid_x - atspi_x) <= 20
                    and abs(wid_y - atspi_y) <= 20
                    and abs(wid_w - atspi_w) <= 20
                    and abs(wid_h - atspi_h) <= 20):
                candidates.append(win)

        # --- Filter 3: wm_name tiebreaker (only if 2+ candidates) ---
        if len(candidates) >= 2:
            registry_wm_name = reg_entry.get("wm_name", "")
            for c in candidates:
                atspi_window_name = getattr(c, 'name', '')
                if (registry_wm_name in atspi_window_name
                        or atspi_window_name in registry_wm_name):
                    return c
            raise EnvironmentalError(
                f"ambiguous atspi window for label={label}"
            )

        if len(candidates) == 1:
            return candidates[0]

        raise EnvironmentalError(
            f"atspi window not found for label={label}"
        )

    def _bfs_atspi(self, window_node, role: str, name: str) -> list:
        """BFS all-matches from confirmed AT-SPI window node.

        Returns a list of every node where (role, name) both match.

        Special case — name=="":
          Role-only mode. Returns the FIRST node matching the role,
          regardless of name.  Implements "first <role>" semantics.

        Normal case — name!="":
          Returns all nodes where both role and name match exactly.
        """
        aliases = ROLE_ALIASES.get(role, [role])
        role_only = (name == "")
        matches = []
        queue = deque([window_node])

        while queue:
            node = queue.popleft()
            matched = False

            if node.getRole() == pyatspi.ROLE_UNKNOWN:
                # ROLE_UNKNOWN path (Qt apps): skip role filter,
                # match on name + STATE_SENSITIVE + STATE_ENABLED
                if role_only:
                    if (node.getState().contains(pyatspi.STATE_SENSITIVE)
                            and node.getState().contains(pyatspi.STATE_ENABLED)):
                        matched = True
                elif (node.name == name
                        and node.getState().contains(pyatspi.STATE_SENSITIVE)
                        and node.getState().contains(pyatspi.STATE_ENABLED)):
                    matched = True
            else:
                role_str = node.getRoleName()
                if role_only:
                    if role_str in aliases:
                        matched = True
                elif role_str in aliases and node.name == name:
                    matched = True

            if matched:
                if role_only:
                    return [node]  # first-match-by-role: return immediately
                matches.append(node)

            # Enqueue children
            for i in range(node.childCount):
                child = node.getChildAtIndex(i)
                if child is not None:
                    queue.append(child)

        return matches

    def _atspi_scroll_to(self, node) -> None:
        """Best-effort scroll element into viewport via AT-SPI scrollable.

        Never raises. Caller re-queries extents regardless of outcome.
        """
        try:
            parent = node.parent
            if parent is not None:
                parent.queryScrollable().scrollTo(
                    pyatspi.SCROLL_ANYWHERE
                )
        except Exception:
            pass

    def _atspi_click(self, node, role: str = "", name: str = "") -> None:
        """Execute click on a single confirmed BFS match.

        Step 1: doAction(0) primary.
        Step 2: extents + xdotool fallback.
        Raises EnvironmentalError if both steps are exhausted.
        """
        # Step 1 — doAction(0) primary
        try:
            action_iface = node.queryAction()
            if action_iface.get_nActions() > 0:
                action_iface.doAction(0)
                return
        except NotImplementedError:
            pass

        # Step 2 — extents + xdotool fallback
        try:
            extents = node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
        except NotImplementedError:
            raise EnvironmentalError("queryComponent not available")

        x, y, w, h = extents.x, extents.y, extents.width, extents.height
        if x == 0 and y == 0 and w == 0 and h == 0:
            # Attempt scroll before giving up
            self._atspi_scroll_to(node)
            extents = node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
            x, y, w, h = extents.x, extents.y, extents.width, extents.height
            if x == 0 and y == 0 and w == 0 and h == 0:
                raise EnvironmentalError(
                    "element has zero extents, cannot click"
                )

        click_x = x + w // 2
        click_y = y + h // 2
        self.controller.click(click_x, click_y)

    def _atspi_type(self, node, text: str) -> None:
        """Execute text injection on a single confirmed BFS match.

        Step 1: SetTextContents primary.
        Step 2: Focus + keyboardEvent fallback.
        Raises EnvironmentalError on failure.
        """
        # Step 1 — SetTextContents primary
        try:
            node.queryEditableText().setTextContents(text)
            return
        except NotImplementedError:
            pass
        except Exception:
            pass

        # Step 2 — Focus + keyboardEvent fallback
        # 2a. Mandatory focus step — extents check
        try:
            extents = node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
        except NotImplementedError:
            raise EnvironmentalError("queryComponent not available for focus step")

        x, y, w, h = extents.x, extents.y, extents.width, extents.height
        if x == 0 and y == 0 and w == 0 and h == 0:
            raise EnvironmentalError(
                "element has zero extents, cannot transfer focus for keyboard injection"
            )

        # 2b. Transfer OS keyboard focus via xdotool click
        self.controller.click(x + w // 2, y + h // 2)

        # 2c. keyboardEvent injection
        for char in text:
            pyatspi.Registry.generateKeyboardEvent(
                ord(char), None, pyatspi.KEY_SYM
            )

    def _truncate(self, text: str) -> str:
        """Apply character cap and truncation marker."""
        if len(text) <= MAX_READ_ELEMENT_CHARS:
            return text
        truncated = text[:MAX_READ_ELEMENT_CHARS]
        marker = READ_ELEMENT_TRUNCATION_MARKER.format(
            total_chars=len(text),
            limit=MAX_READ_ELEMENT_CHARS,
        )
        return truncated + marker

    def _atspi_read(self, node, role: str) -> str:
        """Role-aware text extraction from a confirmed single BFS match.

        Returns a raw string (not yet truncated).
        Raises EnvironmentalError when no readable content available.
        """
        # Group A — name-type roles
        if role in ("button", "link", "label", "image", "menuitem", "tab"):
            result = node.name or ""
            if not result:
                raise EnvironmentalError(
                    f"element has no accessible name: role={role}"
                )
            return result

        # Group B — textfield (queryText primary, no fallback)
        if role == "textfield":
            try:
                result = node.queryText().getText(0, -1)
                return result if result is not None else ""
            except NotImplementedError:
                raise EnvironmentalError(
                    "Text interface not available for textfield element"
                )

        # Group C — dropdown (queryText primary, node.name fallback)
        if role == "dropdown":
            try:
                result = node.queryText().getText(0, -1)
                return result if result is not None else ""
            except NotImplementedError:
                result = node.name or ""
                return result

        # Group D — checkbox, slider, spinbox (node.name + value)
        if role in ("checkbox", "slider", "spinbox"):
            name_part = node.name or ""
            value_part = ""
            try:
                value_iface = node.queryValue()
                current = value_iface.currentValue
                value_part = str(current)
            except NotImplementedError:
                pass
            if name_part and value_part:
                return f"{name_part}: {value_part}"
            return name_part or value_part or ""

        # Group E — ROLE_UNKNOWN / catch-all
        try:
            result = node.queryText().getText(0, -1)
            if result is not None and result != "":
                return result
        except NotImplementedError:
            pass
        result = node.name or ""
        return result
