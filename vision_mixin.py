"""Vision resolution tier mixin for ElementResolver.

Extracted from element_resolver.py.  Mixed into ElementResolver
so that all self._vision_* calls work unchanged.
Also includes viewport helpers and marker coordinate builders
used by the vision disambiguation flow.
"""

import json
import os
import re
import subprocess
import time
import base64
import urllib.request
from collections import deque

import pyatspi

from config import (
    log,
    OPENROUTER_API_KEY,
    VISION_MODEL,
    LOG_DIR,
)
from exceptions import (
    EnvironmentalError,
    ResourceError,
    TransientError,
)


class VisionMixin:
    """Vision model element resolution and viewport/marker helpers."""

    _no_vision_log: bool = False  # set to True to suppress vision_training.jsonl writes

    # -----------------------------------------------------------------
    # Viewport helpers (AT-SPI queries for browser viewport coords)
    # -----------------------------------------------------------------

    def _get_viewport_origin(self, app: str) -> tuple[int, int]:
        """Query AT-SPI for the absolute screen coordinates of the CDP
        viewport origin (top-left of the renderable page area)."""
        reg_entry = self.registry._registry[app]
        target_pid = reg_entry["process"].pid

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
                pass
            if app_node is not None:
                break
            time.sleep(0.5)

        if app_node is None:
            raise TransientError("atspi tree not ready")

        queue = deque([app_node])
        while queue:
            node = queue.popleft()
            if node.getRoleName() in ("document web", "internal frame", "pane"):
                try:
                    ext = node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                except NotImplementedError:
                    raise TransientError(
                        f"document web node has no Component interface for app={app}"
                    )
                return (ext.x, ext.y)
            for i in range(node.childCount):
                child = node.getChildAtIndex(i)
                if child is not None:
                    queue.append(child)

        raise TransientError(
            f"document web node not found for app={app} "
            f"— browser accessibility tree may not be fully rendered"
        )

    def _get_viewport_extents(self, app: str) -> tuple[int, int, int, int]:
        """Query AT-SPI for the absolute screen coordinates and dimensions
        of the CDP viewport (renderable page area)."""
        reg_entry = self.registry._registry[app]
        target_pid = reg_entry["process"].pid

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
                pass
            if app_node is not None:
                break
            time.sleep(0.5)

        if app_node is None:
            raise TransientError("atspi tree not ready")

        queue = deque([app_node])
        while queue:
            node = queue.popleft()
            if node.getRoleName() in ("document web", "internal frame", "pane"):
                try:
                    ext = node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                except NotImplementedError:
                    raise TransientError(
                        f"document web node has no Component interface for app={app}"
                    )
                return (ext.x, ext.y, ext.width, ext.height)
            for i in range(node.childCount):
                child = node.getChildAtIndex(i)
                if child is not None:
                    queue.append(child)

        raise TransientError(
            f"document web node not found for app={app} "
            f"— browser accessibility tree may not be fully rendered"
        )

    # -----------------------------------------------------------------
    # Marker coordinate builders
    # -----------------------------------------------------------------

    def _get_atspi_marker_coords(self, bfs_matches: list) -> tuple[dict, dict]:
        """Convert AT-SPI BFS matches into marker coordinate and
        back-reference dicts for Vision fallback."""
        marker_coords = {}
        marker_to_bfs = {}
        next_marker = 1

        for i, node in enumerate(bfs_matches):
            try:
                ext = node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                x, y, w, h = int(ext.x), int(ext.y), int(ext.width), int(ext.height)
            except NotImplementedError:
                continue
            if x == 0 and y == 0 and w == 0 and h == 0:
                continue
            if not (0 <= x <= 32768 and 0 <= y <= 32768 and 0 < w <= 32768 and 0 < h <= 32768):
                continue
            marker_coords[next_marker] = (x + w // 2, y + h // 2, w, h)
            marker_to_bfs[next_marker] = i
            next_marker += 1

        if not marker_coords:
            raise EnvironmentalError("all atspi vision candidates have zero extents")

        return (marker_coords, marker_to_bfs)

    def _get_cdp_marker_coords(self, ws, bfs_matches: list,
                               viewport_w: int, viewport_h: int) -> tuple[dict, dict]:
        """Convert CDP BFS matches into marker coordinate and
        back-reference dicts for Vision fallback."""
        marker_coords = {}
        marker_to_bfs = {}
        next_marker = 1

        for i, node_dict in enumerate(bfs_matches):
            result = self._cdp_send(
                ws, "DOM.getBoxModel",
                {"backendNodeId": node_dict["backendDOMNodeId"]},
            )
            content = result["model"]["content"]
            cdp_x = (content[0] + content[2] + content[4] + content[6]) / 4
            cdp_y = (content[1] + content[3] + content[5] + content[7]) / 4
            w = content[2] - content[0]
            h = content[7] - content[1]

            if cdp_x < 0 or cdp_x > viewport_w or cdp_y < 0 or cdp_y > viewport_h:
                continue

            marker_coords[next_marker] = (cdp_x, cdp_y, w, h)
            marker_to_bfs[next_marker] = i
            next_marker += 1

        if not marker_coords:
            raise EnvironmentalError("all cdp vision candidates are off-viewport")

        return (marker_coords, marker_to_bfs)

    # -----------------------------------------------------------------
    # Vision model methods
    # -----------------------------------------------------------------

    def _vision_select(self, wid: int, label: str, role: str, name: str,
                       marker_coords: dict) -> int:
        """Capture window screenshot, overlay numbered markers, call
        OpenRouter vision model, and return the selected marker number."""

        # Step 1 — Window geometry via xdotool
        try:
            result = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", str(wid)],
                capture_output=True, text=True, timeout=5,
            )
        except subprocess.TimeoutExpired:
            raise EnvironmentalError(
                f"xdotool getwindowgeometry failed for vision capture wid={wid}"
            )
        if result.returncode != 0:
            raise EnvironmentalError(
                f"xdotool getwindowgeometry failed for vision capture wid={wid}"
            )
        win_x = win_y = win_w = win_h = None
        for line in result.stdout.splitlines():
            if line.startswith("X="):
                win_x = int(line.split("=", 1)[1])
            elif line.startswith("Y="):
                win_y = int(line.split("=", 1)[1])
            elif line.startswith("WIDTH="):
                win_w = int(line.split("=", 1)[1])
            elif line.startswith("HEIGHT="):
                win_h = int(line.split("=", 1)[1])
        if None in (win_x, win_y, win_w, win_h):
            raise EnvironmentalError(
                f"xdotool getwindowgeometry failed for vision capture wid={wid}"
            )

        # Step 2 — Screenshot via scrot
        capture_path = f"/tmp/genie_vision_{label}.png"
        try:
            scrot_result = subprocess.run(
                ["scrot", "-a", f"{win_x},{win_y},{win_w},{win_h}", capture_path],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            raise EnvironmentalError("scrot capture failed")
        if scrot_result.returncode != 0:
            raise EnvironmentalError("scrot capture failed")

        # Step 3 — Marker overlay via OpenCV
        import cv2

        img = cv2.imread(capture_path)
        if img is None:
            raise EnvironmentalError("cv2 failed to read capture")

        for marker_num, (cx, cy, w, h) in marker_coords.items():
            img_x = cx - win_x
            img_y = cy - win_y
            cv2.circle(img, (int(img_x), int(img_y)), 15, (0, 0, 255), -1)
            cv2.putText(
                img, str(marker_num),
                (int(img_x) - 5, int(img_y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
            )
        cv2.imwrite(capture_path, img)
        print(f"[VISION DEBUG] wid={wid} win=({win_x},{win_y},{win_w},{win_h})")
        for m, (cx, cy, w, h) in marker_coords.items():
            print(f"  marker {m}: abs=({cx},{cy}) img=({cx-win_x},{cy-win_y})")
        import shutil
        shutil.copy(capture_path, f"/tmp/genie_vision_debug_{label}.png")

        # Step 4 — Base64 encode
        with open(capture_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        # Step 5 — OpenRouter API call
        api_key = OPENROUTER_API_KEY
        if not api_key:
            raise ResourceError(
                "OPENROUTER_API_KEY not set — vision fallback unavailable"
            )

        request_body = json.dumps({
            "model": VISION_MODEL,
            "max_tokens": 10,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a UI element locator. "
                        "You must respond with a single integer only. "
                        "No explanation, no punctuation, no other text."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Which numbered marker in this screenshot corresponds "
                                f"to the {role} element named '{name}'? "
                                f"Valid marker numbers are 1 to {len(marker_coords)}. "
                                "Reply with a single integer from that range only. "
                                "If multiple markers look identical, choose the lowest-numbered one."
                            ),
                        },
                    ],
                },
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=request_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            raise TransientError(f"vision api call failed: {e}")

        # Step 6 — Parse response
        try:
            response_body = json.loads(resp.read().decode())
            raw_text = response_body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            raise EnvironmentalError(
                "vision model returned malformed response"
            )

        match = re.search(r"\b(\d+)\b", raw_text)
        if match is None:
            raise EnvironmentalError(
                f"vision model returned non-integer: {raw_text!r}"
            )
        marker_num = int(match.group(1))

        if marker_num == 0:
            raise EnvironmentalError(
                "vision model found no matching marker — "
                "element not identifiable from screenshot"
            )

        if marker_num not in marker_coords:
            raise TransientError(
                f"vision model returned out-of-range marker: {marker_num}"
            )

        return marker_num

    def _vision_discover(self, wid: int, label: str, role: str,
                         name: str) -> tuple[int, int]:
        """Vision-based element discovery when CDP/AT-SPI find 0 matches.

        Takes a screenshot of the window, sends it to the vision model, and
        asks it to locate the element by role/name.  Returns absolute screen
        coordinates (abs_x, abs_y) for the element centre.
        """
        # Step 1 — Window geometry via xdotool
        try:
            result = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", str(wid)],
                capture_output=True, text=True, timeout=5,
            )
        except subprocess.TimeoutExpired:
            raise EnvironmentalError(
                f"xdotool getwindowgeometry failed for vision discover wid={wid}"
            )
        if result.returncode != 0:
            raise EnvironmentalError(
                f"xdotool getwindowgeometry failed for vision discover wid={wid}"
            )
        win_x = win_y = win_w = win_h = None
        for line in result.stdout.splitlines():
            if line.startswith("X="):
                win_x = int(line.split("=", 1)[1])
            elif line.startswith("Y="):
                win_y = int(line.split("=", 1)[1])
            elif line.startswith("WIDTH="):
                win_w = int(line.split("=", 1)[1])
            elif line.startswith("HEIGHT="):
                win_h = int(line.split("=", 1)[1])
        if None in (win_x, win_y, win_w, win_h):
            raise EnvironmentalError(
                f"xdotool getwindowgeometry incomplete for vision discover wid={wid}"
            )

        # Step 2 — Screenshot via scrot
        capture_path = f"/tmp/genie_vision_discover_{label}.png"
        try:
            scrot_result = subprocess.run(
                ["scrot", "-a", f"{win_x},{win_y},{win_w},{win_h}", capture_path],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            raise EnvironmentalError("scrot capture failed for vision discover")
        if scrot_result.returncode != 0:
            raise EnvironmentalError("scrot capture failed for vision discover")

        # Step 3 — Base64 encode (no markers — just raw screenshot)
        with open(capture_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        # Step 4 — Vision model API call
        api_key = OPENROUTER_API_KEY
        if not api_key:
            raise ResourceError(
                "OPENROUTER_API_KEY not set — vision discover unavailable"
            )

        if name:
            element_desc = f"the {role} element named '{name}'"
        else:
            element_desc = f"the {role} element (e.g. search box, text input)"

        request_body = json.dumps({
            "model": VISION_MODEL,
            "max_tokens": 30,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a UI element locator. Given a screenshot of a "
                        "browser window, identify the pixel coordinates of the "
                        "requested element. Respond with ONLY two integers: "
                        "x y (space-separated, relative to the screenshot's "
                        "top-left corner). No explanation, no punctuation."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Where is {element_desc} in this screenshot? "
                                f"The screenshot is {win_w}x{win_h} pixels. "
                                "Reply with pixel coordinates: x y"
                            ),
                        },
                    ],
                },
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=request_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            raise TransientError(f"vision discover api call failed: {e}")

        # Step 5 — Parse response — expect "x y" format
        try:
            response_body = json.loads(resp.read().decode())
            raw_text = response_body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            raise EnvironmentalError(
                "vision discover model returned malformed response"
            )

        log(f"vision_discover: raw response for role={role} name={name!r}: {raw_text!r}")

        nums = re.findall(r"\d+", raw_text)
        if len(nums) < 2:
            raise EnvironmentalError(
                f"vision discover: expected 'x y' coordinates, got: {raw_text!r}"
            )
        img_x, img_y = int(nums[0]), int(nums[1])

        if img_x < 0 or img_x > win_w or img_y < 0 or img_y > win_h:
            raise EnvironmentalError(
                f"vision discover: coordinates ({img_x}, {img_y}) out of "
                f"screenshot bounds ({win_w}x{win_h})"
            )

        abs_x = win_x + img_x
        abs_y = win_y + img_y

        log(f"vision_discover: role={role} name={name!r} → "
            f"img=({img_x},{img_y}) abs=({abs_x},{abs_y})")

        return (abs_x, abs_y)

    # -----------------------------------------------------------------
    # Vision screen reader
    # -----------------------------------------------------------------

    def _vision_read_screen(self, wid: int, label: str,
                            question: str = "") -> str:
        """Take a screenshot of a window and return a text description.

        If *question* is provided the vision model answers that specific
        question about the screen content.  Otherwise it returns a general
        description of what is visible.
        """
        # Step 1 — Window geometry via xdotool
        try:
            result = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", str(wid)],
                capture_output=True, text=True, timeout=5,
            )
        except subprocess.TimeoutExpired:
            raise EnvironmentalError(
                f"xdotool getwindowgeometry failed for vision read wid={wid}"
            )
        if result.returncode != 0:
            raise EnvironmentalError(
                f"xdotool getwindowgeometry failed for vision read wid={wid}"
            )
        win_x = win_y = win_w = win_h = None
        for line in result.stdout.splitlines():
            if line.startswith("X="):
                win_x = int(line.split("=", 1)[1])
            elif line.startswith("Y="):
                win_y = int(line.split("=", 1)[1])
            elif line.startswith("WIDTH="):
                win_w = int(line.split("=", 1)[1])
            elif line.startswith("HEIGHT="):
                win_h = int(line.split("=", 1)[1])
        if None in (win_x, win_y, win_w, win_h):
            raise EnvironmentalError(
                f"xdotool getwindowgeometry incomplete for vision read wid={wid}"
            )

        # Step 2 — Screenshot via scrot
        capture_path = f"/tmp/genie_vision_read_{label}.png"
        try:
            scrot_result = subprocess.run(
                ["scrot", "-a", f"{win_x},{win_y},{win_w},{win_h}", capture_path],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            raise EnvironmentalError("scrot capture failed for vision read")
        if scrot_result.returncode != 0:
            raise EnvironmentalError("scrot capture failed for vision read")

        # Step 3 — Base64 encode
        with open(capture_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        # Step 4 — Vision model API call
        api_key = OPENROUTER_API_KEY
        if not api_key:
            raise ResourceError(
                "OPENROUTER_API_KEY not set — vision read unavailable"
            )

        if question:
            user_text = question
        else:
            user_text = (
                "Describe the content visible in this window screenshot. "
                "Focus on: (1) what application or page is shown, "
                "(2) the main text content visible, "
                "(3) any dialogs, popups, or overlays present. "
                "Be factual and concise — max 3 sentences."
            )

        request_body = json.dumps({
            "model": VISION_MODEL,
            "max_tokens": 300,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a screen reader for an automation agent. "
                        "Describe what you see accurately and concisely. "
                        "Focus on visible text content, page identity, and "
                        "any UI state that would affect automated interaction "
                        "(dialogs, popups, loading states, errors). "
                        "Do NOT describe layout or styling."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": user_text,
                        },
                    ],
                },
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=request_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            raise TransientError(f"vision read api call failed: {e}")

        try:
            response_body = json.loads(resp.read().decode())
            description = response_body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            raise EnvironmentalError(
                "vision read model returned malformed response"
            )

        log(f"vision_read_screen: app={label!r} → {len(description)} chars")

        # Step 5 — Log to vision training file
        if not self._no_vision_log:
            try:
                import datetime as _dt
                training_record = json.dumps({
                    "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
                    "app": label,
                    "wid": wid,
                    "screenshot_path": capture_path,
                    "question": question,
                    "description": description,
                    "model": VISION_MODEL,
                })
                vision_log_path = os.path.join(LOG_DIR, "vision_training.jsonl")
                with open(vision_log_path, "a") as vf:
                    vf.write(training_record + "\n")
            except Exception:
                pass

        return description

    def look(self, app: str, question: str = "") -> dict:
        """Take a screenshot of a window and return a text description.

        This is the LLM-callable action.  Returns a compact text description
        that goes into context.

        Args:
            app: registry label of the window to look at
            question: optional specific question about the screen content

        Returns:
            {"description": str} — bounded text description of screen state
        """
        if app not in self.registry._registry:
            raise EnvironmentalError(f"app label '{app}' not in registry")

        wid = self.registry._registry[app].get("wid")
        if not wid:
            raise EnvironmentalError(f"app '{app}' has no wid — window not open")

        description = self._vision_read_screen(wid, app, question)
        return {"description": description}
