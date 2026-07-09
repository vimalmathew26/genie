"""ElementResolver — semantic element resolution for the Genie automation agent.
Translates (app_label, role, name) tuples into executed UI actions across
three resolution tiers: AT-SPI, CDP, and Vision fallback.

Tier implementations are split into separate mixin files:
  - atspi_mixin.py   — AT-SPI resolution methods
  - cdp_mixin.py     — CDP transport, AX BFS, JS helpers
  - vision_mixin.py  — Vision model calls, viewport/marker helpers
"""

import subprocess
import time

import pyatspi

from config import (
    log,
    CDP_ROLE_ALIASES,
    ATSPI_STATE_SETTLE_MS,
)
from window_registry import WindowRegistry
from xdotool_controller import XdotoolController
from exceptions import (
    EnvironmentalError,
    TransientError,
    UnrecoverableError,
)
from atspi_mixin import ATSpiMixin
from cdp_mixin import CDPMixin
from vision_mixin import VisionMixin


class ElementResolver(ATSpiMixin, CDPMixin, VisionMixin):

    def __init__(self, registry: WindowRegistry, controller: XdotoolController):
        self.registry = registry
        self.controller = controller
        self._cdp_id = 0

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def click_element(self, app: str, role: str, name: str,
                      url_hint: str | None = None, retry: bool = False,
                      index: int = 0) -> dict:
        """Click the UI element identified by (app, role, name).

        Args:
            index: 0-based index for positional selection. When role="link"
                   and name="", index=N navigates to the Nth organic search
                   result via JS (avoiding AX tree entirely on heavy pages).
                   Default 0 = first match (backwards compatible).

        Resolves via AT-SPI for atspi/terminal tiers, CDP for cdp_primary tier.
        Raises typed subclasses of ElementResolverError on failure.
        """
        # Validate app label
        if app not in self.registry._registry:
            raise EnvironmentalError(f"app label '{app}' not in registry")

        # Determine tier
        tier = self.registry._registry.get(app, {}).get("tier", "atspi")
        if tier not in ("atspi", "terminal", "cdp_primary"):
            tier = "atspi"

        if tier == "terminal":
            raise UnrecoverableError(
                "click_element not supported for terminal tier — no semantic element tree"
            )

        if tier == "cdp_primary":
            cdp_port = self.registry._registry[app].get("cdp_port")
            if not cdp_port:
                raise EnvironmentalError(
                    f"no cdp_port in registry for app={app}"
                )
            with self._cdp_session(cdp_port, url_hint) as ws:

                # Fast path: role=link, name="" → resolve Nth result URL
                # via JS and navigate directly using Page.navigate.
                # This completely avoids coordinate-click issues (scroll
                # offsets, ads/image-packs appearing before organic results
                # in DOM order, etc.) and bypasses Accessibility.getFullAXTree
                # which crashes on heavy pages like Bing.
                if role == "link" and name == "":
                    # JS fast path: resolve Nth result URL via JS and
                    # navigate directly.  Avoids coordinate-click and AX
                    # tree issues on heavy pages (Bing, etc.).
                    # Two attempts with a 1.5 s gap for page-load headroom.
                    for _link_attempt in range(2):
                        target_url = self._cdp_js_nth_result_url(ws, index)
                        if target_url:
                            self._cdp_navigate(ws, target_url)
                            return {"tier": "cdp", "navigated_to": target_url,
                                    "cdp_port": cdp_port,
                                    "result_index": index}
                        if _link_attempt < 1:
                            time.sleep(1.5)
                    # JS URL extraction failed — fall through to AX BFS

                matches = self._bfs_cdp(ws, role, name)

                if len(matches) == 0:
                    # ── SERP JS fallback ──────────────────────────────
                    # AX BFS found nothing.  Before expensive vision
                    # fallback, if we're looking for a link check
                    # whether the page is a known SERP and extract the
                    # organic result URL via site-specific CSS selectors
                    # (_SERP_RESULTS_JS handles Google, Bing, DDG).
                    # This covers DDG AI overlay where AX tree names
                    # don't match LLM-provided link names.
                    if role == "link":
                        serp_url = self._cdp_js_nth_result_url(ws, index)
                        if serp_url:
                            log(f"click_element: 0 AX matches for "
                                f"role=link name={name!r} — SERP JS "
                                f"fallback navigating to {serp_url}")
                            self._cdp_navigate(ws, serp_url)
                            return {"tier": "cdp_js_serp",
                                    "navigated_to": serp_url,
                                    "cdp_port": cdp_port,
                                    "result_index": index}

                    # Vision discovery fallback: CDP AX tree and JS
                    # SERP extraction both failed.  Take a screenshot
                    # and ask the vision model to locate the element.
                    log(f"click_element: 0 CDP matches for role={role} "
                        f"name={name!r} — trying vision discover")
                    wid = self.registry._registry[app]["wid"]
                    abs_x, abs_y = self._vision_discover(
                        wid, app, role, name
                    )
                    self.controller.click(int(abs_x), int(abs_y))
                    return {"tier": "cdp_vision", "backendDOMNodeId": None,
                            "cdp_port": cdp_port}

                if len(matches) >= 2:
                    viewport_x, viewport_y, viewport_w, viewport_h = self._get_viewport_extents(app)
                    cdp_marker_coords, marker_to_bfs = self._get_cdp_marker_coords(
                        ws, matches, viewport_w, viewport_h
                    )
                    # Convert CDP-relative to absolute screen coords for _vision_select
                    abs_marker_coords = {
                        m: (viewport_x + cx, viewport_y + cy, w, h)
                        for m, (cx, cy, w, h) in cdp_marker_coords.items()
                    }
                    wid = self.registry._registry[app]["wid"]
                    selected = self._vision_select(wid, app, role, name, abs_marker_coords)
                    # click_element Vision path: use marker coords directly, no bfs back-reference needed
                    abs_x, abs_y, _, _ = abs_marker_coords[selected]
                    self.controller.click(int(abs_x), int(abs_y))
                    return {"tier": "cdp", "backendDOMNodeId": None, "cdp_port": cdp_port}

                # Single match — extract backendDOMNodeId
                backend_node_id = matches[0].get("backendDOMNodeId", 0)
                if not backend_node_id:
                    raise EnvironmentalError(
                        f"backendDOMNodeId absent or zero: role={role} name={name}"
                    )

                # Coordinate resolution
                cdp_x, cdp_y = self._resolve_cdp_coords(ws, backend_node_id)

                if retry:
                    # dispatchMouseEvent fallback — viewport-relative, no abs offset
                    self._cdp_send(ws, "Input.dispatchMouseEvent", {
                        "type": "mousePressed",
                        "x": cdp_x, "y": cdp_y,
                        "button": "left", "clickCount": 1, "buttons": 1,
                    })
                    self._cdp_send(ws, "Input.dispatchMouseEvent", {
                        "type": "mouseReleased",
                        "x": cdp_x, "y": cdp_y,
                        "button": "left", "clickCount": 1, "buttons": 0,
                    })
                else:
                    # xdotool primary — needs absolute screen coords
                    viewport_x, viewport_y = self._get_viewport_origin(app)
                    abs_x = viewport_x + cdp_x
                    abs_y = viewport_y + cdp_y
                    self.controller.click(int(abs_x), int(abs_y))


                return {"tier": "cdp", "backendDOMNodeId": backend_node_id, "cdp_port": cdp_port}

        # AT-SPI path (atspi and terminal tiers)
        window_node = self._find_atspi_window(app)
        matches = self._bfs_atspi(window_node, role, name)

        if len(matches) == 0:
            raise EnvironmentalError(
                f"element not found: role={role} name={name}"
            )

        if len(matches) >= 2:
            # Build disambiguation payload
            payload = f"Found {len(matches)} elements matching role={role} name={name}:\n"
            for idx, match in enumerate(matches, start=1):
                parent_name = getattr(
                    getattr(match, 'parent', None), 'name', ''
                ) or '<unknown>'
                ext = match.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                payload += (
                    f"[{idx}] parent={parent_name} "
                    f"extents=({ext.x},{ext.y},{ext.width},{ext.height})\n"
                )

            if tier == "terminal":
                raise EnvironmentalError(payload)
            # tier == "atspi": Vision fallback
            marker_coords, marker_to_bfs = self._get_atspi_marker_coords(matches)
            wid = self.registry._registry[app]["wid"]
            selected = self._vision_select(wid, app, role, name, marker_coords)
            abs_x, abs_y, _, _ = marker_coords[selected]
            self.controller.click(int(abs_x), int(abs_y))
            return {"tier": "atspi", "element_state": None}

        # Exactly 1 match — proceed to click
        self._atspi_click(matches[0], role=role, name=name)
        try:
            time.sleep(ATSPI_STATE_SETTLE_MS / 1000)
            raw_states = matches[0].getState().getStates()
            element_state = [pyatspi.stateToString(s) for s in raw_states]
        except Exception:
            element_state = None
        return {"tier": "atspi", "element_state": element_state}

    def type_element(self, app: str, role: str, name: str, text: str,
                     url_hint: str | None = None) -> dict:
        """Type text into the UI element identified by (app, role, name).

        Resolves via AT-SPI for atspi/terminal tiers, CDP for cdp_primary tier.
        Raises typed subclasses of ElementResolverError on failure.

        Returns a metadata dict: {"tier": "atspi"} for AT-SPI and terminal tiers,
        {"tier": "cdp", "backendDOMNodeId": int | None, "cdp_port": int} for CDP tier.
        Never returns None on non-exception paths.
        """
        # Validate app label
        if app not in self.registry._registry:
            raise EnvironmentalError(f"app label '{app}' not in registry")

        # Determine tier
        tier = self.registry._registry.get(app, {}).get("tier", "atspi")
        if tier not in ("atspi", "terminal", "cdp_primary"):
            tier = "atspi"

        if tier == "terminal":
            wid = self.registry._registry[app]["wid"]
            self.controller.type_to_window(wid, text)
            return {"tier": "terminal"}

        if tier == "cdp_primary":
            cdp_port = self.registry._registry[app].get("cdp_port")
            if not cdp_port:
                raise EnvironmentalError(
                    f"no cdp_port in registry for app={app}"
                )
            with self._cdp_session(cdp_port, url_hint) as ws:
                # Fast path for unnamed input roles: use JS to find and focus
                # the best input element, bypassing Accessibility.getFullAXTree
                # which times out on heavy pages (e.g. Bing, news sites).
                _INPUT_ROLES = ("textfield", "combobox", "searchbox", "search")
                if name == "" and role in _INPUT_ROLES:
                    # Try JS fast path with retries — pages like Bing may still
                    # be loading when the first attempt runs.
                    for _js_attempt in range(3):
                        if self._cdp_js_focus_best_input(ws):
                            self._cdp_type_chars(ws, text)
                            subprocess.run(
                                ["xdotool", "windowfocus", "--sync",
                                 str(self.registry._registry[app]["wid"])],
                                check=False,
                            )
                            return {"tier": "cdp", "backendDOMNodeId": None,
                                    "cdp_port": cdp_port}
                        if _js_attempt < 2:
                            time.sleep(1.5)  # wait for page to finish loading
                    # JS fast path failed after retries — fall through to AX tree BFS.

                matches = self._bfs_cdp(ws, role, name)

                if len(matches) == 0:
                    raise EnvironmentalError(
                        f"element not found: role={role} name={name}"
                    )

                if len(matches) >= 2:
                    viewport_x, viewport_y, viewport_w, viewport_h = self._get_viewport_extents(app)
                    cdp_marker_coords, marker_to_bfs = self._get_cdp_marker_coords(
                        ws, matches, viewport_w, viewport_h
                    )
                    abs_marker_coords = {
                        m: (viewport_x + cx, viewport_y + cy, w, h)
                        for m, (cx, cy, w, h) in cdp_marker_coords.items()
                    }
                    wid = self.registry._registry[app]["wid"]
                    selected = self._vision_select(wid, app, role, name, abs_marker_coords)
                    # Recover backendDOMNodeId from bfs back-reference
                    backend_node_id = matches[marker_to_bfs[selected]].get("backendDOMNodeId", 0)
                    if not backend_node_id:
                        raise EnvironmentalError(
                            f"backendDOMNodeId absent after vision selection: role={role} name={name}"
                        )
                    cdp_x, cdp_y = self._resolve_cdp_coords(ws, backend_node_id)
                    self._cdp_send(ws, "Input.dispatchMouseEvent", {
                        "type": "mousePressed",
                        "x": cdp_x, "y": cdp_y,
                        "button": "left", "clickCount": 1, "buttons": 1,
                    })
                    self._cdp_send(ws, "Input.dispatchMouseEvent", {
                        "type": "mouseReleased",
                        "x": cdp_x, "y": cdp_y,
                        "button": "left", "clickCount": 1, "buttons": 0,
                    })
                    self._cdp_send(ws, "DOM.focus", {"backendNodeId": backend_node_id})
                    self._cdp_type_chars(ws, text)
                    # Restore OS-level window focus so subsequent xdotool key
                    # events (e.g. press_key enter) reach the Chrome window.
                    subprocess.run(
                        ["xdotool", "windowfocus", "--sync",
                         str(self.registry._registry[app]["wid"])],
                        check=False,
                    )
                    return {"tier": "cdp", "backendDOMNodeId": None, "cdp_port": cdp_port}

                # Single match — extract backendDOMNodeId
                backend_node_id = matches[0].get("backendDOMNodeId", 0)
                if not backend_node_id:
                    raise EnvironmentalError(
                        f"backendDOMNodeId absent or zero: role={role} name={name}"
                    )

                # Coordinate resolution
                cdp_x, cdp_y = self._resolve_cdp_coords(ws, backend_node_id)

                # Synthetic click to activate JS components
                self._cdp_send(ws, "Input.dispatchMouseEvent", {
                    "type": "mousePressed",
                    "x": cdp_x, "y": cdp_y,
                    "button": "left", "clickCount": 1, "buttons": 1,
                })
                self._cdp_send(ws, "Input.dispatchMouseEvent", {
                    "type": "mouseReleased",
                    "x": cdp_x, "y": cdp_y,
                    "button": "left", "clickCount": 1, "buttons": 0,
                })

                # Focus and inject text
                self._cdp_send(ws, "DOM.focus", {
                    "backendNodeId": backend_node_id,
                })
                self._cdp_type_chars(ws, text)
                # Restore OS-level window focus so subsequent xdotool key
                # events (e.g. press_key enter) reach the Chrome window.
                subprocess.run(
                    ["xdotool", "windowfocus", "--sync",
                     str(self.registry._registry[app]["wid"])],
                    check=False,
                )

                return {"tier": "cdp", "backendDOMNodeId": backend_node_id, "cdp_port": cdp_port}

        # AT-SPI path (atspi and terminal tiers)
        window_node = self._find_atspi_window(app)
        matches = self._bfs_atspi(window_node, role, name)

        if len(matches) == 0:
            raise EnvironmentalError(
                f"element not found: role={role} name={name}"
            )

        if len(matches) >= 2:
            # Build disambiguation payload
            payload = f"Found {len(matches)} elements matching role={role} name={name}:\n"
            for idx, match in enumerate(matches, start=1):
                parent_name = getattr(
                    getattr(match, 'parent', None), 'name', ''
                ) or '<unknown>'
                ext = match.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                payload += (
                    f"[{idx}] parent={parent_name} "
                    f"extents=({ext.x},{ext.y},{ext.width},{ext.height})\n"
                )

            if tier == "terminal":
                raise EnvironmentalError(payload)
            # tier == "atspi": Vision fallback
            marker_coords, marker_to_bfs = self._get_atspi_marker_coords(matches)
            wid = self.registry._registry[app]["wid"]
            selected = self._vision_select(wid, app, role, name, marker_coords)
            node = matches[marker_to_bfs[selected]]
            self._atspi_type(node, text)
            return {"tier": "atspi"}

        # Exactly 1 match — proceed to type
        self._atspi_type(matches[0], text)
        return {"tier": "atspi"}

    def read_element(self, app: str, role: str, name: str,
                     url_hint: str | None = None,
                     index: int = 0) -> dict:
        """Read and return the content of the UI element identified by (app, role, name).

        Args:
            index: 0-based positional selection. When role="link" and name="",
                   index=N reads the text of the Nth visible link on the page
                   via JS (DOM order).  Default 0.

        Resolves via AT-SPI for atspi/terminal tiers, CDP for cdp_primary tier.
        Raises typed subclasses of ElementResolverError on failure.

        Returns {"content": str} — truncated element text.
        """
        # Validate app label
        if app not in self.registry._registry:
            raise EnvironmentalError(f"app label '{app}' not in registry")

        # Determine tier
        tier = self.registry._registry.get(app, {}).get("tier", "atspi")
        if tier not in ("atspi", "terminal", "cdp_primary"):
            tier = "atspi"

        if tier == "terminal":
            raise UnrecoverableError(
                "read_element not supported for terminal tier — no semantic element tree"
            )

        if tier == "cdp_primary":
            cdp_port = self.registry._registry[app].get("cdp_port")
            if not cdp_port:
                raise EnvironmentalError(
                    f"no cdp_port in registry for app={app}"
                )
            with self._cdp_session(cdp_port, url_hint) as ws:
                # ── JS fast path: role=link, name="" ──────────────────
                # Use DOM enumeration to read the Nth visible link text.
                # This avoids AX BFS (which only returns one link in
                # role-only mode) and works on any page, not just SERPs.
                if role == "link" and name == "":
                    text = self._cdp_js_read_page_links(ws, index)
                    if text:
                        return {"content": self._truncate(text)}
                    # JS extraction failed — fall through to AX BFS path

                try:
                    matches = self._bfs_cdp(ws, role, name)
                except TransientError:
                    # AX tree fetch crashed (common on heavy pages like Bing).
                    # The websocket `ws` is likely dead after the crash, so
                    # open a FRESH CDP session for JS fallback extraction.
                    with self._cdp_session(cdp_port, url_hint) as ws2:
                        # For link-role reads, fall back to JS-based SERP reader
                        # which extracts result titles + URLs without AX tree.
                        if role == "link":
                            serp_results = self._cdp_js_read_serp_results(ws2)
                            if serp_results:
                                lines = []
                                for r in serp_results:
                                    lines.append(
                                        f"[{r['index'] + 1}] {r['title']}  ({r['url']})"
                                    )
                                content = (
                                    "Search results (from JS fallback — AX tree "
                                    "unavailable on this page):\n"
                                    + "\n".join(lines)
                                    + "\n\nTo click result N, use: "
                                    "click_element {role: \"link\", name: \"\", index: N-1}"
                                )
                                return {"content": self._truncate(content)}
                        # For input roles, try reading the input value via JS
                        _INPUT_ROLES = ("textfield", "combobox", "searchbox", "search")
                        if role in _INPUT_ROLES:
                            try:
                                val_result = self._cdp_send(ws2, "Runtime.evaluate", {
                                    "expression": (
                                        "(function(){"
                                        "  var el=" + self._FIND_INPUT_JS.replace("(function(){", "").replace("})()", "") + ";"
                                        "  if(!el) return '';"
                                        "  return el.value || el.innerText || '';"
                                        "})()"
                                    ),
                                    "returnByValue": True,
                                })
                                raw = val_result.get("result", {}).get("value", "") or ""
                                if raw:
                                    return {"content": self._truncate(raw)}
                            except Exception:
                                pass
                        # For any role, try JS-based main content extraction
                        # as a last resort before re-raising.
                        try:
                            content = self._cdp_js_read_main_content(ws2, role, name)
                            if content:
                                return {"content": self._truncate(content)}
                        except Exception:
                            pass
                    # ── Vision auto-fallback: AX crashed + all JS failed ──
                    # Take a screenshot and ask the vision model to read it.
                    try:
                        wid = self.registry._registry[app].get("wid")
                        if wid:
                            question = (
                                f"Read the main text content of this page. "
                                f"I was looking for role={role} name={name!r}. "
                                f"Return the relevant text content you can see."
                            )
                            desc = self._vision_read_screen(wid, app, question)
                            if desc and len(desc.strip()) > 20:
                                log(f"read_element: vision auto-fallback returned "
                                    f"{len(desc)} chars after AX crash + JS fail")
                                return {"content": self._truncate(desc),
                                        "tier": "vision_fallback"}
                    except Exception:
                        pass
                    # Re-raise if no JS fallback worked
                    raise

                if len(matches) == 0:
                    # JS content extraction fallback for text/content roles
                    _TEXT_ROLES = (
                        "paragraph", "heading", "statictext", "text",
                        "label", "group", "article",
                    )
                    if role in _TEXT_ROLES or name:
                        try:
                            content = self._cdp_js_read_main_content(ws, role, name)
                            if content:
                                log(f"read_element: 0-match JS fallback "
                                    f"returned {len(content)} chars")
                                return {"content": self._truncate(content)}
                        except Exception:
                            pass
                    # ── Vision auto-fallback: 0 AX matches + JS empty ──
                    try:
                        wid = self.registry._registry[app].get("wid")
                        if wid:
                            question = (
                                f"Read the main text content visible on this page. "
                                f"I was looking for role={role} name={name!r} but "
                                f"could not find it via structured extraction. "
                                f"Return the relevant text content you can see."
                            )
                            desc = self._vision_read_screen(wid, app, question)
                            if desc and len(desc.strip()) > 20:
                                log(f"read_element: vision auto-fallback returned "
                                    f"{len(desc)} chars (0 AX matches, JS empty)")
                                return {"content": self._truncate(desc),
                                        "tier": "vision_fallback"}
                    except Exception:
                        pass
                    raise EnvironmentalError(
                        f"element not found: role={role} name={name}"
                    )

                if len(matches) >= 2:
                    viewport_x, viewport_y, viewport_w, viewport_h = self._get_viewport_extents(app)
                    cdp_marker_coords, marker_to_bfs = self._get_cdp_marker_coords(
                        ws, matches, viewport_w, viewport_h
                    )
                    abs_marker_coords = {
                        m: (viewport_x + cx, viewport_y + cy, w, h)
                        for m, (cx, cy, w, h) in cdp_marker_coords.items()
                    }
                    wid = self.registry._registry[app]["wid"]
                    selected = self._vision_select(wid, app, role, name, abs_marker_coords)
                    backend_node_id = matches[marker_to_bfs[selected]].get("backendDOMNodeId", 0)
                    if not backend_node_id:
                        raise EnvironmentalError(
                            f"backendDOMNodeId absent after vision selection: role={role} name={name}"
                        )
                    # No xdotool click — read_element Vision path skips controller call
                    use_dom_path = (
                        role in ("textfield", "dropdown")
                        or role in CDP_ROLE_ALIASES.get("textfield", [])
                        or role in CDP_ROLE_ALIASES.get("dropdown", [])
                    )
                    if use_dom_path:
                        resolve_result = self._cdp_send(
                            ws, "DOM.resolveNode", {"backendNodeId": backend_node_id}
                        )
                        object_id = resolve_result["object"]["objectId"]
                        try:
                            call_result = self._cdp_send(
                                ws, "Runtime.callFunctionOn",
                                {
                                    "objectId": object_id,
                                    "functionDeclaration": (
                                        "function(){return this.value !== undefined "
                                        "? this.value : (this.innerText || '')}"
                                    ),
                                    "returnByValue": True,
                                },
                            )
                            raw = call_result["result"]["value"]
                        finally:
                            try:
                                self._cdp_send(
                                    ws, "Runtime.releaseObject", {"objectId": object_id}
                                )
                            except Exception:
                                pass
                    else:
                        raw = matches[marker_to_bfs[selected]].get("name", {}).get("value", "")
                    return {"content": self._truncate(raw)}

                # Single match — extract backendDOMNodeId
                backend_node_id = matches[0].get("backendDOMNodeId", 0)
                if not backend_node_id:
                    raise EnvironmentalError(
                        f"backendDOMNodeId absent or zero: role={role} name={name}"
                    )

                # Role routing — text-type roles use DOM path, others use AX name
                use_dom_path = (
                    role in ("textfield", "dropdown")
                    or role in CDP_ROLE_ALIASES.get("textfield", [])
                    or role in CDP_ROLE_ALIASES.get("dropdown", [])
                )

                if use_dom_path:
                    # DOM path — DOM.resolveNode → callFunctionOn → releaseObject
                    resolve_result = self._cdp_send(
                        ws, "DOM.resolveNode",
                        {"backendNodeId": backend_node_id},
                    )
                    object_id = resolve_result["object"]["objectId"]

                    try:
                        call_result = self._cdp_send(
                            ws, "Runtime.callFunctionOn",
                            {
                                "objectId": object_id,
                                "functionDeclaration": (
                                    "function(){return this.value !== undefined "
                                    "? this.value : (this.innerText || '')}"
                                ),
                                "returnByValue": True,
                            },
                        )
                        raw = call_result["result"]["value"]
                    finally:
                        try:
                            self._cdp_send(
                                ws, "Runtime.releaseObject",
                                {"objectId": object_id},
                            )
                        except Exception:
                            pass

                    return {"content": self._truncate(raw)}
                else:
                    # AX name field path — no DOM calls needed
                    raw = matches[0].get("name", {}).get("value", "")
                    return {"content": self._truncate(raw)}

        # AT-SPI path (atspi and terminal tiers)
        window_node = self._find_atspi_window(app)
        matches = self._bfs_atspi(window_node, role, name)

        if len(matches) == 0:
            raise EnvironmentalError(
                f"element not found: role={role} name={name}"
            )

        if len(matches) >= 2:
            # Build disambiguation payload
            payload = f"Found {len(matches)} elements matching role={role} name={name}:\n"
            for idx, match in enumerate(matches, start=1):
                parent_name = getattr(
                    getattr(match, 'parent', None), 'name', ''
                ) or '<unknown>'
                ext = match.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                payload += (
                    f"[{idx}] parent={parent_name} "
                    f"extents=({ext.x},{ext.y},{ext.width},{ext.height})\n"
                )

            if tier == "terminal":
                raise EnvironmentalError(payload)
            # tier == "atspi": Vision fallback
            marker_coords, marker_to_bfs = self._get_atspi_marker_coords(matches)
            wid = self.registry._registry[app]["wid"]
            selected = self._vision_select(wid, app, role, name, marker_coords)
            node = matches[marker_to_bfs[selected]]
            # No xdotool click — read_element Vision path skips controller call
            raw = self._atspi_read(node, role)
            return {"content": self._truncate(raw)}

        # Exactly 1 match — proceed to read
        raw = self._atspi_read(matches[0], role)
        return {"content": self._truncate(raw)}
