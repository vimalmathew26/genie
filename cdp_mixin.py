"""CDP resolution tier mixin for ElementResolver.

Extracted from element_resolver.py.  Mixed into ElementResolver
so that all self._cdp_* and self._bfs_cdp calls work unchanged.
"""

import json
import time
import urllib.request
from contextlib import contextmanager
from collections import deque

from websockets.sync.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from config import (
    log,
    CDP_RECV_TIMEOUT_SECONDS,
    CDP_EVENT_DISCARD_CAP,
    CDP_ROLE_ALIASES,
)
from exceptions import (
    EnvironmentalError,
    TransientError,
    UnrecoverableError,
)


class CDPMixin:
    """CDP transport, AX tree traversal, and JS helper methods."""

    # -----------------------------------------------------------------
    # CDP transport layer
    # -----------------------------------------------------------------

    @contextmanager
    def _cdp_session(self, cdp_port, url_hint):
        """Context manager yielding a WebSocket connection to the correct
        browser tab identified via CDP's /json endpoint."""
        url = f"http://localhost:{cdp_port}/json"
        try:
            resp = urllib.request.urlopen(url)
            targets = json.loads(resp.read().decode())
        except Exception:
            raise TransientError(
                f"cdp connection refused: {cdp_port}"
            )

        pages = [t for t in targets if t.get("type") == "page"]
        if not pages:
            raise TransientError(
                f"cdp no page targets found on port {cdp_port}"
            )

        if url_hint is not None:
            selected = None
            for p in pages:
                if url_hint in p["url"]:
                    selected = p
                    break
            if selected is None:
                raise EnvironmentalError(
                    f"cdp url_hint '{url_hint}' matched no open tabs"
                )
        else:
            # Prefer real web pages over Chrome internal pages
            web_pages = [
                p for p in pages
                if p.get("url", "").startswith("http")
            ]
            selected = web_pages[0] if web_pages else pages[0]

        ws_url = selected["webSocketDebuggerUrl"]

        try:
            conn = ws_connect(ws_url)
        except Exception:
            raise TransientError(
                f"cdp websocket connect failed: {ws_url}"
            )

        with conn:
            yield conn

    def _cdp_send(self, ws, method: str, params: dict) -> dict:
        """Send one CDP command and return response['result'].

        All CDP commands go through this method.
        """
        self._cdp_id += 1
        cmd_id = self._cdp_id

        payload = json.dumps({"id": cmd_id, "method": method, "params": params})
        try:
            ws.send(payload)
        except ConnectionClosed:
            raise TransientError(
                f"cdp connection closed before send: method={method}"
            )

        discard_count = 0
        while True:
            try:
                raw = ws.recv(timeout=CDP_RECV_TIMEOUT_SECONDS)
            except ConnectionClosed:
                raise TransientError(
                    f"cdp connection closed during recv: method={method}"
                )
            except TimeoutError:
                raise TransientError(
                    f"cdp recv timeout after {CDP_RECV_TIMEOUT_SECONDS}s: method={method}"
                )
            response = json.loads(raw)
            if response.get("id") == cmd_id:
                break
            discard_count += 1
            if discard_count >= CDP_EVENT_DISCARD_CAP:
                raise UnrecoverableError(
                    f"cdp event discard cap exceeded: method={method}"
                )

        if "error" in response:
            error = response["error"]
            msg = error.get("message", "").lower()
            code = error.get("code")
            if "node with given id does not belong to the document" in msg:
                raise TransientError(f"cdp stale node: {msg}")
            elif "no node with given id found" in msg:
                raise TransientError(f"cdp stale node: {msg}")
            elif "cannot find context with specified id" in msg:
                raise TransientError(f"cdp stale context: {msg}")
            elif "method not found" in msg:
                raise UnrecoverableError(f"cdp method not found: {method}")
            elif "invalid parameters" in msg:
                raise UnrecoverableError(f"cdp invalid parameters: {method}")
            else:
                log(
                    f"CDP unclassified error: code={code} message='{msg}' "
                    f"method={method} — defaulting to UnrecoverableError"
                )
                raise UnrecoverableError(
                    f"cdp unclassified error: code={code} message='{msg}'"
                )

        exception_details = (response.get("result") or {}).get("exceptionDetails")
        if exception_details:
            log(
                f"CDP callFunctionOn exception: "
                f"{json.dumps(exception_details)} — raising TransientError"
            )
            raise TransientError(
                f"cdp callFunctionOn exception in method={method}"
            )

        result = response.get("result")
        if result is None:
            log(
                f"CDP null result with no error field: "
                f"method={method} raw={raw}"
            )
            raise UnrecoverableError(
                f"cdp null result: method={method}"
            )

        return result

    def _cdp_type_chars(self, ws, text: str) -> None:
        """Type text via CDP character key events.

        Uses Input.dispatchKeyEvent type='char' per character rather than
        Input.insertText. This fires the browser's JS input/change event
        handlers — required for the Chrome omnibox and React/Angular inputs.
        """
        for char in text:
            self._cdp_send(ws, "Input.dispatchKeyEvent", {
                "type": "char",
                "text": char,
            })

    # -----------------------------------------------------------------
    # CDP AX tree traversal
    # -----------------------------------------------------------------

    def _bfs_cdp(self, ws, role: str, name: str) -> list:
        """BFS all-matches from CDP full AX tree.

        Returns a list of raw AX node dicts where (role, name) both match.

        Special case — name=="":
          Role-only mode. Returns the FIRST node matching the role aliases,
          regardless of name.

        Normal case — name!="":
          Returns all nodes where both role and name match exactly.
        """
        result = self._cdp_send(ws, "Accessibility.getFullAXTree", {})
        nodes = result.get("nodes", [])

        if not nodes:
            raise TransientError("cdp ax tree empty \u2014 page may still be loading")

        node_map = {node["nodeId"]: node for node in nodes}
        aliases = CDP_ROLE_ALIASES.get(role, [role])
        role_only = (name == "")

        matches = []
        name_role_misses = []
        _link_fallback = None
        queue = deque([nodes[0]])

        while queue:
            node = queue.popleft()

            role_str = node.get("role", {}).get("value", "")
            name_str = node.get("name", {}).get("value", "")

            if role_only:
                if role_str in aliases:
                    if role != "link":
                        return [node]
                    link_href = ""
                    for prop in node.get("properties", []):
                        if prop.get("name") == "url":
                            link_href = prop.get("value", {}).get("value", "") or ""
                            break
                    _SEARCH_ENGINE_DOMAINS = (
                        "google.com", "bing.com", "duckduckgo.com",
                        "yahoo.com", "yandex.com", "baidu.com",
                    )
                    _is_external = (
                        (link_href.startswith("http://") or link_href.startswith("https://"))
                        and not any(sd in link_href for sd in _SEARCH_ENGINE_DOMAINS)
                    )
                    if _is_external:
                        return [node]
                    if _link_fallback is None:
                        _link_fallback = node
            else:
                if name_str == name:
                    if role_str in aliases:
                        matches.append(node)
                    else:
                        name_role_misses.append((name_str, role_str))

            for child_id in node.get("childIds", []):
                child = node_map.get(child_id)
                if child is not None:
                    queue.append(child)

        if role_only and role == "link" and _link_fallback is not None:
            return [_link_fallback]

        if not role_only and len(matches) == 0:
            for missed_name, missed_role in name_role_misses:
                log(
                    f"CDP BFS role-miss: name='{missed_name}' matched but "
                    f"role='{missed_role}' not in CDP_ROLE_ALIASES "
                    f"for canonical '{role}'"
                )

        return matches

    # -----------------------------------------------------------------
    # CDP coordinate resolution
    # -----------------------------------------------------------------

    def _resolve_cdp_coords(self, ws, backend_node_id: int) -> tuple[float, float]:
        """Resolve a DOM node's viewport-relative center coordinates.

        Returns (cdp_x, cdp_y) as floats.
        """
        # Step 1 — DOM.resolveNode: obtain JS objectId
        resolve_result = self._cdp_send(
            ws,
            "DOM.resolveNode",
            {"backendNodeId": backend_node_id},
        )
        object_id = resolve_result["object"]["objectId"]

        # Step 2 — scrollIntoView, Step 3 — releaseObject in finally
        try:
            self._cdp_send(
                ws,
                "Runtime.callFunctionOn",
                {
                    "objectId": object_id,
                    "functionDeclaration": "function(){this.scrollIntoView({block:'center'})}",
                    "silent": True,
                },
            )
        finally:
            try:
                self._cdp_send(
                    ws,
                    "Runtime.releaseObject",
                    {"objectId": object_id},
                )
            except Exception:
                log(f"DEBUG: releaseObject failed for objectId={object_id} "
                    f"— swallowing silently")

        # Step 4 — DOM.getBoxModel
        box_result = self._cdp_send(
            ws,
            "DOM.getBoxModel",
            {"backendNodeId": backend_node_id},
        )

        # Step 5 — center extraction from content quad
        content = box_result["model"]["content"]
        cdp_x = (content[0] + content[2] + content[4] + content[6]) / 4
        cdp_y = (content[1] + content[3] + content[5] + content[7]) / 4

        return (cdp_x, cdp_y)

    # -----------------------------------------------------------------
    # CDP navigation
    # -----------------------------------------------------------------

    def _cdp_navigate(self, ws, url: str, timeout: float = 10.0) -> None:
        """Call Page.navigate and wait for the response."""
        self._cdp_send(ws, "Page.navigate", {"url": url})
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                raw = ws.recv(timeout=max(0.3, deadline - time.time()))
                msg = json.loads(raw)
                if msg.get("method") == "Page.frameNavigated":
                    break
                if msg.get("method") in ("Page.loadEventFired",
                                         "Page.domContentEventFired"):
                    break
            except Exception:
                break

    # -----------------------------------------------------------------
    # JS constants
    # -----------------------------------------------------------------

    _SERP_RESULTS_JS = (
        "(function() {"
        "  var results = [];"
        "  var seen = {};"
        "  function add(url) {"
        "    if (!url || !url.startsWith('http') || seen[url]) return;"
        "    seen[url] = true;"
        "    results.push(url);"
        "  }"
        "  var h3s = document.querySelectorAll('#rso .g h3 a[href]');"
        "  for (var i = 0; i < h3s.length; i++) {"
        "    if (h3s[i].closest('[aria-label=\"Ad\"]')) continue;"
        "    var raw = h3s[i].getAttribute('href') || '';"
        "    if (!raw || raw === '#' || raw.indexOf('javascript:') !== -1) continue;"
        "    var resolved = h3s[i].href || '';"
        "    if (resolved.indexOf('/url?') !== -1) {"
        "      var m = resolved.match(/[?&]q=([^&]+)/);"
        "      if (m) resolved = decodeURIComponent(m[1]);"
        "    }"
        "    add(resolved);"
        "  }"
        "  if (results.length > 0) return JSON.stringify(results);"
        "  var rso = document.querySelectorAll('#rso a[href]');"
        "  for (var i = 0; i < rso.length; i++) {"
        "    var resolved = rso[i].href || '';"
        "    if (resolved.indexOf('/url?') !== -1) {"
        "      var m = resolved.match(/[?&]q=([^&]+)/);"
        "      if (m) resolved = decodeURIComponent(m[1]);"
        "    }"
        "    if (resolved.startsWith('http')"
        "        && resolved.indexOf('google.com') === -1)"
        "      add(resolved);"
        "  }"
        "  if (results.length > 0) return JSON.stringify(results);"
        "  var bing = document.querySelectorAll("
        "    '#b_results .b_algo h2 a[href],"
        "    #b_results .b_algo .b_title a[href]'"
        "  );"
        "  for (var i = 0; i < bing.length; i++) {"
        "    if (bing[i].closest('.b_ad, [data-tag=\"ad\"]')) continue;"
        "    var resolved = bing[i].href || '';"
        "    if (resolved.indexOf('bing.com') === -1) add(resolved);"
        "  }"
        "  if (results.length > 0) return JSON.stringify(results);"
        "  var ddg = document.querySelectorAll("
        "    '[data-testid=\"result-title-a\"],"
        "    .result__a,"
        "    .react-results--main a.result__a,"
        "    article[data-testid=\"result\"] h2 a[href],"
        "    ol.react-results--main li h2 a[href],"
        "    section[data-testid=\"mainline\"] a[href],"
        "    [data-nrn=\"result\"] a[href],"
        "    .results--main a[href],"
        "    li[data-layout=\"organic\"] h2 a[href],"
        "    li[data-layout=\"organic\"] a[data-testid=\"result-extras-url-link\"]'"
        "  );"
        "  for (var i = 0; i < ddg.length; i++) {"
        "    var resolved = ddg[i].href || '';"
        "    if (resolved.indexOf('duckduckgo.com') === -1) add(resolved);"
        "  }"
        "  if (results.length > 0) return JSON.stringify(results);"
        "  var generic = document.querySelectorAll("
        "    'main a[href], #search a[href], #results a[href], article a[href]'"
        "  );"
        "  for (var i = 0; i < generic.length; i++) {"
        "    var resolved = generic[i].href || '';"
        "    if (resolved.startsWith('http')"
        "        && resolved.indexOf('google.com') === -1"
        "        && resolved.indexOf('bing.com') === -1"
        "        && resolved.indexOf('duckduckgo.com') === -1)"
        "      add(resolved);"
        "  }"
        "  if (results.length > 0) return JSON.stringify(results);"
        "  var _sDomains=['google.com','bing.com','duckduckgo.com','yahoo.com','yandex.com'];"
        "  var _all=document.querySelectorAll('a[href]');"
        "  for(var i=0;i<_all.length;i++){"
        "    var _el=_all[i];"
        "    var _r=_el.getBoundingClientRect();"
        "    if(_r.width<10||_r.height<10) continue;"
        "    var _href=_el.href||'';"
        "    if(!_href.startsWith('http')) continue;"
        "    var _skip=false;"
        "    for(var d=0;d<_sDomains.length;d++){if(_href.indexOf(_sDomains[d])!==-1){_skip=true;break;}}"
        "    if(_skip) continue;"
        "    var _anc=_el.parentElement,_isNav=false;"
        "    while(_anc){var _t=_anc.tagName;if(_t==='NAV'||_t==='HEADER'||_t==='FOOTER'){_isNav=true;break;}_anc=_anc.parentElement;}"
        "    if(!_isNav) add(_href);"
        "  }"
        "  return JSON.stringify(results);"
        "})()"
    )

    _COOKIE_DISMISS_JS = (
        "(function() {"
        "  var labels = ["
        "    'accept all', 'accept cookies', 'accept', 'agree',"
        "    'i agree', 'allow all', 'allow cookies', 'got it',"
        "    'consent', 'ok', 'yes, i agree', 'continue',"
        "    'i accept', 'acknowledge'"
        "  ];"
        "  var sels = 'button, a, [role=button], input[type=button],"
        "             input[type=submit], [class*=consent] button,"
        "             [class*=cookie] button, [id*=cookie] button,"
        "             [class*=accept], [id*=accept]';"
        "  var els = document.querySelectorAll(sels);"
        "  for (var i = 0; i < els.length; i++) {"
        "    var el = els[i];"
        "    var txt = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().toLowerCase();"
        "    for (var j = 0; j < labels.length; j++) {"
        "      if (txt === labels[j]) { el.click(); return 'dismissed:' + txt; }"
        "    }"
        "  }"
        "  return '';"
        "})()"
    )

    _PAGE_LINKS_JS = (
        "(function() {"
        "  var domain = location.hostname;"
        "  var content = [];"
        "  var nav = [];"
        "  var links = document.querySelectorAll('a[href]');"
        "  for (var i = 0; i < links.length; i++) {"
        "    var el = links[i];"
        "    if (el.offsetParent === null && getComputedStyle(el).position !== 'fixed') continue;"
        "    var txt = (el.innerText || '').trim();"
        "    if (!txt || txt.length < 2) continue;"
        "    var href = el.href || '';"
        "    var isNav = false;"
        "    try {"
        "      var u = new URL(href);"
        "      if (u.hostname === domain || u.hostname.endsWith('.' + domain)) isNav = true;"
        "    } catch(e) { isNav = true; }"
        "    var anc = el.parentElement;"
        "    while (anc) {"
        "      var tag = anc.tagName;"
        "      if (tag === 'NAV' || tag === 'HEADER' || tag === 'FOOTER') { isNav = true; break; }"
        "      anc = anc.parentElement;"
        "    }"
        "    if (isNav) { nav.push({text: txt, href: href}); }"
        "    else { content.push({text: txt, href: href}); }"
        "  }"
        "  return JSON.stringify(content.concat(nav));"
        "})()"
    )

    _FIND_INPUT_JS = (
        "(function(){"
        "  var el=document.querySelector('#sb_form_q');"
        "  if(el&&!el.disabled&&el.offsetParent!==null) return el;"
        "  el=document.querySelector('#sb_form textarea, #searchbox textarea,"
        "    textarea[name=\"q\"], .b_searchbox');"
        "  if(el&&!el.disabled&&el.offsetParent!==null) return el;"
        "  el=document.querySelector('#sb_form input:not([type=\"hidden\"]),"
        "    #sb_form textarea, form[action*=\"bing\"] input:not([type=\"hidden\"]),"
        "    form[action*=\"bing\"] textarea');"
        "  if(el&&!el.disabled&&el.offsetParent!==null) return el;"
        "  el=document.querySelector('#searchbox_input, #search_form_input,"
        "    input[name=\"q\"][data-testid],"
        "    [data-testid=\"searchbox_input\"]');"
        "  if(el&&!el.disabled&&el.offsetParent!==null) return el;"
        "  el=document.querySelector('input[name=\"q\"]:not([type=\"hidden\"])');"
        "  if(el&&!el.disabled&&el.offsetParent!==null) return el;"
        "  el=document.querySelector('input[type=\"search\"]');"
        "  if(el&&!el.disabled&&el.offsetParent!==null) return el;"
        "  var tas=document.querySelectorAll('textarea');"
        "  for(var i=0;i<tas.length;i++){"
        "    var r=tas[i].getBoundingClientRect();"
        "    if(r.width>50&&r.height>10&&!tas[i].disabled) return tas[i];"
        "  }"
        "  el=document.querySelector('[role=\"searchbox\"],[role=\"combobox\"]');"
        "  if(el&&!el.disabled&&el.offsetParent!==null) return el;"
        "  var all=document.querySelectorAll('input[type=\"text\"],input:not([type]),textarea');"
        "  for(var i=0;i<all.length;i++){"
        "    var r=all[i].getBoundingClientRect();"
        "    if(r.width>0&&r.height>0&&!all[i].disabled) return all[i];"
        "  }"
        "  return null;"
        "})()"
    )

    # -----------------------------------------------------------------
    # JS helper methods
    # -----------------------------------------------------------------

    def _cdp_js_nth_result_url(self, ws, n: int = 0) -> str | None:
        """Return the URL of the Nth (0-based) organic search result on the
        current CDP page, or None if fewer than N+1 results / any failure."""
        try:
            eval_result = self._cdp_send(ws, "Runtime.evaluate", {
                "expression": self._SERP_RESULTS_JS,
                "returnByValue": True,
            })
            raw = eval_result.get("result", {}).get("value", "") or ""
            if not raw:
                return None
            urls = json.loads(raw)
            if not isinstance(urls, list) or n >= len(urls):
                return None
            url = urls[n]
            return url if url.startswith("http") else None
        except Exception:
            return None

    def _cdp_js_read_serp_results(self, ws, max_results: int = 10) -> list[dict]:
        """Read visible search result titles and URLs from the current page via JS."""
        JS = (
            "(function() {"
            "  var out = [];"
            "  var h3s = document.querySelectorAll('#rso .g h3');"
            "  for (var i = 0; i < h3s.length; i++) {"
            "    var a = h3s[i].closest('a') || h3s[i].querySelector('a');"
            "    if (!a) continue;"
            "    var t = h3s[i].innerText || '';"
            "    var u = a.href || '';"
            "    if (u && u.startsWith('http')) out.push({title: t, url: u});"
            "  }"
            "  if (out.length > 0) return JSON.stringify(out);"
            "  var bing = document.querySelectorAll("
            "    '#b_results .b_algo h2 a[href],"
            "    #b_results .b_algo .b_title a[href]'"
            "  );"
            "  for (var i = 0; i < bing.length; i++) {"
            "    if (bing[i].closest('.b_ad, [data-tag=\"ad\"]')) continue;"
            "    var t = bing[i].innerText || '';"
            "    var u = bing[i].href || '';"
            "    if (u && u.indexOf('bing.com') === -1) out.push({title: t, url: u});"
            "  }"
            "  if (out.length > 0) return JSON.stringify(out);"
            "  var ddg = document.querySelectorAll("
            "    '[data-testid=\"result-title-a\"],"
            "    .result__a,"
            "    article[data-testid=\"result\"] h2 a[href]'"
            "  );"
            "  for (var i = 0; i < ddg.length; i++) {"
            "    var t = ddg[i].innerText || '';"
            "    var u = ddg[i].href || '';"
            "    if (u && u.indexOf('duckduckgo.com') === -1) out.push({title: t, url: u});"
            "  }"
            "  return JSON.stringify(out);"
            "})()"
        )
        try:
            eval_result = self._cdp_send(ws, "Runtime.evaluate", {
                "expression": JS,
                "returnByValue": True,
            })
            raw = eval_result.get("result", {}).get("value", "") or "[]"
            items = json.loads(raw)
            results = []
            for idx, item in enumerate(items[:max_results]):
                results.append({
                    "index": idx,
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                })
            return results
        except Exception:
            return []

    def _cdp_dismiss_cookie_banner(self, ws) -> bool:
        """Attempt to dismiss any cookie-consent overlay via JS click."""
        try:
            result = self._cdp_send(ws, "Runtime.evaluate", {
                "expression": self._COOKIE_DISMISS_JS,
                "returnByValue": True,
            })
            val = result.get("result", {}).get("value", "") or ""
            if val:
                log(f"_cdp_dismiss_cookie_banner: {val}")
                return True
            return False
        except Exception:
            return False

    def _cdp_js_read_page_links(self, ws, index: int = 0) -> str | None:
        """Return the visible text of the Nth content link on the page via JS."""
        try:
            eval_result = self._cdp_send(ws, "Runtime.evaluate", {
                "expression": self._PAGE_LINKS_JS,
                "returnByValue": True,
            })
            raw = eval_result.get("result", {}).get("value", "") or "[]"
            items = json.loads(raw)
            if isinstance(items, list) and index < len(items):
                text = items[index].get("text", "") or None
                if text:
                    return text
        except Exception:
            pass
        return None

    def _cdp_js_read_main_content(self, ws, role: str = "",
                                  name: str = "") -> str:
        """Extract main text content from the current page via JS.

        Heuristic-based: tries common content containers in order,
        then falls back to the first substantial <p> on the page.
        """
        # Dismiss any cookie-consent overlay before reading content.
        self._cdp_dismiss_cookie_banner(ws)

        JS = (
            "(function() {"
            "  var wp = document.querySelectorAll('#mw-content-text .mw-parser-output > p');"
            "  for (var i = 0; i < wp.length; i++) {"
            "    var t = wp[i].innerText.trim();"
            "    if (t.length > 80) return t;"
            "  }"
            "  var soSels = '#answers .s-prose p, .answercell .post-text p,"
            "               .answer .s-prose p, .answer .post-text p';"
            "  var so = document.querySelectorAll(soSels);"
            "  for (var i = 0; i < so.length; i++) {"
            "    var t = so[i].innerText.trim();"
            "    if (t.length > 80) return t;"
            "  }"
            "  var art = document.querySelector('article');"
            "  if (art) {"
            "    var ps = art.querySelectorAll('p');"
            "    for (var i = 0; i < ps.length; i++) {"
            "      var t = ps[i].innerText.trim();"
            "      if (t.length > 80) return t;"
            "    }"
            "  }"
            "  var main = document.querySelector('main');"
            "  if (main) {"
            "    var ps = main.querySelectorAll('p');"
            "    for (var i = 0; i < ps.length; i++) {"
            "      var t = ps[i].innerText.trim();"
            "      if (t.length > 80) return t;"
            "    }"
            "  }"
            "  var all = document.querySelectorAll('p');"
            "  for (var i = 0; i < all.length; i++) {"
            "    var t = all[i].innerText.trim();"
            "    if (t.length > 80) return t;"
            "  }"
            "  var hSels = ['h1','h2','h3'];"
            "  for (var hi = 0; hi < hSels.length; hi++) {"
            "    var hs = document.querySelectorAll(hSels[hi]);"
            "    for (var i = 0; i < hs.length; i++) {"
            "      var t = hs[i].innerText.trim();"
            "      if (t.length > 20) return t;"
            "    }"
            "  }"
            "  return '';"
            "})()"
        )
        try:
            eval_result = self._cdp_send(ws, "Runtime.evaluate", {
                "expression": JS,
                "returnByValue": True,
            })
            raw = eval_result.get("result", {}).get("value", "") or ""
            if raw:
                log(f"_cdp_js_read_main_content: extracted {len(raw)} chars")
                return raw
            time.sleep(2)
            eval_result = self._cdp_send(ws, "Runtime.evaluate", {
                "expression": JS,
                "returnByValue": True,
            })
            raw = eval_result.get("result", {}).get("value", "") or ""
            if raw:
                log(f"_cdp_js_read_main_content: extracted {len(raw)} chars (retry after dismiss)")
                return raw

            _BRUTE_JS = (
                "(function(){"
                "  var t=(document.body||document.documentElement).innerText||'';"
                "  return t.substring(0, 8000);"
                "})()"
            )
            try:
                brute_result = self._cdp_send(ws, "Runtime.evaluate", {
                    "expression": _BRUTE_JS,
                    "returnByValue": True,
                })
                brute_raw = brute_result.get("result", {}).get("value", "") or ""
                if len(brute_raw.strip()) > 100:
                    log(f"_cdp_js_read_main_content: brute-force fallback returned {len(brute_raw)} chars")
                    return brute_raw
            except Exception:
                pass

            return ""
        except Exception:
            return ""

    def _cdp_js_focus_best_input(self, ws) -> bool:
        """Use JS to find and focus the best search/text input on the current
        page, bypassing Accessibility.getFullAXTree."""
        try:
            eval_result = self._cdp_send(ws, "Runtime.evaluate", {
                "expression": self._FIND_INPUT_JS,
                "returnByValue": False,
            })
            object_id = (
                eval_result.get("result", {})
                            .get("objectId")
            )
            if not object_id:
                return False
            try:
                self._cdp_send(ws, "Runtime.callFunctionOn", {
                    "objectId": object_id,
                    "functionDeclaration": (
                        "function(){"
                        "  this.scrollIntoView({block:'center'});"
                        "  this.click();"
                        "  this.focus();"
                        "}"
                    ),
                    "silent": True,
                })
            finally:
                try:
                    self._cdp_send(ws, "Runtime.releaseObject",
                                   {"objectId": object_id})
                except Exception:
                    pass
            return True
        except Exception:
            return False
