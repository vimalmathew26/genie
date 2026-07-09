"""GTK4 chat widgets for the Genie UI. Extracted from genie.py."""

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib, Gdk, Pango

import datetime
import os
import threading
import time

from orchestrator import TaskResult
import persistence
from config import CHECKPOINT_PATH, log


# =============================================================================
# Per-message chat widget
# =============================================================================

class MessageWidget(Gtk.Box):
    """Discrete per-message widget rendered inside the chat ListBox.

    Contains a header row (sender, timestamp, copy button) and a body
    (read-only Gtk.TextView for selectable text).
    """

    def __init__(self, sender: str, text: str, msg_type: str = "chat",
                 window=None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self._sender = sender
        self._text = text
        self._msg_type = msg_type
        self._window = window
        self._body_tv: Gtk.TextView | None = None

        is_user = (sender == "You")

        # Base style
        self.add_css_class("msg-widget")
        if msg_type == "chat" and is_user:
            self.add_css_class("msg-user")
            self.set_halign(Gtk.Align.END)
        elif msg_type == "action":
            self.add_css_class("msg-action")
            self.set_halign(Gtk.Align.START)
        elif msg_type == "error":
            self.add_css_class("msg-error")
            self.set_halign(Gtk.Align.START)
        elif msg_type in ("cost", "task_summary"):
            self.add_css_class("msg-summary")
            self.set_halign(Gtk.Align.FILL)
        else:
            self.add_css_class("msg-genie")
            self.set_halign(Gtk.Align.START)

        # ---- Header row ----
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.add_css_class("msg-header")

        sender_lbl = Gtk.Label(label=sender)
        sender_lbl.add_css_class("msg-sender")
        sender_lbl.add_css_class("msg-sender-user" if is_user else "msg-sender-genie")
        sender_lbl.set_halign(Gtk.Align.START)
        header.append(sender_lbl)

        ts = datetime.datetime.now().strftime("%H:%M:%S")

        ts_lbl = Gtk.Label(label=ts)
        ts_lbl.add_css_class("msg-timestamp")
        ts_lbl.set_halign(Gtk.Align.START)
        header.append(ts_lbl)

        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        header.append(spacer)

        self._copy_btn = Gtk.Button.new_from_icon_name("edit-copy-symbolic")
        self._copy_btn.add_css_class("msg-copy-btn")
        self._copy_btn.add_css_class("flat")
        self._copy_btn.set_tooltip_text("Copy message")
        self._copy_btn.connect("clicked", self._on_copy)
        header.append(self._copy_btn)

        self.append(header)

        # ---- Body ----
        if msg_type in ("cost", "task_summary"):
            self._build_summary_body(text)
        else:
            self._build_text_body(text, msg_type)

    def _build_text_body(self, text: str, msg_type: str) -> None:
        tv = Gtk.TextView()
        tv.set_editable(False)
        tv.set_cursor_visible(False)
        tv.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        tv.add_css_class("msg-body")
        if msg_type == "action":
            tv.add_css_class("msg-body-action")
        elif msg_type == "error":
            tv.add_css_class("msg-body-error")
        tv.get_buffer().set_text(text)
        tv.set_left_margin(0)
        tv.set_right_margin(0)
        tv.set_top_margin(0)
        tv.set_bottom_margin(0)
        self.append(tv)
        self._setup_body_interactions(tv)

    def _build_summary_body(self, text: str) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.add_css_class("msg-summary-body")

        lower = text.lower()
        if "success" in lower:
            badge_text, badge_cls = "\u2713 SUCCESS", "badge-success"
        elif "fail" in lower or "error" in lower:
            badge_text, badge_cls = "\u2717 FAILED", "badge-fail"
        else:
            badge_text, badge_cls = "\u2014 DONE", "badge-done"

        badge = Gtk.Label(label=badge_text)
        badge.add_css_class("msg-badge")
        badge.add_css_class(badge_cls)
        badge.set_halign(Gtk.Align.START)
        box.append(badge)

        tv = Gtk.TextView()
        tv.set_editable(False)
        tv.set_cursor_visible(False)
        tv.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        tv.add_css_class("msg-body")
        tv.add_css_class("msg-body-summary")
        tv.get_buffer().set_text(text)
        tv.set_left_margin(0)
        tv.set_right_margin(0)
        tv.set_top_margin(2)
        tv.set_bottom_margin(0)
        box.append(tv)
        self._setup_body_interactions(tv)

        self.append(box)

    def _on_copy(self, _btn) -> None:
        clipboard = Gdk.Display.get_default().get_clipboard()
        clipboard.set(self._text)
        self._copy_btn.set_icon_name("object-select-symbolic")
        GLib.timeout_add(1500, self._restore_copy_icon)

    def _restore_copy_icon(self) -> bool:
        self._copy_btn.set_icon_name("edit-copy-symbolic")
        return False

    # ---- Right-click context menu & selection helpers -----------------------

    def _setup_body_interactions(self, tv: Gtk.TextView) -> None:
        """Attach right-click context menu and suppress default GTK menu."""
        self._body_tv = tv
        # Capture right-click before the built-in TextView handler
        gesture = Gtk.GestureClick()
        gesture.set_button(3)
        gesture.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        gesture.connect("pressed", self._on_right_click)
        tv.add_controller(gesture)

    def _on_right_click(self, gesture, n_press, x, y):
        """Show custom context popover on right-click."""
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        tv = self._body_tv
        if tv is None:
            return

        popover = Gtk.Popover()
        popover.set_parent(tv)
        popover.add_css_class("msg-context-popover")
        rect = Gdk.Rectangle()
        rect.x, rect.y, rect.width, rect.height = int(x), int(y), 1, 1
        popover.set_pointing_to(rect)
        popover.set_has_arrow(True)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # -- Copy selection (or full body if nothing selected) --
        copy_sel_btn = Gtk.Button(label="Copy selection")
        copy_sel_btn.add_css_class("flat")
        copy_sel_btn.add_css_class("context-menu-item")
        def _copy_sel(_b):
            Gdk.Display.get_default().get_clipboard().set(self._get_selected_or_all())
            popover.popdown()
        copy_sel_btn.connect("clicked", _copy_sel)
        box.append(copy_sel_btn)

        # -- Copy all --
        copy_all_btn = Gtk.Button(label="Copy all")
        copy_all_btn.add_css_class("flat")
        copy_all_btn.add_css_class("context-menu-item")
        def _copy_all(_b):
            Gdk.Display.get_default().get_clipboard().set(self._text)
            popover.popdown()
        copy_all_btn.connect("clicked", _copy_all)
        box.append(copy_all_btn)

        # -- Add to prompt --
        add_btn = Gtk.Button(label="Add to prompt")
        add_btn.add_css_class("flat")
        add_btn.add_css_class("context-menu-item")
        def _add_to_prompt(_b):
            text = self._get_selected_or_all()
            if self._window is not None:
                self._window._inject_into_prompt(text)
            popover.popdown()
        add_btn.connect("clicked", _add_to_prompt)
        box.append(add_btn)

        popover.set_child(box)
        popover.connect("closed", lambda p: p.unparent())
        popover.popup()

    def _get_selected_or_all(self) -> str:
        """Return selected text from body TextView, or full body text."""
        if self._body_tv is not None:
            buf = self._body_tv.get_buffer()
            bounds = buf.get_selection_bounds()
            if bounds:
                start, end = bounds
                return buf.get_text(start, end, False)
        return self._text


# =============================================================================
# Activity label helper (Change 5)
# =============================================================================

def _format_action_status(action: str, args: dict | None) -> str:
    """Build a one-liner from action name + its most meaningful arg."""
    if args is None:
        return action

    def _tail(path: str, n: int = 2) -> str:
        parts = path.replace("\\", "/").rstrip("/").split("/")
        return "/".join(parts[-n:]) if len(parts) >= n else path

    if action in ("write_file", "read_file", "append_file", "delete_file"):
        p = args.get("path", "")
        return f"{action} \u2192 {_tail(p)}" if p else action

    if action == "run_command":
        cmd = args.get("cmd", "")
        return f"{action} \u2192 {cmd[:50]}" if cmd else action

    if action in ("click_element", "type_element", "read_element"):
        app = args.get("app", "")
        role = args.get("role", "")
        name = args.get("name", "")
        detail = " ".join(filter(None, [app, role, f"'{name}'" if name else ""]))
        return f"{action} \u2192 {detail}" if detail else action

    if action in ("open_app", "focus_window"):
        app = args.get("app", "")
        return f"{action} \u2192 {app}" if app else action

    if action == "press_key":
        key = args.get("key", "")
        return f"{action} \u2192 {key}" if key else action

    if action == "type_text":
        text = args.get("text", "")
        t = text[:30] + ("\u2026" if len(text) > 30 else "")
        return f"{action} \u2192 '{t}'" if text else action

    if action == "fetch_url":
        url = args.get("url", "")
        # Strip protocol prefix
        for prefix in ("https://", "http://"):
            if url.startswith(prefix):
                url = url[len(prefix):]
                break
        t = url[:45] + ("\u2026" if len(url) > 45 else "")
        return f"{action} \u2192 {t}" if url else action

    if action == "look":
        app = args.get("app", "")
        return f"{action} \u2192 {app}" if app else action

    if action in ("list_dir", "index_codebase"):
        p = args.get("path", "")
        return f"{action} \u2192 {p}" if p else action

    return action


# =============================================================================
# Nested task disclosure widgets
# =============================================================================

class IterationWidget(Gtk.Box):
    """One brain-loop iteration: header row + collapsible action detail list.

    Header shows iteration number, phase icon (animated spinner while running,
    checkmark/cross on completion), comma-separated action names, and a toggle.
    """
    _SPIN_FRAMES = "\u25d0\u25d3\u25d1\u25d2"  # ◐◓◑◒

    def __init__(self, iter_n: int):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add_css_class("iter-widget")
        self._iter_n = iter_n
        self._action_names: list[str] = []
        self._spin_idx: int = 0
        self._spin_timer_id: int | None = None

        # ---- Header row ----
        self._header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._header.add_css_class("iter-header")

        # Phase icon — animated quarter-circle while running
        self._phase_icon = Gtk.Label(label=self._SPIN_FRAMES[0])
        self._phase_icon.add_css_class("iter-icon")
        self._phase_icon.add_css_class("iter-icon-running")
        self._header.append(self._phase_icon)
        # Start spinning
        self._spin_timer_id = GLib.timeout_add(200, self._spin_tick)

        num_lbl = Gtk.Label(label=f"Iter {iter_n}")
        num_lbl.add_css_class("iter-number")
        num_lbl.set_halign(Gtk.Align.START)
        self._header.append(num_lbl)

        self._summary_lbl = Gtk.Label(label="")
        self._summary_lbl.add_css_class("iter-summary")
        self._summary_lbl.set_halign(Gtk.Align.START)
        self._summary_lbl.set_hexpand(True)
        self._summary_lbl.set_ellipsize(Pango.EllipsizeMode.END)
        self._header.append(self._summary_lbl)

        self._toggle_icon = Gtk.Image.new_from_icon_name("pan-down-symbolic")
        self._toggle_icon.add_css_class("iter-toggle")
        self._header.append(self._toggle_icon)

        gesture = Gtk.GestureClick()
        gesture.connect("released", lambda g, n, x, y: self._toggle())
        self._header.add_controller(gesture)

        self.append(self._header)

        # ---- Revealer with detail rows (starts expanded) ----
        self._revealer = Gtk.Revealer()
        self._revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self._revealer.set_transition_duration(150)
        self._revealer.set_reveal_child(True)

        self._detail_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
        self._revealer.set_child(self._detail_box)
        self.append(self._revealer)

    # -- spin animation -------------------------------------------------------

    def _spin_tick(self) -> bool:
        self._spin_idx += 1
        self._phase_icon.set_label(
            self._SPIN_FRAMES[self._spin_idx % len(self._SPIN_FRAMES)]
        )
        return True

    # -- public API -----------------------------------------------------------

    def add_action(self, action_name: str, result_str: str) -> None:
        """Append an action detail row and update the header summary."""
        self._action_names.append(action_name)
        self._summary_lbl.set_text(", ".join(self._action_names))

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        row.add_css_class("iter-detail-row")

        truncated = result_str[:120] + ("\u2026" if len(result_str) > 120 else "")
        detail_lbl = Gtk.Label(label=f"{action_name} \u2192 {truncated}")
        detail_lbl.add_css_class("iter-detail-text")
        detail_lbl.set_halign(Gtk.Align.START)
        detail_lbl.set_ellipsize(Pango.EllipsizeMode.END)
        detail_lbl.set_hexpand(True)
        row.append(detail_lbl)

        self._detail_box.append(row)

    def set_expanded(self, expanded: bool) -> None:
        """Programmatic expand/collapse."""
        self._revealer.set_reveal_child(expanded)
        icon = "pan-down-symbolic" if expanded else "pan-end-symbolic"
        self._toggle_icon.set_from_icon_name(icon)

    def set_complete(self, had_error: bool = False) -> None:
        """Stop spin animation and set final icon (✓ or ✗)."""
        if self._spin_timer_id is not None:
            GLib.source_remove(self._spin_timer_id)
            self._spin_timer_id = None
        self._phase_icon.remove_css_class("iter-icon-running")
        if had_error:
            self._phase_icon.set_label("\u2717")  # ✗
            self._phase_icon.add_css_class("iter-icon-error")
        else:
            self._phase_icon.set_label("\u2713")  # ✓
            self._phase_icon.add_css_class("iter-icon-ok")

    # -- toggle ---------------------------------------------------------------

    def _toggle(self) -> None:
        revealed = self._revealer.get_reveal_child()
        self._revealer.set_reveal_child(not revealed)
        icon = "pan-down-symbolic" if not revealed else "pan-end-symbolic"
        self._toggle_icon.set_from_icon_name(icon)


class CollapsibleTaskWidget(Gtk.Box):
    """Full task run: outer header + collapsible list of IterationWidgets.

    Header shows the task goal (truncated), final stats (populated on
    completion), a progress bar, and a toggle button.  Starts expanded.
    """

    def __init__(self, goal: str):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add_css_class("task-widget")
        self._goal = goal
        self._iterations: dict[int, IterationWidget] = {}
        self._running = True
        self._last_iter_n: int | None = None

        # ---- Outer header ----
        self._header = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self._header.add_css_class("task-header")

        top_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        truncated_goal = goal[:80] + ("\u2026" if len(goal) > 80 else "")
        self._goal_lbl = Gtk.Label(label=truncated_goal)
        self._goal_lbl.add_css_class("task-goal")
        self._goal_lbl.set_halign(Gtk.Align.START)
        self._goal_lbl.set_hexpand(True)
        self._goal_lbl.set_ellipsize(Pango.EllipsizeMode.END)
        top_row.append(self._goal_lbl)

        self._toggle_icon = Gtk.Image.new_from_icon_name("pan-down-symbolic")
        self._toggle_icon.add_css_class("task-toggle")
        top_row.append(self._toggle_icon)

        self._header.append(top_row)

        gesture = Gtk.GestureClick()
        gesture.connect("released", lambda g, n, x, y: self._toggle())
        self._header.add_controller(gesture)

        # Stats row (hidden until complete)
        self._stats_lbl = Gtk.Label(label="")
        self._stats_lbl.add_css_class("task-stats")
        self._stats_lbl.set_halign(Gtk.Align.START)
        self._stats_lbl.set_visible(False)
        self._header.append(self._stats_lbl)

        # Progress bar (hidden until first update_progress call)
        self._progress_bar = Gtk.ProgressBar()
        self._progress_bar.add_css_class("task-progress-bar")
        self._progress_bar.set_visible(False)
        self._header.append(self._progress_bar)

        self.append(self._header)

        # ---- Revealer with iteration list (starts expanded) ----
        self._revealer = Gtk.Revealer()
        self._revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self._revealer.set_transition_duration(200)
        self._revealer.set_reveal_child(True)

        self._iter_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self._iter_box.set_margin_top(4)
        self._iter_box.set_margin_bottom(4)
        self._revealer.set_child(self._iter_box)
        self.append(self._revealer)

    # -- public API -----------------------------------------------------------

    def add_subtask_header(self, n: int, description: str, total: int) -> None:
        """Insert a visible divider row marking the start of a subtask."""
        truncated = description[:100] + ("\u2026" if len(description) > 100 else "")
        lbl = Gtk.Label(label=f"Subtask {n}/{total}: {truncated}")
        lbl.add_css_class("subtask-header")
        lbl.set_halign(Gtk.Align.FILL)
        lbl.set_hexpand(True)
        lbl.set_xalign(0.0)
        lbl.set_wrap(False)
        lbl.set_ellipsize(Pango.EllipsizeMode.END)
        self._iter_box.append(lbl)
        # Expand the revealer so the header is immediately visible
        if not self._revealer.get_reveal_child():
            self._revealer.set_reveal_child(True)
            self._toggle_icon.set_from_icon_name("pan-down-symbolic")

    def add_subtask_summary(self, message: str) -> None:
        """Append a compact completion-summary row after a subtask's iterations."""
        lbl = Gtk.Label(label=message)
        lbl.add_css_class("subtask-summary")
        lbl.set_halign(Gtk.Align.FILL)
        lbl.set_hexpand(True)
        lbl.set_xalign(0.0)
        lbl.set_wrap(True)
        lbl.set_max_width_chars(120)
        self._iter_box.append(lbl)

    def add_action(self, iter_n: int, action_name: str, result_str: str,
                   had_error: bool = False) -> None:
        """Route an action into the correct IterationWidget (create if new).

        When a new iteration arrives, collapse the previous one and mark it
        complete so only the current iteration stays expanded.
        """
        if iter_n not in self._iterations:
            # Collapse + complete previous iteration
            if self._last_iter_n is not None and self._last_iter_n in self._iterations:
                prev = self._iterations[self._last_iter_n]
                prev.set_complete(had_error)
                prev.set_expanded(False)
            iw = IterationWidget(iter_n)
            self._iterations[iter_n] = iw
            self._iter_box.append(iw)
            self._last_iter_n = iter_n
        self._iterations[iter_n].add_action(action_name, result_str)

    def update_progress(self, fraction: float) -> None:
        """Set progress bar fraction and make it visible."""
        self._progress_bar.set_fraction(max(0.0, min(1.0, fraction)))
        if not self._progress_bar.get_visible():
            self._progress_bar.set_visible(True)

    def complete(self, outcome: str, iterations: int,
                 cost_usd: float, wall_time_s: float) -> None:
        """Finalise the header with stats, colour-code, collapse revealer."""
        self._running = False
        mins = int(wall_time_s // 60)
        secs = int(wall_time_s % 60)
        time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
        self._stats_lbl.set_text(
            f"{outcome.upper()}  \u00b7  {iterations} iters  \u00b7  "
            f"${cost_usd:.4f}  \u00b7  {time_str}"
        )
        self._stats_lbl.set_visible(True)

        # Mark last iteration complete
        low = outcome.lower()
        if self._last_iter_n is not None and self._last_iter_n in self._iterations:
            last_err = "fail" in low or "error" in low
            self._iterations[self._last_iter_n].set_complete(last_err)

        # Snap progress bar to final state with outcome color
        self._progress_bar.set_fraction(1.0)
        self._progress_bar.set_visible(True)
        # Remove any previous color class
        for cls in ("task-progress-success", "task-progress-fail", "task-progress-cancel"):
            self._progress_bar.remove_css_class(cls)
        if "success" in low or "done" in low:
            self._progress_bar.add_css_class("task-progress-success")
            self._header.add_css_class("task-header-success")
        elif "fail" in low or "error" in low:
            self._progress_bar.add_css_class("task-progress-fail")
            self._header.add_css_class("task-header-fail")
        else:
            self._progress_bar.add_css_class("task-progress-cancel")
            self._header.add_css_class("task-header-cancel")

        # Collapse the revealer on completion
        self._revealer.set_reveal_child(False)
        self._toggle_icon.set_from_icon_name("pan-end-symbolic")

    # -- toggle ---------------------------------------------------------------

    def _toggle(self) -> None:
        revealed = self._revealer.get_reveal_child()
        self._revealer.set_reveal_child(not revealed)
        icon = "pan-down-symbolic" if not revealed else "pan-end-symbolic"
        self._toggle_icon.set_from_icon_name(icon)


class QuestionWidget(Gtk.Box):
    """In-stream clarifying-question bubble with option buttons.

    Appended to _msg_list.  After the user answers, buttons become
    insensitive and the widget stays as a historical record.
    """

    def __init__(self, question: str, options: list[str],
                 orchestrator, window):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add_css_class("msg-widget")
        self.add_css_class("msg-genie")
        self.set_halign(Gtk.Align.START)

        # Header
        sender_lbl = Gtk.Label(label="Genie")
        sender_lbl.add_css_class("msg-sender")
        sender_lbl.add_css_class("msg-sender-genie")
        sender_lbl.set_halign(Gtk.Align.START)
        self.append(sender_lbl)

        # Question text
        q_lbl = Gtk.Label(label=question)
        q_lbl.set_wrap(True)
        q_lbl.set_halign(Gtk.Align.START)
        q_lbl.set_margin_bottom(4)
        self.append(q_lbl)

        # Option buttons
        self._btn_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        for opt in options:
            btn = Gtk.Button(label=opt)
            btn.add_css_class("question-btn")
            btn.set_halign(Gtk.Align.FILL)

            def _on_option(_b, text=opt):
                orchestrator.answer_clarification(text)
                self._disable_buttons()
                GLib.idle_add(window.add_message, "You", text, "chat")
                GLib.idle_add(window._scroll_to_bottom)

            btn.connect("clicked", _on_option)
            self._btn_box.append(btn)

        # "Other\u2026" button
        other_btn = Gtk.Button(label="Other\u2026")
        other_btn.add_css_class("question-btn")
        other_btn.set_halign(Gtk.Align.FILL)
        other_btn.connect("clicked", lambda _b: self._show_entry())
        self._btn_box.append(other_btn)
        self.append(self._btn_box)

        # Free-text entry row (hidden initially)
        self._entry_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,
                                  spacing=4)
        self._entry_row.set_visible(False)
        self._entry = Gtk.Entry()
        self._entry.add_css_class("other-entry")
        self._entry.set_hexpand(True)
        self._entry_row.append(self._entry)

        submit_btn = Gtk.Button(label="Submit")
        submit_btn.add_css_class("submit-btn")

        def _submit(_w=None):
            text = self._entry.get_text().strip()
            if not text:
                return
            orchestrator.answer_clarification(text)
            self._disable_buttons()
            self._entry.set_sensitive(False)
            submit_btn.set_sensitive(False)
            GLib.idle_add(window.add_message, "You", text, "chat")
            GLib.idle_add(window._scroll_to_bottom)

        submit_btn.connect("clicked", _submit)
        self._entry.connect("activate", _submit)
        self._entry_row.append(submit_btn)
        self.append(self._entry_row)

    def _show_entry(self) -> None:
        self._btn_box.set_visible(False)
        self._entry_row.set_visible(True)
        self._entry.grab_focus()

    def _disable_buttons(self) -> None:
        child = self._btn_box.get_first_child()
        while child is not None:
            if isinstance(child, Gtk.Button):
                child.set_sensitive(False)
            child = child.get_next_sibling()


class PlanConfirmWidget(Gtk.Box):
    """In-stream plan confirmation card with Approve / Reject buttons.

    Appended to _msg_list.  After the user responds, buttons become
    insensitive and the widget stays as a historical record.
    """

    def __init__(self, plan: dict, orchestrator, window):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.add_css_class("msg-widget")
        self.add_css_class("plan-confirm-widget")
        self.set_halign(Gtk.Align.FILL)

        # Header
        sender_lbl = Gtk.Label(label="Genie \u2014 Plan")
        sender_lbl.add_css_class("msg-sender")
        sender_lbl.add_css_class("msg-sender-genie")
        sender_lbl.set_halign(Gtk.Align.START)
        self.append(sender_lbl)

        # Plan content
        def _add_label(text: str) -> None:
            lbl = Gtk.Label(label=text)
            lbl.set_halign(Gtk.Align.START)
            lbl.set_wrap(True)
            self.append(lbl)

        _add_label("PLAN: " + plan.get("goal", ""))
        _add_label("Estimated iterations: "
                   + str(plan.get("estimated_iterations", "?")))
        for i, step in enumerate(plan.get("steps", []), start=1):
            desc = (step if isinstance(step, str)
                    else step.get("description", str(step)))
            _add_label(f"  {i}. {desc}")
        for risk in plan.get("risks", []):
            _add_label("\u26a0 Risk: " + risk)

        # Button row
        btn_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        btn_row.set_margin_top(6)

        approve_btn = Gtk.Button(label="Approve")
        approve_btn.add_css_class("question-btn")

        reject_btn = Gtk.Button(label="Reject")
        reject_btn.add_css_class("question-btn")

        def _on_approve(_b):
            orchestrator.answer_plan_confirm(True)
            approve_btn.set_label("✓ Approved")
            approve_btn.remove_css_class("question-btn")
            approve_btn.add_css_class("plan-btn-chosen-approve")
            approve_btn.set_sensitive(False)
            reject_btn.remove_css_class("question-btn")
            reject_btn.add_css_class("plan-btn-faded")
            reject_btn.set_sensitive(False)

        def _on_reject(_b):
            orchestrator.answer_plan_confirm(False)
            reject_btn.set_label("✗ Rejected")
            reject_btn.remove_css_class("question-btn")
            reject_btn.add_css_class("plan-btn-chosen-reject")
            reject_btn.set_sensitive(False)
            approve_btn.remove_css_class("question-btn")
            approve_btn.add_css_class("plan-btn-faded")
            approve_btn.set_sensitive(False)

        approve_btn.connect("clicked", _on_approve)
        reject_btn.connect("clicked", _on_reject)
        btn_row.append(approve_btn)
        btn_row.append(reject_btn)
        self.append(btn_row)


class ResumeWidget(Gtk.Box):
    """In-stream resume/discard card shown after a cancelled task or on startup
    when a checkpoint from a previous session is found.

    Buttons stay in the stream as historical record after being clicked.
    """

    def __init__(self, checkpoint: dict, orchestrator, window):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.add_css_class("msg-widget")
        self.add_css_class("plan-confirm-widget")
        self.set_halign(Gtk.Align.FILL)

        self._orchestrator = orchestrator
        self._window = window
        self._checkpoint = checkpoint

        # Header
        sender_lbl = Gtk.Label(label="Genie \u2014 Paused Task")
        sender_lbl.add_css_class("msg-sender")
        sender_lbl.add_css_class("msg-sender-genie")
        sender_lbl.set_halign(Gtk.Align.START)
        self.append(sender_lbl)

        # Info
        iteration = checkpoint.get("iteration", "?")
        cost = checkpoint.get("cost_usd", 0.0)
        goal = checkpoint.get("goal", "")
        info_lbl = Gtk.Label(
            label=f"Task paused at iter {iteration} (${cost:.3f}). Resume?\n\nGoal: {goal}"
        )
        info_lbl.set_halign(Gtk.Align.START)
        info_lbl.set_wrap(True)
        self.append(info_lbl)

        # Button row
        btn_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        btn_row.set_margin_top(6)

        self._resume_btn = Gtk.Button(label="Resume")
        self._resume_btn.add_css_class("question-btn")

        self._discard_btn = Gtk.Button(label="Discard")
        self._discard_btn.add_css_class("question-btn")

        self._resume_btn.connect("clicked", self._on_resume)
        self._discard_btn.connect("clicked", self._on_discard)
        btn_row.append(self._resume_btn)
        btn_row.append(self._discard_btn)
        self.append(btn_row)

    def _on_resume(self, _b):
        # Mark buttons
        self._resume_btn.set_label("\u2713 Resuming")
        self._resume_btn.remove_css_class("question-btn")
        self._resume_btn.add_css_class("plan-btn-chosen-approve")
        self._resume_btn.set_sensitive(False)
        self._discard_btn.remove_css_class("question-btn")
        self._discard_btn.add_css_class("plan-btn-faded")
        self._discard_btn.set_sensitive(False)

        # Enter executing state
        self._window._enter_executing_state()
        self._window._active_task_widget = None
        self._window._task_start_time = time.time()

        checkpoint = self._checkpoint

        def _run_resumed():
            try:
                result: TaskResult = self._orchestrator.run_task(
                    checkpoint=checkpoint,
                    on_update=self._window._on_update_callback,
                )
                if self._window._active_task_widget is not None:
                    wall = time.time() - self._window._task_start_time if self._window._task_start_time else 0.0
                    GLib.idle_add(
                        self._window._active_task_widget.complete,
                        str(result.outcome),
                        result.iterations,
                        result.cost_usd,
                        wall,
                    )
                self._window._active_task_widget = None
                self._window._task_start_time = None

                # If cancelled again, show another ResumeWidget
                if str(result.outcome) == "cancelled":
                    new_cp = persistence.load_checkpoint()
                    if new_cp:
                        GLib.idle_add(self._window._inject_resume_widget, new_cp)
                    GLib.idle_add(self._window._reenable_input)
                else:
                    GLib.idle_add(self._window._reenable_input)
            except Exception as exc:
                log(f"Error in resumed run_task: {exc}")
                GLib.idle_add(
                    self._window.add_message, "Genie", f"Error: {exc}", "error"
                )
                self._window._active_task_widget = None
                self._window._task_start_time = None
                GLib.idle_add(self._window._reenable_input)

        t = threading.Thread(target=_run_resumed, daemon=True)
        self._window._task_thread = t
        t.start()

    def _on_discard(self, _b):
        # Mark buttons
        self._discard_btn.set_label("\u2717 Discarded")
        self._discard_btn.remove_css_class("question-btn")
        self._discard_btn.add_css_class("plan-btn-chosen-reject")
        self._discard_btn.set_sensitive(False)
        self._resume_btn.remove_css_class("question-btn")
        self._resume_btn.add_css_class("plan-btn-faded")
        self._resume_btn.set_sensitive(False)

        # Delete checkpoint
        try:
            os.remove(CHECKPOINT_PATH)
        except OSError:
            pass


class PlanDraftWidget(Gtk.Box):
    """In-stream structured subtask plan widget.

    Shows the current draft as a numbered subtask list with model tier
    and description.  ``update_draft()`` replaces the content in place
    (removes old rows and builds fresh ones) rather than appending a
    new widget.
    """

    def __init__(self, subtasks: list | None = None, draft_n: int = 1, min_subtasks: int = 0):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.add_css_class("msg-widget")
        self.add_css_class("plan-draft-widget")
        self.set_halign(Gtk.Align.FILL)

        self._draft_n = draft_n
        self._min_subtasks = min_subtasks

        # Header
        self._header = Gtk.Label(label=self._header_text(len(subtasks) if subtasks else 0))
        self._header.add_css_class("plan-draft-header")
        self._header.set_halign(Gtk.Align.START)
        self.append(self._header)

        # Container for subtask rows (replaced on each update)
        self._rows_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
        self.append(self._rows_box)

        if subtasks:
            self._render_subtasks(subtasks)

    def _header_text(self, n_subtasks: int) -> str:
        """Build header label text, showing count and warning if below suggested minimum."""
        base = f"Plan \u2014 Draft {self._draft_n}"
        if n_subtasks == 0:
            return base
        if self._min_subtasks > 0 and n_subtasks < self._min_subtasks:
            return f"{base}  ({n_subtasks} subtasks \u26a0 suggested \u2265 {self._min_subtasks})"
        return f"{base}  ({n_subtasks} subtasks)"

    def update_draft(self, subtasks: list, draft_n: int | None = None) -> None:
        """Replace the displayed subtask list with a new draft."""
        if draft_n is not None:
            self._draft_n = draft_n
        else:
            self._draft_n += 1
        self._header.set_text(self._header_text(len(subtasks)))

        # Remove old rows
        child = self._rows_box.get_first_child()
        while child is not None:
            next_child = child.get_next_sibling()
            self._rows_box.remove(child)
            child = next_child

        self._render_subtasks(subtasks)

    def _render_subtasks(self, subtasks: list) -> None:
        for s in subtasks:
            n = s.n if hasattr(s, "n") else s.get("n", "?")
            desc = s.description if hasattr(s, "description") else s.get("description", "")
            tier = s.model_tier if hasattr(s, "model_tier") else s.get("model_tier", "tier_0")
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            num_lbl = Gtk.Label(label=f"{n}.")
            num_lbl.add_css_class("plan-draft-row")
            num_lbl.set_halign(Gtk.Align.START)
            num_lbl.set_valign(Gtk.Align.START)
            row.append(num_lbl)

            desc_lbl = Gtk.Label(label=desc)
            desc_lbl.add_css_class("plan-draft-row")
            desc_lbl.set_halign(Gtk.Align.START)
            desc_lbl.set_wrap(True)
            desc_lbl.set_hexpand(True)
            row.append(desc_lbl)

            tier_lbl = Gtk.Label(label=f"[{tier}]")
            tier_lbl.add_css_class("plan-draft-tier")
            tier_lbl.set_halign(Gtk.Align.END)
            tier_lbl.set_valign(Gtk.Align.START)
            row.append(tier_lbl)

            self._rows_box.append(row)
