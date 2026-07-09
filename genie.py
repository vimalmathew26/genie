#!/usr/bin/env python3
"""
Genie - AI Desktop Automation Assistant
Thin GTK4 UI shell — all LLM, planning, and action execution logic
lives in orchestrator.py.

Architecture:
    MessageWidget          - Per-message chat bubble widget
    GenieWindow            - GTK4 chat UI (scrolled message list + entry box)
    GenieApplication       - GTK Application wrapper
    GenieOrchestrator      - (imported) ReAct brain loop & action dispatch
"""

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib, Gdk, Gio, Pango

import datetime
import json
import os
import sys
import threading
import time
import uuid

from xdotool_controller import XdotoolController
from orchestrator import GenieOrchestrator, TaskResult
import config as config
from config import (
    CHECKPOINT_PATH,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    log,
)
import persistence
from chat_logger import chat_logger as _chat_logger
from widgets import (
    MessageWidget, _format_action_status,
    IterationWidget, CollapsibleTaskWidget,
    QuestionWidget, PlanConfirmWidget,
    ResumeWidget, PlanDraftWidget,
)


# =============================================================================
# Trivial goal detection — skip plan phase for simple single-pass tasks
# =============================================================================

_MULTI_STEP_MARKERS = (
    " then ", " after ", " next ", " also ", " and then ", " followed by ",
    " once ", " finally ", " subsequently ", " afterwards ",
    # Navigation / interaction verbs that imply multi-step execution:
    " click ", "click on", " open the ", " show me", " navigate", " go to ",
    " scroll", " select ", " choose ",
)

def _is_trivial_goal(goal: str) -> bool:
    """Return True when the goal is simple enough to skip the plan LLM call.

    Heuristics:
    - No multi-step language (then, after, followed by, …)
    - Word count ≤ 20
    These are sufficient for typical one-action-chain goals like
    "open chrome and search X" or "open gedit".
    """
    g = goal.lower()
    if any(m in g for m in _MULTI_STEP_MARKERS):
        return False
    return len(goal.split()) <= 6


# =============================================================================
# GTK4 User Interface
# =============================================================================

class GenieWindow(Gtk.Window):
    """Main chat window with a scrolled message list and an input entry."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ----- Dark theme via CssProvider ------------------------------------
        # Request dark theme variant from GTK so Adwaita base is dark
        settings = Gtk.Settings.get_default()
        if settings is not None:
            settings.set_property("gtk-application-prefer-dark-theme", True)

        css = """
        /* ================================================================
           Genie — dense dark professional theme
           ================================================================ */

        /* ---- global dark surface ---- */
        window, window.background {
            background-color: #0a0a0a;
            color: #e0eaff;
        }
        box, scrolledwindow, viewport {
            background-color: transparent;
            color: #e0eaff;
        }

        /* ---- message list ---- */
        .msg-list {
            background-color: #0a0a0a;
            padding-top: 8px;
        }
        .msg-list > row {
            background-color: transparent;
            border: none;
            padding: 1px 0;
            outline: none;
        }
        .msg-list > row:focus {
            outline: none;
            box-shadow: none;
        }
        .msg-list > row:selected {
            background-color: transparent;
        }

        /* ---- message widget base ---- */
        .msg-widget {
            padding: 8px 12px;
            border-radius: 8px;
            margin: 2px 0;
        }
        .msg-user {
            background-color: #0f1e38;
            border: 1px solid #2255aa;
            margin-left: 40px;
        }
        .msg-genie {
            background-color: #141414;
            border: 1px solid #242424;
            margin-right: 20px;
        }
        .msg-action {
            background-color: transparent;
            margin-right: 0;
            padding: 4px 12px;
        }
        .msg-error {
            background-color: #160c0c;
            border-left: 3px solid #cc3333;
            margin-right: 20px;
        }
        .msg-summary {
            background-color: #111111;
            border: 1px solid #242424;
            border-radius: 8px;
        }

        /* ---- message header ---- */
        .msg-header {
            margin-bottom: 2px;
        }
        .msg-sender {
            font-weight: 700;
            font-size: 11px;
        }
        .msg-sender-user {
            color: #4da6ff;
        }
        .msg-sender-genie {
            color: #00d49a;
        }
        .msg-timestamp {
            color: #4d6080;
            font-size: 10px;
        }
        .msg-copy-btn {
            background: none;
            border: none;
            box-shadow: none;
            color: #4d6080;
            padding: 0;
            min-height: 18px;
            min-width: 18px;
        }
        .msg-copy-btn:hover {
            color: #6080b0;
        }

        /* ---- message body textview ---- */
        .msg-body,
        .msg-body text {
            background-color: transparent;
            color: #e0eaff;
            font-family: system-ui, -apple-system, sans-serif;
            font-size: 13px;
        }
        .msg-body-action,
        .msg-body-action text {
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 11px;
            color: #6080b0;
        }
        .msg-body-error,
        .msg-body-error text {
            color: #ff6b6b;
        }
        .msg-body-summary,
        .msg-body-summary text {
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 11px;
            color: #6080b0;
        }

        /* ---- outcome badges ---- */
        .msg-badge {
            font-size: 10px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 3px;
        }
        .badge-success {
            background-color: #0d2a1e;
            color: #00d49a;
        }
        .badge-fail {
            background-color: #2a0d0d;
            color: #ff6b6b;
        }
        .badge-done {
            background-color: #141428;
            color: #8888ff;
        }

        /* ---- input area ---- */
        #input-scroll {
            border: 1px solid #242424;
            border-radius: 10px;
            background-color: #0f0f0f;
        }
        #input-scroll:focus {
            border-color: #4da6ff;
        }
        #input-view {
            background-color: transparent;
            color: #e0eaff;
            font-family: system-ui, -apple-system, sans-serif;
            font-size: 13px;
            padding: 8px;
        }
        #input-view text {
            background-color: transparent;
            color: #e0eaff;
        }
        #input-scroll undershoot.top,
        #input-scroll undershoot.bottom,
        #input-scroll overshoot.top,
        #input-scroll overshoot.bottom {
            background: none;
        }

        /* ---- send button ---- */
        .send-btn {
            background-color: #0d2218;
            color: #00d49a;
            border: none;
            border-radius: 6px;
            box-shadow: none;
            padding: 0;
            min-height: 26px;
            min-width: 26px;
            margin: 4px;
        }
        .send-btn:hover {
            background-color: #142a20;
        }

        /* ---- cancel button (in status bar) ---- */
        #cancel-btn {
            background-color: #160c0c;
            color: #ff6b6b;
            border: 1px solid #330d0d;
            border-radius: 5px;
            padding: 2px 10px;
            font-size: 11px;
            font-weight: 600;
        }
        #cancel-btn:hover {
            background-color: #200d0d;
        }
        #cancel-btn:disabled {
            opacity: 0.3;
        }

        /* ---- status bar ---- */
        .status-label {
            color: #4d6080;
            font-size: 11px;
        }

        /* ---- scrollbar ---- */
        scrollbar slider {
            background-color: #242424;
            border-radius: 9999px;
            min-width: 5px;
            min-height: 5px;
        }
        scrollbar trough {
            background-color: transparent;
        }

        /* ---- labels ---- */
        label {
            color: #e0eaff;
        }

        /* ---- clarifying question buttons ---- */
        .question-btn {
            background-color: #0d2218;
            color: #00d49a;
            border: 1px solid #0d2218;
            border-radius: 6px;
            padding: 6px 14px;
            margin: 2px 0;
        }
        .question-btn:hover { background-color: #142a20; }

        /* ---- plan confirm chosen / faded states ---- */
        .plan-btn-chosen-approve {
            background-color: #0d3020;
            color: #00d49a;
            border: 1px solid #00d49a44;
            border-radius: 6px;
            padding: 6px 14px;
            margin: 2px 0;
            font-weight: bold;
        }
        .plan-btn-chosen-reject {
            background-color: #2a0d0d;
            color: #ff6b6b;
            border: 1px solid #ff6b6b44;
            border-radius: 6px;
            padding: 6px 14px;
            margin: 2px 0;
            font-weight: bold;
        }
        .plan-btn-faded {
            background-color: #0a0a0a;
            color: #4d6080;
            border: 1px solid #242424;
            border-radius: 6px;
            padding: 6px 14px;
            margin: 2px 0;
            opacity: 0.35;
        }

        /* ---- Other… free-text entry ---- */
        .other-entry {
            background-color: #0f0f0f;
            color: #e0eaff;
            border: 1px solid #2255aa;
            border-radius: 6px;
            padding: 6px;
            font-family: system-ui, -apple-system, sans-serif;
            font-size: 13px;
        }

        /* ---- submit button for Other… entry ---- */
        .submit-btn {
            background-color: #0d2218;
            color: #00d49a;
            border: 1px solid #2255aa;
            border-radius: 6px;
            padding: 6px 14px;
            margin-top: 4px;
        }
        .submit-btn:hover { background-color: #142a20; }

        /* ---- context menu popover ---- */
        .msg-context-popover {
            background-color: #0a0a0a;
            border: 1px solid #242424;
        }
        .msg-context-popover > contents {
            background-color: #0a0a0a;
        }
        .context-menu-item {
            color: #e0eaff;
            font-size: 12px;
            padding: 6px 16px;
            background: none;
            border: none;
            box-shadow: none;
        }
        .context-menu-item:hover {
            background-color: #0d2218;
            color: #00d49a;
        }

        /* ---- header hover feedback ---- */
        .task-header:hover {
            background-color: #0d2218;
            border-radius: 4px;
        }
        .iter-header:hover {
            background-color: #0d2218;
            border-radius: 4px;
        }

        /* ---- collapsible task widget ---- */
        .task-widget {
            background-color: #111111;
            border: 1px solid #242424;
            border-radius: 8px;
            padding: 0;
            margin: 4px 0;
        }
        .task-header {
            padding: 8px 12px;
        }
        .task-header-success { border-left: 3px solid #00d49a; }
        .task-header-fail    { border-left: 3px solid #ff6b6b; }
        .task-header-cancel  { border-left: 3px solid #8888ff; }
        .task-goal {
            font-weight: 700;
            font-size: 12px;
            color: #e0eaff;
        }
        .task-stats {
            font-size: 10px;
            color: #4d6080;
            font-family: "JetBrains Mono", "Fira Code", monospace;
        }
        .task-toggle, .iter-toggle {
            background: none;
            border: none;
            box-shadow: none;
            color: #4d6080;
            padding: 0;
            min-height: 20px;
            min-width: 20px;
        }
        .task-toggle:hover, .iter-toggle:hover {
            color: #888888;
        }

        /* ---- subtask summary row (completion message) ---- */
        .subtask-summary {
            font-size: 11px;
            font-style: italic;
            color: #00d49a;
            padding: 2px 8px 6px 14px;
            margin: 0 8px 0 8px;
        }

        /* ---- subtask divider header ---- */
        .subtask-header {
            font-size: 11px;
            font-weight: 700;
            color: #00d49a;
            background-color: #0a0a0a;
            border-left: 3px solid #00d49a;
            border-radius: 3px;
            padding: 3px 8px;
            margin: 6px 8px 2px 8px;
        }

        /* ---- iteration widget ---- */
        .iter-widget {
            padding: 0;
            margin: 0 8px 4px 8px;
        }
        .iter-header {
            padding: 4px 8px;
            background-color: #141414;
            border-radius: 4px;
        }
        .iter-number {
            font-weight: 700;
            font-size: 11px;
            color: #4da6ff;
        }
        .iter-summary {
            font-size: 10px;
            color: #4d6080;
            font-family: "JetBrains Mono", "Fira Code", monospace;
        }
        .iter-detail-row {
            padding: 2px 12px;
        }
        .iter-detail-text,
        .iter-detail-text text {
            background-color: transparent;
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 11px;
            color: #6080b0;
        }

        /* ---- header bar ---- */
        headerbar {
            background-color: #0a0a0a;
            border-bottom: 1px solid #161616;
            min-height: 36px;
            padding: 0 6px;
        }
        .hdr-btn {
            background-color: #141414;
            color: #8898b8;
            border: 1px solid #242424;
            border-radius: 5px;
            padding: 2px 10px;
            font-size: 11px;
            min-height: 24px;
        }
        .hdr-btn:hover {
            background-color: #242424;
            color: #e0eaff;
        }
        .session-title-lbl {
            font-weight: 700;
            font-size: 13px;
            color: #e0eaff;
        }

        /* ---- sessions popover ---- */
        .sessions-popover {
            background-color: #0a0a0a;
            border: 1px solid #242424;
        }
        .sessions-popover > contents {
            background-color: #0a0a0a;
        }
        .session-list {
            background-color: #0a0a0a;
        }
        .session-list > row {
            background-color: transparent;
            padding: 6px 10px;
            border-bottom: 1px solid #141414;
            outline: none;
        }
        .session-list > row:hover {
            background-color: #141414;
        }
        .session-list > row:selected {
            background-color: transparent;
        }
        .session-list > row.session-active {
            background-color: #0d2218;
            border-left: 2px solid #00d49a;
        }
        .session-date {
            font-size: 10px;
            color: #4d6080;
        }
        .session-preview {
            font-size: 12px;
            color: #8898b8;
        }

        /* ---- activity bar ---- */
        .activity-bar {
            background-color: #141414;
            border: 1px solid #242424;
            border-radius: 8px;
            padding: 6px 12px;
            margin: 4px 0;
        }
        .activity-spinner {
            color: #00d49a;
            min-width: 16px;
            min-height: 16px;
        }
        .activity-spinner-thinking {
            color: #4da6ff;
        }
        .activity-label {
            color: #e0eaff;
            font-size: 13px;
            font-weight: 600;
        }

        /* ---- activity pills ---- */
        .activity-pill {
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 10px;
            border-radius: 10px;
            padding: 2px 8px;
        }
        .progress-pill {
            background-color: #1a1a1a;
            color: #4da6ff;
            border: 1px solid #242424;
        }
        .cost-pill {
            background-color: #1a1a1a;
            color: #4d6080;
            border: 1px solid #242424;
        }
        .cost-pill-warn {
            background-color: #ffaa00;
            color: #0a0a0a;
            border: 1px solid #ffaa00;
        }
        .cost-pill-danger {
            background-color: #ff6b6b;
            color: #0a0a0a;
            border: 1px solid #ff6b6b;
        }

        /* ---- task progress bar ---- */
        .task-progress-bar trough {
            min-height: 3px;
            border-radius: 2px;
            background-color: #242424;
        }
        .task-progress-bar progress {
            min-height: 3px;
            border-radius: 2px;
            background-color: #4da6ff;
        }
        .task-progress-success progress { background-color: #00d49a; }
        .task-progress-fail progress    { background-color: #ff6b6b; }
        .task-progress-cancel progress  { background-color: #8888ff; }

        /* ---- iteration phase icon ---- */
        .iter-icon {
            font-size: 11px;
            font-weight: bold;
            min-width: 16px;
        }
        .iter-icon-running { color: #4da6ff; }
        .iter-icon-ok      { color: #00d49a; }
        .iter-icon-error   { color: #ff6b6b; }

        /* ---- toolbar toggles ---- */
        .toolbar-row {
            margin: 2px 0;
        }
        .toolbar-toggle {
            background-color: #141414;
            color: #4d6080;
            border: 1px solid #242424;
            border-radius: 5px;
            padding: 2px 10px;
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 11px;
            min-height: 24px;
        }
        .toolbar-toggle:hover {
            background-color: #242424;
            color: #888888;
        }
        .toolbar-toggle:checked {
            background-color: #0d2218;
            color: #00d49a;
            border-color: #00d49a;
        }

        /* ---- input busy state ---- */
        .input-busy {
            border-color: #4da6ff;
            opacity: 0.6;
        }

        /* ---- plan confirm widget ---- */
        .plan-confirm-widget {
            background-color: #111111;
            border: 1px solid #242424;
            border-radius: 8px;
            padding: 10px 14px;
        }

        /* ---- plan draft widget ---- */
        .plan-draft-widget {
            background-color: #0d1117;
            border: 1px solid #1a3a2a;
            border-radius: 8px;
            padding: 10px 14px;
        }
        .plan-draft-header {
            color: #00d49a;
            font-weight: bold;
            margin-bottom: 4px;
        }
        .plan-draft-row {
            color: #c0c8d8;
            font-family: monospace;
            font-size: 0.9em;
        }
        .plan-draft-tier {
            color: #666;
            font-size: 0.8em;
        }
        .plan-start-btn {
            background-color: #0d2218;
            color: #00d49a;
            border: 1px solid #00d49a;
            border-radius: 4px;
            padding: 4px 16px;
            font-weight: bold;
        }
        .plan-start-btn:hover {
            background-color: #143a28;
        }
        """
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(css.encode("utf-8"))
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        # Window properties
        self.set_default_size(WINDOW_WIDTH, WINDOW_HEIGHT)

        # ----- Header bar with session controls ------------------------------
        headerbar = Gtk.HeaderBar()
        headerbar.set_show_title_buttons(True)

        new_chat_btn = Gtk.Button(label="\uff0b New")
        new_chat_btn.add_css_class("hdr-btn")
        new_chat_btn.set_tooltip_text("New Chat")
        new_chat_btn.connect("clicked", self._on_new_chat)
        headerbar.pack_start(new_chat_btn)

        self._session_title_lbl = Gtk.Label(label="New session")
        self._session_title_lbl.add_css_class("session-title-lbl")
        self._session_title_lbl.set_ellipsize(Pango.EllipsizeMode.END)
        self._session_title_lbl.set_max_width_chars(40)
        headerbar.set_title_widget(self._session_title_lbl)

        sessions_btn = Gtk.Button.new_from_icon_name("view-list-symbolic")
        sessions_btn.add_css_class("hdr-btn")
        sessions_btn.set_tooltip_text("Sessions")
        sessions_btn.connect("clicked", self._on_open_sessions)
        headerbar.pack_end(sessions_btn)

        # Sessions popover
        self._sessions_popover = Gtk.Popover()
        self._sessions_popover.set_parent(sessions_btn)
        self._sessions_popover.add_css_class("sessions-popover")
        self._sessions_popover.set_size_request(300, -1)

        pop_scroll = Gtk.ScrolledWindow()
        pop_scroll.set_min_content_height(100)
        pop_scroll.set_max_content_height(400)
        pop_scroll.set_propagate_natural_height(True)
        pop_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._session_listbox = Gtk.ListBox()
        self._session_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self._session_listbox.add_css_class("session-list")
        self._session_listbox.connect("row-activated", self._on_session_row_activated)
        pop_scroll.set_child(self._session_listbox)
        self._sessions_popover.set_child(pop_scroll)

        self.set_titlebar(headerbar)
        self._session_title_set = False

        # ----- Initialize backend components ---------------------------------
        controller = XdotoolController()
        self.orchestrator = GenieOrchestrator(controller)

        # ----- Build UI layout -----------------------------------------------
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        vbox.set_margin_top(8)
        vbox.set_margin_bottom(8)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)
        self.set_child(vbox)

        # ---- Scrolled message list (replaces old flat TextView) ----
        self._scrolled = Gtk.ScrolledWindow()
        self._scrolled.set_vexpand(True)
        self._scrolled.set_hexpand(True)
        self._scrolled.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )

        self._msg_list = Gtk.ListBox()
        self._msg_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._msg_list.add_css_class("msg-list")
        self._scrolled.set_child(self._msg_list)
        vbox.append(self._scrolled)

        adj = self._scrolled.get_vadjustment()
        self._msg_list_was_at_bottom = True
        adj.connect("value-changed", self._on_adj_value_changed)
        adj.connect("notify::upper", self._on_adj_upper_changed)

        # ----- Activity bar (replaces old status row) -------------------------
        self._activity_revealer = Gtk.Revealer()
        self._activity_revealer.set_transition_type(
            Gtk.RevealerTransitionType.SLIDE_DOWN
        )
        self._activity_revealer.set_transition_duration(250)
        self._activity_revealer.set_reveal_child(False)

        activity_bar = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=8
        )
        activity_bar.add_css_class("activity-bar")

        self.status_spinner = Gtk.Spinner()
        self.status_spinner.add_css_class("activity-spinner")
        self.status_spinner.set_visible(True)
        activity_bar.append(self.status_spinner)

        self.status_label = Gtk.Label(label="")
        self.status_label.set_halign(Gtk.Align.START)
        self.status_label.set_hexpand(True)
        self.status_label.add_css_class("activity-label")
        activity_bar.append(self.status_label)

        # Progress pill — "iter 7 / ~20"
        self._progress_pill = Gtk.Label(label="")
        self._progress_pill.add_css_class("activity-pill")
        self._progress_pill.add_css_class("progress-pill")
        self._progress_pill.set_visible(False)
        activity_bar.append(self._progress_pill)

        # Cost pill — "$0.003 / $0.50"
        self._cost_pill = Gtk.Label(label="")
        self._cost_pill.add_css_class("activity-pill")
        self._cost_pill.add_css_class("cost-pill")
        self._cost_pill.set_visible(False)
        activity_bar.append(self._cost_pill)

        self.cancel_button = Gtk.Button(label="Cancel")
        self.cancel_button.set_name("cancel-btn")
        self.cancel_button.connect("clicked", self.on_cancel)
        self.cancel_button.set_sensitive(False)
        activity_bar.append(self.cancel_button)

        self._start_button = Gtk.Button(label="\u25b6 Start")
        self._start_button.add_css_class("plan-start-btn")
        self._start_button.connect("clicked", self._on_plan_start_clicked)
        self._start_button.set_visible(False)
        activity_bar.append(self._start_button)

        self._activity_revealer.set_child(activity_bar)
        vbox.append(self._activity_revealer)

        # ---- Toolbar row (toggle switches for task options) ----
        toolbar_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar_row.add_css_class("toolbar-row")
        toolbar_row.set_margin_top(2)
        toolbar_row.set_margin_bottom(2)

        self._decompose_toggle = Gtk.ToggleButton(label="\u26a1 Subtasks")
        self._decompose_toggle.add_css_class("toolbar-toggle")
        self._decompose_toggle.set_active(True)
        self._decompose_toggle.set_tooltip_text(
            "Decompose goal into subtasks via GoalTracker"
        )
        toolbar_row.append(self._decompose_toggle)

        self._clarify_toggle = Gtk.ToggleButton(label="? Clarify")
        self._clarify_toggle.add_css_class("toolbar-toggle")
        self._clarify_toggle.set_active(True)
        self._clarify_toggle.set_tooltip_text(
            "Ask clarifying questions before executing"
        )
        toolbar_row.append(self._clarify_toggle)

        vbox.append(toolbar_row)

        # ---- Input area with embedded send button ----
        input_overlay = Gtk.Overlay()

        self._input_scroll = Gtk.ScrolledWindow()
        self._input_scroll.set_min_content_height(36)
        self._input_scroll.set_max_content_height(72)
        self._input_scroll.set_propagate_natural_height(True)
        self._input_scroll.set_name("input-scroll")

        self.input_view = Gtk.TextView()
        self.input_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self.input_view.set_name("input-view")
        self.input_view.set_top_margin(8)
        self.input_view.set_bottom_margin(8)
        self.input_view.set_left_margin(10)
        self.input_view.set_right_margin(36)  # room for send button
        self._input_scroll.set_child(self.input_view)
        input_overlay.set_child(self._input_scroll)

        send_btn = Gtk.Button.new_from_icon_name("go-up-symbolic")
        send_btn.add_css_class("send-btn")
        send_btn.set_tooltip_text("Send")
        send_btn.set_halign(Gtk.Align.END)
        send_btn.set_valign(Gtk.Align.END)
        send_btn.set_margin_end(6)
        send_btn.set_margin_bottom(6)
        send_btn.connect("clicked", self.on_send)
        input_overlay.add_overlay(send_btn)

        vbox.append(input_overlay)

        # ----- Input placeholder hint ----------------------------------------
        self._hint_text = "Type a task\u2026"
        self._showing_hint = True
        input_buf = self.input_view.get_buffer()
        hint_tag = Gtk.TextTag(name="hint_text")
        hint_tag.set_property("foreground", "#333333")
        input_buf.get_tag_table().add(hint_tag)
        input_buf.set_text(self._hint_text)
        input_buf.apply_tag_by_name(
            "hint_text",
            input_buf.get_start_iter(),
            input_buf.get_end_iter(),
        )

        focus_ctl = Gtk.EventControllerFocus()
        focus_ctl.connect("enter", self._on_input_focus_in)
        focus_ctl.connect("leave", self._on_input_focus_out)
        self.input_view.add_controller(focus_ctl)

        # Key press handler for Shift+Enter / Enter
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self.on_key_pressed)
        self.input_view.add_controller(key_controller)

        # ---- Active task tracking -------------------------------------------
        self._active_task_widget: CollapsibleTaskWidget | None = None
        self._active_task_id: str | None = None
        self._task_start_time: float | None = None

        # ---- Visibility overhaul state --------------------------------------
        self._current_estimated_iters: int | None = None
        self._current_task_budget: float | None = None
        self._last_iter_had_error: bool = False

        # ---- UI state: "idle" | "planning" | "executing" -------------------
        self._ui_state: str = "idle"
        self._planning_session = None      # planner.PlanningSession
        self._planning_thread = None       # Thread running planning LLM calls
        self._plan_draft_widget: PlanDraftWidget | None = None
        self._plan_draft_n: int = 0  # auto-incremented by _show_plan_draft
        self._planning_goal: str | None = None  # goal being planned
        self._task_thread: threading.Thread | None = None  # current task daemon thread

        # Load active session or show welcome message
        has_history = self._replay_session(_chat_logger.active_filename)
        if not has_history:
            self.add_message(
                "Genie",
                "Hello! I'm Genie, your desktop assistant.\n"
                "Try: \"Open firefox and search for cats\""
            )
        GLib.idle_add(self.input_view.grab_focus)

        # ---- Check for interrupted checkpoint (crash/shutdown resume) ----
        _startup_cp = persistence.load_checkpoint()
        if _startup_cp:
            self._inject_resume_widget(_startup_cp)

    # ----- Chat helpers ------------------------------------------------------

    def add_message(self, sender: str, text: str, msg_type: str = "chat",
                    *, _scroll: bool = True, _log: bool = True) -> None:
        """Append a MessageWidget to the chat list and scroll to bottom."""
        widget = MessageWidget(sender, text, msg_type, window=self)
        self._msg_list.append(widget)
        if _log:
            _chat_logger.log({
                "type": "msg", "ts": time.time(),
                "sender": sender, "text": text, "msg_type": msg_type,
            })

    def _on_adj_value_changed(self, adj) -> None:
        """Track whether the user is scrolled to the bottom."""
        self._msg_list_was_at_bottom = (adj.get_value() + adj.get_page_size() + 10) >= adj.get_upper()

    def _on_adj_upper_changed(self, adj, _pspec) -> None:
        """Auto-scroll when content grows and user was at the bottom."""
        if getattr(self, '_msg_list_was_at_bottom', True):
            adj.set_value(adj.get_upper())

    def _scroll_to_bottom(self) -> None:
        """Scroll the message list to the very end."""
        adj = self._scrolled.get_vadjustment()
        adj.set_value(adj.get_upper())
        return False

    # ----- Session management ------------------------------------------------

    def _replay_session(self, filename: str) -> bool:
        """Replay a session file into the widget list.

        Returns True if at least one record was replayed.
        All widgets are built without per-record scroll; a single
        scroll-to-bottom fires after the full replay.
        Also sets the session title from the first user message.
        """
        records = _chat_logger.read_session(filename, 200)
        if not records:
            return False

        task_widgets: dict[str, CollapsibleTaskWidget] = {}

        for rec in records:
            try:
                rtype = rec.get("type")
                if rtype == "msg":
                    self.add_message(
                        rec["sender"], rec["text"],
                        rec.get("msg_type", "chat"),
                        _scroll=False, _log=False,
                    )
                    if (not self._session_title_set
                            and rec.get("sender") == "You"):
                        text = rec.get("text", "")
                        clean = " ".join(text.split())
                        trunc = (clean[:40] + "\u2026") if len(clean) > 40 else clean
                        self._session_title_lbl.set_text(trunc)
                        self._session_title_set = True
                elif rtype == "task_start":
                    tw = CollapsibleTaskWidget(rec["goal"])
                    self._msg_list.append(tw)
                    task_widgets[rec["task_id"]] = tw
                elif rtype == "action":
                    tw = task_widgets.get(rec.get("task_id"))
                    if tw is not None:
                        tw.add_action(
                            rec["iter_n"], rec["action_name"],
                            rec["result_str"],
                        )
                elif rtype == "task_complete":
                    tw = task_widgets.get(rec.get("task_id"))
                    if tw is not None:
                        tw.complete(
                            rec["outcome"], rec["iterations"],
                            rec["cost_usd"], rec["wall_time_s"],
                        )
            except (KeyError, TypeError, ValueError):
                continue  # skip malformed records silently

        self._scroll_to_bottom()
        return True

    def _clear_msg_list(self) -> None:
        """Remove all widgets from the message list."""
        while True:
            row = self._msg_list.get_row_at_index(0)
            if row is None:
                break
            self._msg_list.remove(row)

    def _on_new_chat(self, _btn=None) -> None:
        """Create a new session, clear the message list, show welcome."""
        _chat_logger.new_session()
        self._clear_msg_list()
        self._active_task_widget = None
        self._active_task_id = None
        self._task_start_time = None
        self._session_title_set = False
        self._session_title_lbl.set_text("New session")
        self.add_message(
            "Genie",
            "Hello! I'm Genie, your desktop assistant.\n"
            "Try: \"Open firefox and search for cats\""
        )
        self.input_view.grab_focus()

    def _switch_to_session(self, filename: str) -> None:
        """Switch to an existing session and replay it."""
        _chat_logger.switch_session(filename)
        self._clear_msg_list()
        self._active_task_widget = None
        self._active_task_id = None
        self._task_start_time = None
        self._session_title_set = False
        self._session_title_lbl.set_text("New session")
        has_history = self._replay_session(filename)
        if not has_history:
            self.add_message(
                "Genie",
                "Hello! I'm Genie, your desktop assistant.\n"
                "Try: \"Open firefox and search for cats\""
            )
        self.input_view.grab_focus()

    def _on_open_sessions(self, _btn) -> None:
        """Toggle the sessions popover, repopulating rows each time."""
        if self._sessions_popover.is_visible():
            self._sessions_popover.popdown()
            return
        self._populate_sessions_popover()
        self._sessions_popover.popup()

    def _populate_sessions_popover(self) -> None:
        """Rebuild the session list inside the popover."""
        while True:
            row = self._session_listbox.get_row_at_index(0)
            if row is None:
                break
            self._session_listbox.remove(row)

        active = _chat_logger.active_filename
        for filename in _chat_logger.list_sessions(50):
            date_str, preview = _chat_logger.session_meta(filename)

            row_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            row_box.set_margin_top(2)
            row_box.set_margin_bottom(2)

            date_lbl = Gtk.Label(label=date_str or "Unknown")
            date_lbl.add_css_class("session-date")
            date_lbl.set_halign(Gtk.Align.START)
            row_box.append(date_lbl)

            prev_lbl = Gtk.Label(label=preview)
            prev_lbl.add_css_class("session-preview")
            prev_lbl.set_halign(Gtk.Align.START)
            prev_lbl.set_ellipsize(Pango.EllipsizeMode.END)
            row_box.append(prev_lbl)

            row = Gtk.ListBoxRow()
            row.set_child(row_box)
            row._session_filename = filename
            if filename == active:
                row.add_css_class("session-active")
            self._session_listbox.append(row)

    def _on_session_row_activated(self, _listbox, row) -> None:
        """Handle click on a session row in the popover."""
        filename = getattr(row, "_session_filename", None)
        if filename:
            self._sessions_popover.popdown()
            self._switch_to_session(filename)

    # ----- Event handlers ----------------------------------------------------

    def on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events on the input TextView.
        
        - Plain Enter: Send message
        - Shift+Enter: Insert newline
        """
        if keyval == Gdk.KEY_Return or keyval == Gdk.KEY_KP_Enter:
            # Check if Shift is held
            if state & Gdk.ModifierType.SHIFT_MASK:
                # Shift+Enter: allow newline to be inserted (return False)
                return False
            else:
                # Plain Enter: send message (return True to block default)
                self.on_send()
                return True
        return False

    def on_send(self, _entry: Gtk.Entry = None) -> None:
        """Called when the user presses Enter (or sends a message).

        Routes based on UI state:
        - idle: start task (or enter planning if Subtasks toggle active)
        - planning: send refinement to planning session
        - executing: ignored (input is disabled)
        """
        # Auto-discard stale checkpoint from previous cancelled tasks
        try:
            os.remove(CHECKPOINT_PATH)
        except OSError:
            pass

        buffer = self.input_view.get_buffer()
        start = buffer.get_start_iter()
        end = buffer.get_end_iter()
        text = buffer.get_text(start, end, False).strip()

        if not text or text == self._hint_text:
            return

        self._showing_hint = False
        buffer.set_text("")
        self.add_message("You", text)

        # Update session title on first user message
        if not self._session_title_set:
            # Collapse to single line; strip markdown noise for a clean title
            clean = " ".join(text.split())
            trunc = (clean[:40] + "\u2026") if len(clean) > 40 else clean
            self._session_title_lbl.set_text(trunc)
            self._session_title_set = True

        # -- Planning state: route to refinement ----------------------------
        if self._ui_state == "planning":
            self._send_planning_refinement(text)
            return

        # -- Idle state: decide whether to enter planning or execute --------
        _use_goaltracker = self._decompose_toggle.get_active()
        config.CLARIFY_ENABLED = self._clarify_toggle.get_active()

        if _use_goaltracker:
            # Enter planning state — interactive multi-turn planning
            self._start_planning_session(text)
            return

        # -- Direct execution (no Subtasks toggle) --------------------------
        self._active_task_widget = None
        self._pending_task_goal = text
        self._task_start_time = None

        task_id = getattr(self.orchestrator, '_task_id', None) or str(uuid.uuid4())
        self._active_task_id = task_id
        _chat_logger.log({
            "type": "task_start", "ts": time.time(),
            "task_id": task_id, "goal": text,
        })

        self._enter_executing_state()

        def _run_task():
            try:
                result: TaskResult = self.orchestrator.run_task(
                    goal=text,
                    mode="interactive",
                    task_type="auto",
                    on_update=self._on_update_callback,
                    skip_plan=_is_trivial_goal(text),
                    use_goaltracker=False,
                )
                # Complete the task widget with final stats
                if self._active_task_widget is not None:
                    wall = time.time() - self._task_start_time if self._task_start_time else 0.0
                    GLib.idle_add(
                        self._active_task_widget.complete,
                        str(result.outcome),
                        result.iterations,
                        result.cost_usd,
                        wall,
                    )
                self._active_task_widget = None
                self._task_start_time = None

                # Cancelled → offer Resume/Discard
                if str(result.outcome) == "cancelled":
                    cp = persistence.load_checkpoint()
                    if cp:
                        GLib.idle_add(self._inject_resume_widget, cp)
            except Exception as exc:
                log(f"Error in run_task: {exc}")
                GLib.idle_add(
                    self.add_message, "Genie", f"Error: {exc}", "error"
                )
                self._active_task_widget = None
                self._task_start_time = None
            finally:
                GLib.idle_add(self._reenable_input)

        t = threading.Thread(target=_run_task, daemon=True)
        self._task_thread = t
        t.start()

    # ----- UI state transitions ----------------------------------------------

    def _enter_executing_state(self) -> None:
        """Transition UI to executing state."""
        self._ui_state = "executing"
        self.input_view.set_sensitive(False)
        self.cancel_button.set_sensitive(True)
        self.cancel_button.set_label("Cancel")
        self._start_button.set_visible(False)
        self.status_spinner.set_visible(True)
        self.status_spinner.start()
        self.status_label.set_text("Starting\u2026")
        self._activity_revealer.set_reveal_child(True)
        self._input_scroll.add_css_class("input-busy")

    def _enter_planning_state(self) -> None:
        """Transition UI to planning state."""
        self._ui_state = "planning"
        self.input_view.set_sensitive(True)
        self.cancel_button.set_sensitive(True)
        self.cancel_button.set_label("Cancel")
        self._start_button.set_visible(True)
        self.status_spinner.set_visible(True)
        self.status_spinner.start()
        self.status_label.set_text("Planning\u2026")
        self._activity_revealer.set_reveal_child(True)

    def _enter_idle_state(self) -> None:
        """Transition UI back to idle state."""
        self._ui_state = "idle"
        self._planning_session = None
        self._planning_goal = None
        self._plan_draft_widget = None
        self._start_button.set_visible(False)
        self._reenable_input()

    # ----- Planning session methods ------------------------------------------

    def _start_planning_session(self, goal: str) -> None:
        """Enter planning state and generate draft 1 via LLM."""
        import planner as planner_mod

        self._planning_goal = goal
        self._pending_task_goal = goal
        self._enter_planning_state()
        self.status_label.set_text("Clarifying\u2026")

        def _gen_draft():
            try:
                # -- Phase 5.5: clarifying questions BEFORE planning ------
                enriched_goal = goal
                if config.CLARIFY_ENABLED:
                    orch = self.orchestrator
                    # Ensure the on_update callback is wired so clarify()
                    # can fire QuestionWidget events to the GTK UI.
                    orch._on_update = self._on_update_callback
                    orch._goal = goal
                    if not orch._task_id:
                        orch._task_id = "planning"
                    # Load monthly cost once so _llm_call can accumulate.
                    if orch._monthly_cost_usd == 0.0:
                        orch._load_monthly_cost()

                    clarifications = planner_mod.clarify(orch)
                    if clarifications.get("clarifications"):
                        block = "\n\nClarifications:\n" + "".join(
                            f"- Q: {c['question']}\n  A: {c['answer']}\n"
                            for c in clarifications["clarifications"]
                        )
                        enriched_goal = goal + block
                    self._planning_goal = enriched_goal
                    self._pending_task_goal = enriched_goal
                    # Store complexity for mid-task ask_user budget
                    orch._ask_user_complexity = clarifications.get("complexity", "medium")

                # Bail out if cancelled during clarify
                if self.orchestrator.cancel_event.is_set():
                    self.orchestrator.cancel_event.clear()
                    GLib.idle_add(self._enter_idle_state)
                    return

                GLib.idle_add(
                    self.status_label.set_text, "Generating plan\u2026"
                )
                session = planner_mod.run_planning_session(
                    self.orchestrator, enriched_goal,
                )
                self._planning_session = session
                draft = session.current_draft
                if draft:
                    GLib.idle_add(self._show_plan_draft, draft, 1)
                    GLib.idle_add(
                        self.status_label.set_text,
                        "Plan ready \u2014 refine or Start",
                    )
                    GLib.idle_add(self.status_spinner.stop)
                    GLib.idle_add(self.status_spinner.set_visible, False)
                else:
                    GLib.idle_add(
                        self.add_message, "Genie",
                        "Failed to generate plan. Try again.", "error",
                    )
                    GLib.idle_add(self._enter_idle_state)
            except Exception as exc:
                log(f"Error in planning session: {exc}")
                GLib.idle_add(
                    self.add_message, "Genie", f"Planning error: {exc}", "error",
                )
                GLib.idle_add(self._enter_idle_state)

        self._planning_thread = threading.Thread(target=_gen_draft, daemon=True)
        self._planning_thread.start()

    def _send_planning_refinement(self, text: str) -> None:
        """Send a refinement message to the active planning session."""
        if self._planning_session is None:
            return

        self.status_spinner.set_visible(True)
        self.status_spinner.start()
        self.status_label.set_text("Refining plan\u2026")
        self.input_view.set_sensitive(False)

        session = self._planning_session

        def _refine():
            try:
                draft = session.refine(text)
                if draft:
                    GLib.idle_add(self._show_plan_draft, draft)
                    GLib.idle_add(
                        self.status_label.set_text,
                        "Plan updated \u2014 refine or Start",
                    )
                else:
                    GLib.idle_add(
                        self.add_message, "Genie",
                        "LLM failed to produce a valid plan. Try again.", "error",
                    )
            except Exception as exc:
                log(f"Error in planning refinement: {exc}")
                GLib.idle_add(
                    self.add_message, "Genie", f"Refinement error: {exc}", "error",
                )
            finally:
                GLib.idle_add(self.status_spinner.stop)
                GLib.idle_add(self.status_spinner.set_visible, False)
                GLib.idle_add(self.input_view.set_sensitive, True)
                GLib.idle_add(self.input_view.grab_focus)

        threading.Thread(target=_refine, daemon=True).start()

    def _show_plan_draft(self, subtasks: list, draft_n: int | None = None) -> None:
        """Create and append a NEW PlanDraftWidget to the message list.

        Each refinement creates a fresh widget below the user's refinement
        message instead of replacing Draft 1 in-place (which would appear
        above the refinement prompt that caused it).
        """
        if draft_n is not None:
            self._plan_draft_n = draft_n
        else:
            self._plan_draft_n += 1
        min_st = self._planning_session.min_subtasks if self._planning_session else 0
        widget = PlanDraftWidget(subtasks, self._plan_draft_n, min_subtasks=min_st)
        self._plan_draft_widget = widget
        self._msg_list.append(widget)
        return False

    def _on_plan_start_clicked(self, _button=None) -> None:
        """Called when the user clicks Start to lock the plan and execute."""
        if self._ui_state != "planning" or self._planning_session is None:
            return

        draft = self._planning_session.current_draft
        if not draft:
            return

        goal = self._planning_goal or ""
        _plan_min = getattr(self._planning_session, 'min_subtasks', 0)
        _plan_max = getattr(self._planning_session, 'max_subtasks', None)
        self._planning_session = None
        self._planning_goal = None
        self._plan_draft_widget = None

        # Transition to executing
        self._active_task_widget = None
        self._task_start_time = None
        task_id = getattr(self.orchestrator, '_task_id', None) or str(uuid.uuid4())
        self._active_task_id = task_id
        _chat_logger.log({
            "type": "task_start", "ts": time.time(),
            "task_id": task_id, "goal": goal,
        })

        self._enter_executing_state()

        pre_built = list(draft)  # capture

        def _run_task():
            try:
                result: TaskResult = self.orchestrator.run_task(
                    goal=goal,
                    mode="interactive",
                    task_type="auto",
                    on_update=self._on_update_callback,
                    skip_plan=True,
                    use_goaltracker=True,
                    pre_built_plan=pre_built,
                    plan_min_subtasks=_plan_min,
                    plan_max_subtasks=_plan_max,
                )
                if self._active_task_widget is not None:
                    wall = time.time() - self._task_start_time if self._task_start_time else 0.0
                    GLib.idle_add(
                        self._active_task_widget.complete,
                        str(result.outcome),
                        result.iterations,
                        result.cost_usd,
                        wall,
                    )
                self._active_task_widget = None
                self._task_start_time = None

                # Cancelled → offer Resume/Discard
                if str(result.outcome) == "cancelled":
                    cp = persistence.load_checkpoint()
                    if cp:
                        GLib.idle_add(self._inject_resume_widget, cp)
            except Exception as exc:
                log(f"Error in run_task: {exc}")
                GLib.idle_add(
                    self.add_message, "Genie", f"Error: {exc}", "error",
                )
                self._active_task_widget = None
                self._task_start_time = None
            finally:
                GLib.idle_add(self._reenable_input)

        t = threading.Thread(target=_run_task, daemon=True)
        self._task_thread = t
        t.start()

    def _handle_plan_reentry(self, event_dict: dict) -> None:
        """Handle subtask failure plan reentry — re-enter planning state.

        Called from _on_update_callback on the GTK main thread.
        """
        import planner as planner_mod

        failed = event_dict.get("failed_subtask", {})
        failure_reason = event_dict.get("failure_reason", "unknown")
        continuation_draft_raw = event_dict.get("continuation_draft", [])

        # Build Subtask objects from the draft dicts
        continuation_draft = []
        for s in continuation_draft_raw:
            continuation_draft.append(planner_mod.Subtask(
                n=s.get("n", 1),
                description=s.get("description", ""),
                model_tier=s.get("model_tier", "tier_0"),
            ))

        # Show failure context
        self.add_message(
            "Genie",
            f"Subtask {failed.get('n', '?')} failed: {failure_reason}\n"
            "Entering planning mode for continuation.",
            "error",
        )

        # Detach the current (now-failed) task widget so that post-reentry
        # subtask events create a fresh widget *below* the plan-draft card.
        self._active_task_widget = None

        # Enter planning state for continuation
        self._ui_state = "planning"

        # Create a PlanningSession seeded from the continuation draft so the
        # user can refine the plan via the input box.
        goal = getattr(self, '_pending_task_goal', '') or ''
        session = planner_mod.PlanningSession(self.orchestrator, goal)
        # Prime conversation history with the draft as an assistant turn so
        # subsequent refinements have full context.
        session._messages.append({
            "role": "assistant",
            "content": "\n".join(
                f"{s.n}. [{s.model_tier}] {s.description}"
                for s in continuation_draft
            ),
        })
        session._current_draft = list(continuation_draft)
        self._planning_session = session

        self._start_button.set_visible(True)
        self.input_view.set_sensitive(True)
        self.cancel_button.set_sensitive(True)
        self.status_label.set_text("Replan \u2014 refine or Start")
        self.status_spinner.stop()
        self.status_spinner.set_visible(False)

        # Show the draft as a PlanDraftWidget
        if continuation_draft:
            self._show_plan_draft(continuation_draft, 1)

        # Store context for Start button
        self._reentry_draft = continuation_draft
        self._reentry_event_dict = event_dict

        # Override Start button for reentry flow
        self._start_button.disconnect_by_func(self._on_plan_start_clicked)
        self._start_button.connect("clicked", self._on_reentry_start_clicked)

    def _on_reentry_start_clicked(self, _button=None) -> None:
        """Called when user approves a continuation plan after failure."""
        import planner as planner_mod

        # Prefer the latest refined draft (session.current_draft picks up any
        # edits the user sent via the input box); fall back to the original.
        session = getattr(self, '_planning_session', None)
        if session is not None and session.current_draft:
            draft = session.current_draft
        else:
            draft = getattr(self, "_reentry_draft", None)

        if not draft:
            return

        # Reconnect normal Start handler
        self._start_button.disconnect_by_func(self._on_reentry_start_clicked)
        self._start_button.connect("clicked", self._on_plan_start_clicked)

        self._start_button.set_visible(False)
        self._ui_state = "executing"
        self.input_view.set_sensitive(False)
        self.status_spinner.set_visible(True)
        self.status_spinner.start()
        self.status_label.set_text("Resuming\u2026")

        # Null the task widget so post-reentry subtask events create a new
        # widget below the plan-draft card rather than appending to the old one.
        self._active_task_widget = None
        self._task_start_time = None

        # Signal orchestrator to continue with approved plan
        self.orchestrator.answer_plan_reentry(draft)

        self._reentry_draft = None
        self._reentry_event_dict = None
        self._planning_session = None

    def _on_update_callback(self, event_dict: dict) -> None:
        """Called from the brain loop thread with progress events.

        All GTK operations are dispatched via GLib.idle_add.
        """
        # Clarifying question (Phase 5.5) — fired directly by planner.clarify()
        if event_dict.get("clarification_question"):
            question = event_dict.get("question", "")
            options  = event_dict.get("options", [])
            GLib.idle_add(self._show_question, question, options)
            return

        if event_dict.get("plan_confirm"):
            plan = event_dict.get("plan", {})
            GLib.idle_add(self._show_plan_confirm, plan)
            return

        if event_dict.get("plan_reentry"):
            GLib.idle_add(self._handle_plan_reentry, event_dict)
            return

        action  = event_dict.get("action")
        result  = event_dict.get("result")
        message = event_dict.get("message")
        outcome = event_dict.get("outcome")

        # Subtask boundary event — fired by _run_goaltracker_loop
        if event_dict.get("subtask_n") is not None:
            sn    = event_dict["subtask_n"]
            sdesc = event_dict.get("subtask_description", "")
            stot  = event_dict.get("subtask_total", sn)
            GLib.idle_add(self._ensure_task_widget)
            # Use a nested idle so the widget definitely exists when we call add_subtask_header
            def _add_st_header(sn=sn, sdesc=sdesc, stot=stot):
                if self._active_task_widget is not None:
                    self._active_task_widget.add_subtask_header(sn, sdesc, stot)
                    self._active_task_widget.update_progress(sn / stot)
            GLib.idle_add(_add_st_header)
            return

        # ---- Transient status events (thinking / LLM retries) ----
        transient = event_dict.get("transient_status")
        if transient is not None:
            iter_n = event_dict.get("iteration", 0)
            if transient == "thinking":
                GLib.idle_add(
                    self.status_label.set_text, f"Thinking\u2026 iter {iter_n}"
                )
                GLib.idle_add(
                    self.status_spinner.add_css_class, "activity-spinner-thinking"
                )
            elif transient.startswith("llm_retry_"):
                # e.g. "llm_retry_1_of_3"
                parts = transient.split("_")
                attempt = parts[2] if len(parts) > 2 else "?"
                total = parts[4] if len(parts) > 4 else "?"
                GLib.idle_add(
                    self.status_label.set_text,
                    f"\u26a0 LLM retry {attempt}/{total}",
                )
            return  # no widget entry, no chat message

        # ---- Action progress → route into CollapsibleTaskWidget ----
        if action is not None and result is not None:
            iter_n = event_dict.get("iteration", 1)
            args = event_dict.get("args")
            observation = event_dict.get("observation", "")

            # Capture estimated_iterations and task_budget on first event
            if self._current_estimated_iters is None:
                est = event_dict.get("estimated_iterations")
                if est is not None:
                    self._current_estimated_iters = est
            if self._current_task_budget is None:
                budget = event_dict.get("task_budget_usd")
                if budget is not None and budget > 0:
                    self._current_task_budget = budget

            # Lazily create the task widget on first action
            if self._active_task_widget is None:
                GLib.idle_add(self._ensure_task_widget)

            # Remove thinking CSS class (back to green)
            GLib.idle_add(
                self.status_spinner.remove_css_class, "activity-spinner-thinking"
            )

            # Activity bar: spinner + rich status label
            GLib.idle_add(self.status_spinner.start)
            GLib.idle_add(self.status_spinner.set_visible, True)
            status_text = _format_action_status(action, args)
            GLib.idle_add(self.status_label.set_text, status_text)

            # Update progress pill
            def _update_progress_pill(it=iter_n, est=self._current_estimated_iters):
                if est is not None:
                    self._progress_pill.set_text(f"iter {it} / ~{est}")
                else:
                    self._progress_pill.set_text(f"iter {it}")
                self._progress_pill.set_visible(True)
            GLib.idle_add(_update_progress_pill)

            # Update cost pill
            task_cost = event_dict.get("task_cost_usd", 0.0)
            def _update_cost_pill(cost=task_cost, budget=self._current_task_budget):
                if budget is not None and budget > 0:
                    self._cost_pill.set_text(f"${cost:.3f} / ${budget:.2f}")
                    ratio = cost / budget
                    # Remove old color classes
                    self._cost_pill.remove_css_class("cost-pill-warn")
                    self._cost_pill.remove_css_class("cost-pill-danger")
                    if ratio >= 0.8:
                        self._cost_pill.add_css_class("cost-pill-danger")
                    elif ratio >= 0.5:
                        self._cost_pill.add_css_class("cost-pill-warn")
                else:
                    self._cost_pill.set_text(f"${cost:.3f}")
                self._cost_pill.set_visible(True)
            GLib.idle_add(_update_cost_pill)

            # Update task widget progress bar
            if self._current_estimated_iters and self._active_task_widget is not None:
                frac = iter_n / self._current_estimated_iters
                GLib.idle_add(self._active_task_widget.update_progress, frac)

            # Snapshot previous error state, route action to widget
            prev_error = self._last_iter_had_error
            def _deferred_add_action(it=iter_n, act=action, res=str(result), err=prev_error):
                if self._active_task_widget is not None:
                    self._active_task_widget.add_action(it, act, res, err)
            GLib.idle_add(_deferred_add_action)

            # Update _last_iter_had_error based on *result*, not observation text.
            # Using "error" in obs_str was overly broad — file contents that
            # happen to contain the word "error" triggered a false-positive red X.
            _result_str = str(result).lower() if result else ""
            self._last_iter_had_error = (
                _result_str in ("command_failed", "environmental_failure",
                                "unrecoverable", "command_timeout")
                or action == "schema_validation_error"
            )

            _chat_logger.log({
                "type": "action", "ts": time.time(),
                "task_id": getattr(self, '_active_task_id', '') or '',
                "iter_n": iter_n, "action_name": action,
                "result_str": str(result),
            })

        # Chat message — real conversational replies stay in the main stream.
        # Skip when action == "chat": the message is already rendered inside
        # the CollapsibleTaskWidget as an action row.  Adding it again as a
        # top-level MessageWidget causes it to appear *below* the task widget
        # while new iterations keep rendering *inside* the widget above it,
        # making the chat messages look "locked to the bottom".
        if message is not None and outcome is None and action != "chat":
            GLib.idle_add(self.add_message, "Genie", message, "chat")

        # Final outcome → complete the task widget, clear tracking
        if outcome is not None:
            # is_subtask_outcome: an intermediate subtask inside a GoalTracker
            # completed — route its message inline and keep the shared task
            # widget alive; only the final non-subtask outcome finalises it.
            if event_dict.get("is_subtask_outcome"):
                if message and self._active_task_widget is not None:
                    GLib.idle_add(
                        self._active_task_widget.add_subtask_summary, message
                    )
                return
            if message:
                GLib.idle_add(self.add_message, "Genie", message, "chat")
            if self._active_task_widget is not None:
                iterations = event_dict.get("iterations", len(self._active_task_widget._iterations))
                cost_usd = event_dict.get("cost_usd", 0.0)
                wall_time_s = (time.time() - self._task_start_time
                               if self._task_start_time else 0.0)
                GLib.idle_add(
                    self._active_task_widget.complete,
                    str(outcome), iterations, cost_usd, wall_time_s,
                )
                _chat_logger.log({
                    "type": "task_complete", "ts": time.time(),
                    "task_id": getattr(self, '_active_task_id', '') or '',
                    "outcome": str(outcome), "iterations": iterations,
                    "cost_usd": cost_usd, "wall_time_s": wall_time_s,
                })
            self._active_task_widget = None
            self._active_task_id = None
            self._task_start_time = None
            GLib.idle_add(self.status_spinner.stop)
            GLib.idle_add(self.status_spinner.set_visible, False)
            GLib.idle_add(self.status_label.set_text, "")
            GLib.idle_add(self._reenable_input)

    def _ensure_task_widget(self) -> None:
        """Lazily create the CollapsibleTaskWidget the first time execution events
        arrive.  Must be called on the GTK main thread (via GLib.idle_add).
        """
        if self._active_task_widget is not None:
            return
        goal = getattr(self, "_pending_task_goal", "") or "Task"
        task_w = CollapsibleTaskWidget(goal)
        self._msg_list.append(task_w)
        self._active_task_widget = task_w
        if self._task_start_time is None:
            self._task_start_time = time.time()

    def _show_plan_confirm(self, plan: dict) -> None:
        """Inject a PlanConfirmWidget into _msg_list.

        Called on the GTK main thread via GLib.idle_add.
        The widget stays in the stream as a historical record.
        Returns False (GLib.idle_add convention).
        """
        widget = PlanConfirmWidget(plan, self.orchestrator, self)
        self._msg_list.append(widget)
        return False

    def _show_question(self, question: str, options: list[str]) -> None:
        """Inject a QuestionWidget into _msg_list.

        Called on the GTK main thread via GLib.idle_add.
        The widget stays in the stream as a historical record.
        """
        widget = QuestionWidget(question, options, self.orchestrator, self)
        self._msg_list.append(widget)
        self._scroll_to_bottom()
        return False

    def on_cancel(self, _button) -> None:
        """Called when the user clicks the Cancel button."""
        if self._ui_state == "planning":
            # Discard plan and return to idle
            self._planning_session = None
            self._planning_goal = None
            self._plan_draft_widget = None
            # If reentry was in progress, signal cancellation to orchestrator
            if hasattr(self, '_reentry_event_dict') and self._reentry_event_dict:
                self.orchestrator.answer_plan_reentry(None)
                self._reentry_draft = None
                self._reentry_event_dict = None
            # Unblock any pending clarify question so the planning thread
            # doesn't hang on _clarify_event.wait() forever.
            self.orchestrator.cancel()
            GLib.idle_add(self._enter_idle_state)
            return
        self.orchestrator.cancel()
        self.cancel_button.set_label("Cancelling\u2026")
        self.cancel_button.set_sensitive(False)
        self.status_label.set_text("Cancelling\u2026 waiting for current action to finish")

    def _reenable_input(self) -> None:
        """Re-enable input controls after task completion."""
        self._ui_state = "idle"
        self._task_thread = None
        self.status_spinner.stop()
        self.status_spinner.set_visible(False)
        self.status_spinner.remove_css_class("activity-spinner-thinking")
        self.status_label.set_text("")
        self._activity_revealer.set_reveal_child(False)
        self._input_scroll.remove_css_class("input-busy")
        self.input_view.set_sensitive(True)
        self.cancel_button.set_sensitive(False)
        self.cancel_button.set_label("Cancel")
        self._start_button.set_visible(False)
        # Reset visibility overhaul state
        self._progress_pill.set_visible(False)
        self._cost_pill.set_visible(False)
        self._cost_pill.remove_css_class("cost-pill-warn")
        self._cost_pill.remove_css_class("cost-pill-danger")
        self._current_estimated_iters = None
        self._current_task_budget = None
        self._last_iter_had_error = False
        self.input_view.grab_focus()
        return False

    def _inject_resume_widget(self, checkpoint: dict) -> None:
        """Add a ResumeWidget to the message list for a paused/crashed task."""
        widget = ResumeWidget(checkpoint, self.orchestrator, self)
        self._msg_list.append(widget)
        return False

    def do_close_request(self) -> bool:
        """Handle window close — cancel running task and wait briefly."""
        if self._ui_state == "executing":
            self.orchestrator.cancel()
            if self._task_thread is not None:
                self._task_thread.join(timeout=3)
        return False

    def _on_input_focus_in(self, controller) -> None:
        """Clear placeholder hint on focus."""
        buf = self.input_view.get_buffer()
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        if text == self._hint_text:
            buf.set_text("")
            self._showing_hint = False

    def _on_input_focus_out(self, controller) -> None:
        """Restore placeholder hint if input is empty."""
        buf = self.input_view.get_buffer()
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()
        if not text:
            buf.set_text(self._hint_text)
            buf.apply_tag_by_name(
                "hint_text",
                buf.get_start_iter(),
                buf.get_end_iter(),
            )
            self._showing_hint = True

    # ---- Prompt injection & floating selection button -----------------------

    def _inject_into_prompt(self, text: str) -> None:
        """Inject *text* as a quoted block into the input buffer."""
        quoted = f"> {text}\n\n"
        buf = self.input_view.get_buffer()
        current = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        if not current.strip() or self._showing_hint:
            buf.set_text(quoted)
        else:
            buf.insert(buf.get_start_iter(), quoted)
        self._showing_hint = False
        buf.place_cursor(buf.get_end_iter())
        self.input_view.grab_focus()


# =============================================================================
# GTK Application
# =============================================================================

class GenieApplication(Gtk.Application):
    """Top-level GTK Application that owns the main window.

    Uses GtkApplication single-instance mode so that running
    ``python genie.py --show`` while Genie is already open simply
    brings the existing window to the front instead of starting a
    second instance.
    """

    def __init__(self):
        super().__init__(
            application_id="com.genie.assistant",
            flags=Gio.ApplicationFlags.HANDLES_COMMAND_LINE,
        )
        self.main_window = None

    def do_command_line(self, command_line):
        """Handle command-line invocation (including --show).

        GtkApplication routes all CLI invocations here when
        HANDLES_COMMAND_LINE is set.  We simply trigger activate,
        which will create or present the window.
        """
        self.activate()
        return 0

    def do_activate(self):
        if self.main_window is not None:
            self.main_window.present()
            return
        self.main_window = GenieWindow()
        self.main_window.set_application(self)
        self.main_window.present()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    log("Starting Genie…")
    app = GenieApplication()
    app.run(sys.argv)
