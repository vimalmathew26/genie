#!/usr/bin/env python3
"""
Genie — review_bot.py
Telegram bot for the Human Review Pipeline (§3.3).

Usage:
    python review_bot.py          # long-poll mode (default)
    python review_bot.py --once   # check once and exit (for cron / testing)

The bot:
  1. Reads pending entries from the review queue.
  2. Sends each to the configured Telegram chat as a formatted message
     with inline keyboard buttons: ✅ Approve | ❌ Reject.
  3. On Approve → marks 'approved' → sends confirmation.
  4. On Reject  → asks for feedback → marks 'rejected' with feedback.
  5. The orchestrator's revision loop picks up rejected tasks.

Requires:  TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment / config.
No external dependencies beyond stdlib (uses Telegram Bot HTTP API directly).
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.parse
import urllib.request

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_POLL_TIMEOUT
from review_queue import ReviewQueue


# =========================================================================
# Telegram HTTP helpers (no external library needed)
# =========================================================================

_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def _tg_request(method: str, payload: dict) -> dict:
    """POST to Telegram Bot API, return parsed JSON response."""
    url = f"{_API}/{method}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=TELEGRAM_POLL_TIMEOUT + 10)
        return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[review_bot] Telegram API error ({method}): {exc}", file=sys.stderr)
        return {}


def send_message(text: str, reply_markup: dict | None = None) -> dict:
    """Send a plain message (with optional inline keyboard)."""
    payload: dict = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    return _tg_request("sendMessage", payload)


def answer_callback(callback_query_id: str, text: str = "") -> None:
    """Acknowledge a callback query (removes the ⏳ spinner)."""
    _tg_request("answerCallbackQuery", {
        "callback_query_id": callback_query_id,
        "text": text,
    })


def edit_message_text(chat_id: int | str, message_id: int, text: str) -> None:
    """Edit an already-sent message (removes inline buttons)."""
    _tg_request("editMessageText", {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
    })


def delete_message(chat_id: int | str, message_id: int) -> None:
    """Delete a message from the chat."""
    _tg_request("deleteMessage", {
        "chat_id": chat_id,
        "message_id": message_id,
    })


def _schedule_delete(chat_id: int | str, message_id: int, delay: float = 300.0) -> None:
    """Delete *message_id* after *delay* seconds (default 5 min) in a daemon thread."""
    def _worker() -> None:
        time.sleep(delay)
        delete_message(chat_id, message_id)
    t = threading.Thread(target=_worker, daemon=True, name=f"msg-delete-{message_id}")
    t.start()


# =========================================================================
# Review message formatting
# =========================================================================

def _format_review_message(entry: dict) -> str:
    """Format a review queue entry for Telegram (plain text)."""
    files = ", ".join(entry.get("output_files", [])) or "(none)"
    rev = entry.get("revision_count", 0)
    rev_tag = f" (revision #{rev})" if rev else ""
    tid = entry['task_id'][:8]
    return (
        f"New task for review{rev_tag}\n\n"
        f"Goal: {entry['goal'][:300]}\n\n"
        f"Summary: {entry['summary'][:200]}\n"
        f"Output files: {files}\n"
        f"Cost: ${entry['cost_usd']:.4f} | Iterations: {entry['iterations']}\n"
        f"Task ID: {tid}..."
    )


def _inline_keyboard(task_id: str) -> dict:
    """Inline keyboard with Approve / Reject buttons."""
    return {
        "inline_keyboard": [[
            {"text": "✅ Approve", "callback_data": f"approve:{task_id}"},
            {"text": "❌ Reject",  "callback_data": f"reject:{task_id}"},
        ]]
    }


# =========================================================================
# Poll loop: listen for Approve / Reject button presses
# =========================================================================

# Track which task_ids have been sent in THIS session to avoid re-sending.
_notified_task_ids: set[str] = set()
# Track task_ids waiting for rejection feedback text.
_awaiting_feedback: dict[str, dict] = {}  # task_id → {"chat_id", "message_id"}


def poll_once(queue: ReviewQueue, offset: int) -> int:
    """One getUpdates cycle. Returns the new offset."""
    params = urllib.parse.urlencode({
        "offset": offset,
        "timeout": TELEGRAM_POLL_TIMEOUT,
        "allowed_updates": json.dumps(["callback_query", "message"]),
    })
    url = f"{_API}/getUpdates?{params}"
    try:
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=TELEGRAM_POLL_TIMEOUT + 10)
        data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[review_bot] poll error: {exc}", file=sys.stderr)
        return offset

    if not data.get("ok"):
        return offset

    for update in data.get("result", []):
        offset = max(offset, update["update_id"] + 1)

        # ── Callback query (button press) ────────────────────────────
        cb = update.get("callback_query")
        if cb:
            cb_data = cb.get("data", "")
            cb_id = cb["id"]
            chat_id = cb["message"]["chat"]["id"]
            message_id = cb["message"]["message_id"]

            if cb_data.startswith("approve:"):
                task_id = cb_data.split(":", 1)[1]
                queue.update_status(task_id, "approved")
                answer_callback(cb_id, "Approved!")
                edit_message_text(
                    chat_id, message_id,
                    f"Approved — Task {task_id[:8]}... marked for delivery.",
                )
                _schedule_delete(chat_id, message_id)
                queue.mark_delivered(task_id)
                print(f"[review_bot] Approved: {task_id}")

            elif cb_data.startswith("reject:"):
                task_id = cb_data.split(":", 1)[1]
                answer_callback(cb_id, "Please send rejection feedback as a text message.")
                r = send_message(
                    f"Rejecting task {task_id[:8]}...\n\n"
                    f"Please reply with your feedback for Genie to revise:"
                )
                prompt_msg_id = r.get("result", {}).get("message_id")
                if prompt_msg_id:
                    _schedule_delete(chat_id, prompt_msg_id)
                _awaiting_feedback[task_id] = {
                    "chat_id": chat_id,
                    "message_id": message_id,
                }
                print(f"[review_bot] Reject requested for {task_id}, awaiting feedback...")
            continue

        # ── Text message (rejection feedback) ─────────────────────────
        msg = update.get("message", {})
        text = msg.get("text", "").strip()
        if text and _awaiting_feedback:
            # The most recently rejected task gets this feedback.
            task_id = next(iter(_awaiting_feedback))
            info = _awaiting_feedback.pop(task_id)
            queue.update_status(task_id, "rejected", feedback=text)
            edit_message_text(
                info["chat_id"], info["message_id"],
                f"Rejected — Task {task_id[:8]}...\nFeedback: {text[:300]}",
            )
            _schedule_delete(info["chat_id"], info["message_id"])
            r = send_message(f"Feedback recorded for {task_id[:8]}... Genie will revise.")
            conf_msg_id = r.get("result", {}).get("message_id")
            if conf_msg_id:
                _schedule_delete(info["chat_id"], conf_msg_id)
            print(f"[review_bot] Rejected {task_id} with feedback: {text[:80]}")

    return offset


def run_bot_in_thread(stop_event) -> None:
    """Run the bot loop in a background thread until stop_event is set.

    Designed to be called from orchestrator.run_task(review=True) so no
    separate terminal is needed.

    Usage::
        import threading, review_bot
        stop = threading.Event()
        t = threading.Thread(target=review_bot.run_bot_in_thread, args=(stop,), daemon=True)
        t.start()
        ...  # do work
        stop.set(); t.join(timeout=5)
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[review_bot] No Telegram credentials — bot thread not starting.",
              file=sys.stderr)
        return

    queue = ReviewQueue()
    offset = 0

    while not stop_event.is_set():
        try:
            # Notify any pending entries we haven't sent yet
            for entry in queue.get_pending():
                tid = entry["task_id"]
                if tid not in _notified_task_ids:
                    msg = _format_review_message(entry)
                    kb = _inline_keyboard(tid)
                    result = send_message(msg, reply_markup=kb)
                    if result.get("ok"):
                        _notified_task_ids.add(tid)
                        print(f"[review_bot] Notified: {tid}")
                        msg_id = result.get("result", {}).get("message_id")
                        if msg_id:
                            _schedule_delete(TELEGRAM_CHAT_ID, msg_id)

            offset = poll_once(queue, offset)
        except Exception as exc:  # noqa: BLE001
            print(f"[review_bot] thread error: {exc}", file=sys.stderr)
            stop_event.wait(2.0)


def run_bot(once: bool = False) -> None:
    """Main entry point: notify pending tasks and poll for decisions."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[review_bot] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Exiting.",
              file=sys.stderr)
        sys.exit(1)

    queue = ReviewQueue()
    offset = 0

    print(f"[review_bot] Starting Telegram review bot...")
    print(f"[review_bot] Queue path: {queue.path}")

    # Initial scan: notify all pending reviews
    pending = queue.get_pending()
    for entry in pending:
        tid = entry["task_id"]
        if tid not in _notified_task_ids:
            msg = _format_review_message(entry)
            kb = _inline_keyboard(tid)
            result = send_message(msg, reply_markup=kb)
            if result.get("ok"):
                _notified_task_ids.add(tid)
                print(f"[review_bot] Notified: {tid}")
                msg_id = result.get("result", {}).get("message_id")
                if msg_id:
                    _schedule_delete(TELEGRAM_CHAT_ID, msg_id)

    if once:
        # Single poll cycle, then exit
        poll_once(queue, offset)
        return

    # Continuous polling
    print(f"[review_bot] Polling for review decisions (Ctrl+C to stop)...")
    try:
        while True:
            # Check for new pending entries
            for entry in queue.get_pending():
                tid = entry["task_id"]
                if tid not in _notified_task_ids:
                    msg = _format_review_message(entry)
                    kb = _inline_keyboard(tid)
                    result = send_message(msg, reply_markup=kb)
                    if result.get("ok"):
                        _notified_task_ids.add(tid)
                        print(f"[review_bot] Notified new: {tid}")
                        msg_id = result.get("result", {}).get("message_id")
                        if msg_id:
                            _schedule_delete(TELEGRAM_CHAT_ID, msg_id)

            offset = poll_once(queue, offset)
    except KeyboardInterrupt:
        print("\n[review_bot] Stopped.")


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    once = "--once" in sys.argv
    run_bot(once=once)
