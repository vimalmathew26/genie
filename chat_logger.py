"""Append-only JSONL chat logger with session management. Extracted from genie.py."""

import datetime
import json
import os
import queue
import threading
import time
import uuid


class _ChatLogger:
    """Fire-and-forget JSONL writer for session-based chat persistence.

    All writes go through a queue consumed by a single daemon thread
    so the GTK main thread is never blocked.
    Sessions live in logs/chats/ as individual JSONL files.
    """

    def __init__(self):
        self._log_dir = os.path.join(os.path.dirname(__file__), "logs")
        self._chats_dir = os.path.join(self._log_dir, "chats")
        os.makedirs(self._chats_dir, exist_ok=True)
        self._migrate_legacy()

        # Determine active session — most recent, or create new
        sessions = self.list_sessions()
        if sessions:
            self._path = os.path.join(self._chats_dir, sessions[0])
        else:
            self._path = self._make_session_file()

        self._queue: queue.Queue = queue.Queue()
        _t = threading.Thread(target=self._writer, daemon=True)
        _t.start()

    # -- migration ------------------------------------------------------------

    def _migrate_legacy(self) -> None:
        """One-time migration: copy chat_log.jsonl into logs/chats/."""
        legacy = os.path.join(self._log_dir, "chat_log.jsonl")
        if not os.path.isfile(legacy):
            return
        try:
            if any(f.startswith("session_") for f in os.listdir(self._chats_dir)):
                return
        except OSError:
            return
        try:
            import shutil
            ts_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            short_id = uuid.uuid4().hex[:4]
            dest = os.path.join(
                self._chats_dir, f"session_{ts_str}_{short_id}.jsonl"
            )
            shutil.copy2(legacy, dest)
        except Exception:
            pass  # best-effort

    # -- session file management ----------------------------------------------

    def _make_session_file(self) -> str:
        """Create a new empty session file and return its full path."""
        ts_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        short_id = uuid.uuid4().hex[:4]
        path = os.path.join(
            self._chats_dir, f"session_{ts_str}_{short_id}.jsonl"
        )
        open(path, "a").close()
        return path

    def new_session(self) -> str:
        """Create a new session and make it active.  Returns filename."""
        self._path = self._make_session_file()
        return os.path.basename(self._path)

    def switch_session(self, filename: str) -> None:
        """Make an existing session file the active write target."""
        self._path = os.path.join(self._chats_dir, filename)

    @property
    def active_filename(self) -> str:
        return os.path.basename(self._path)

    def list_sessions(self, limit: int = 50) -> list[str]:
        """Return session filenames sorted most-recent-first."""
        try:
            files = [
                f for f in os.listdir(self._chats_dir)
                if f.startswith("session_") and f.endswith(".jsonl")
            ]
        except OSError:
            return []
        files.sort(reverse=True)
        return files[:limit]

    # -- background writer ----------------------------------------------------

    def _writer(self) -> None:
        while True:
            record = self._queue.get()
            try:
                with open(self._path, "a") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass  # fire-and-forget — never crash the app

    # -- public API -----------------------------------------------------------

    def log(self, record: dict) -> None:
        """Enqueue a record for async write.  Never blocks."""
        self._queue.put_nowait(record)

    def read_session(self, filename: str, n: int = 200) -> list[dict]:
        """Return the last *n* valid records from a session file."""
        path = os.path.join(self._chats_dir, filename)
        try:
            with open(path, "r") as fh:
                lines = fh.readlines()
        except (FileNotFoundError, OSError):
            return []
        records: list[dict] = []
        for line in lines[-n:]:
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
        return records

    def read_session_head(self, filename: str, n: int = 30) -> list[dict]:
        """Return the first *n* valid records from a session file."""
        path = os.path.join(self._chats_dir, filename)
        try:
            records: list[dict] = []
            with open(path, "r") as fh:
                for i, line in enumerate(fh):
                    if i >= n:
                        break
                    try:
                        records.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        continue
            return records
        except (FileNotFoundError, OSError):
            return []

    def session_meta(self, filename: str) -> tuple[str, str]:
        """Return (date_display, first_user_msg_preview) for a session."""
        # Parse date from filename: session_YYYY-MM-DDTHH-MM-SS_xxxx.jsonl
        date_str = ""
        try:
            ts_part = filename.replace("session_", "").rsplit("_", 1)[0]
            dt = datetime.datetime.strptime(ts_part, "%Y-%m-%dT%H-%M-%S")
            date_str = dt.strftime("%b %d \u00b7 %H:%M")
        except (ValueError, IndexError):
            pass
        preview = "Empty session"
        for rec in self.read_session_head(filename, 30):
            if rec.get("type") == "msg" and rec.get("sender") == "You":
                text = rec.get("text", "")
                preview = (text[:40] + "\u2026") if len(text) > 40 else text
                break
        return date_str, preview


chat_logger = _ChatLogger()
