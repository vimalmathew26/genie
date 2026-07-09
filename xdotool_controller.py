"""XdotoolController — desktop automation via xdotool subprocess calls."""

import subprocess

from config import log


# Mapping from LLM key names to xdotool keysym names
_XDOTOOL_KEY_MAP = {
    "enter": "Return",
    "escape": "Escape",
    "backspace": "BackSpace",
    "delete": "Delete",
    "tab": "Tab",
    "space": "space",
    "insert": "Insert",
    "home": "Home",
    "end": "End",
    "pageup": "Prior",
    "pagedown": "Next",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4",
    "f5": "F5", "f6": "F6", "f7": "F7", "f8": "F8",
    "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
}


class XdotoolController:
    """Desktop automation through xdotool (X11-compatible)."""

    @staticmethod
    def _run(cmd: list[str]) -> subprocess.CompletedProcess:
        """Run a subprocess command, logging and handling errors."""
        log(f"Executing: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                log(f"xdotool stderr: {result.stderr.strip()}")
            return result
        except FileNotFoundError:
            log("ERROR: xdotool not found. Is it installed and on PATH?")
            raise
        except subprocess.TimeoutExpired:
            log("ERROR: xdotool command timed out.")
            raise

    def type_text(self, text: str, delay_ms: int = 50) -> None:
        """Type a string into the currently focused window.

        delay_ms: inter-keystroke delay passed to xdotool --delay. Default 50ms
        is conservative for reliable general use. Pass 0 for maximum speed when
        the target widget handles rapid input (e.g. address bar after ctrl+l).
        """
        self._run(["xdotool", "type", "--clearmodifiers",
                   "--delay", str(delay_ms), "--", text])

    def type_to_window(self, wid: int, text: str) -> None:
        """Focus window by WID then type text. Used for terminal tier where --window targeting is unsupported."""
        subprocess.run(
            ["xdotool", "windowfocus", "--sync", str(wid)],
            check=True,
        )
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "50", "--", text],
            check=True,
        )

    def press_key(self, key: str) -> str:
        """
        Simulate a key press using xdotool keysym names.

        Translates the LLM's colon-format (e.g. "ctrl:c", "alt:tab")
        to xdotool's plus-format (e.g. "ctrl+c", "alt+Tab").

        Args:
            key: Key descriptor, e.g. "enter", "ctrl:c", "alt:tab".

        Returns:
            Error message string if the key is unknown, else empty string.
        """
        translated = self._translate_key(key.lower())
        if translated is None:
            log(f"ERROR: Unknown key '{key}'.")
            return f"Unknown key: {key}"
        self._run(["xdotool", "key", translated])
        return ""

    @staticmethod
    def _translate_key(key: str) -> str | None:
        """
        Translate an LLM key descriptor to an xdotool keysym string.

        Returns None if the key cannot be translated.
        """
        # Combo key — accept both colon-format ("ctrl:c") and plus-format ("ctrl+l")
        delimiter = ":" if ":" in key else "+" if "+" in key else None
        if delimiter is not None:
            parts = key.split(delimiter)
            translated_parts = []
            for part in parts:
                mapped = _XDOTOOL_KEY_MAP.get(part)
                if mapped:
                    translated_parts.append(mapped)
                elif len(part) == 1 and (part.isalpha() or part.isdigit()):
                    translated_parts.append(part)
                elif part in ("ctrl", "alt", "shift", "super"):
                    translated_parts.append(part)
                else:
                    return None
            return "+".join(translated_parts)

        # Single key — check map first, then letters/digits/punctuation
        mapped = _XDOTOOL_KEY_MAP.get(key)
        if mapped:
            return mapped
        if len(key) == 1 and (key.isalpha() or key.isdigit()):
            return key
        return None

    def click(self, x: int, y: int) -> None:
        """Move the mouse to (x, y) and perform a left click."""
        self._run(["xdotool", "mousemove", "--sync", str(x), str(y)])
        self._run(["xdotool", "click", "1"])
