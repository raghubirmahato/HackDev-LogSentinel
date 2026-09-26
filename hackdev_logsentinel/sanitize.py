"""Make untrusted log content safe to display."""

from __future__ import annotations

import re

# C0/C1 control characters (ANSI escape sequences, BEL, CR/LF, the 8-bit CSI)
# and Unicode bidi overrides ("Trojan Source"-style spoofing of what a
# displayed path appears to say).
_UNSAFE_CHARS = re.compile("[\x00-\x1f\x7f-\x9f‪-‮⁦-⁩]")


def _escape(match: re.Match) -> str:
    code = ord(match.group())
    return f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}"


def printable(value: object, max_length: int = 0) -> str:
    """Render `value` for a terminal or chat message: control and bidi
    characters become visible escapes (so request data can't inject ANSI
    escape sequences or forge extra output lines), and the result is
    truncated to `max_length` characters when that is non-zero."""
    text = _UNSAFE_CHARS.sub(_escape, str(value))
    if max_length and len(text) > max_length:
        text = text[: max_length - 1] + "…"
    return text
