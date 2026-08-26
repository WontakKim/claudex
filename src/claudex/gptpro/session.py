"""Storage-state persistence and static validation for gptpro sessions."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from claudex import paths
from claudex.providers.auth_support import write_private_json_atomic

AUTH_COOKIE_PREFIX = "__Secure-next-auth.session-token"
AUTH_COOKIE_NAME_PATTERN = re.compile("^" + AUTH_COOKIE_PREFIX.replace(".", r"\."))
_LOGIN_COMMAND = "run claudex-gateway gptpro login"


class GptProSessionError(Exception):
    """Raised when a gptpro storage-state file cannot be used safely."""


def _load_storage_state(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GptProSessionError(f"cannot read gptpro session file at {path}") from exc
    try:
        state = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise GptProSessionError(
            f"gptpro session file at {path} is not valid JSON"
        ) from exc
    if not isinstance(state, dict):
        raise GptProSessionError(f"gptpro session file at {path} has an invalid format")
    if not isinstance(state.get("cookies"), list) or not isinstance(
        state.get("origins"), list
    ):
        raise GptProSessionError(f"gptpro session file at {path} has an invalid format")
    return state


def find_auth_cookie(cookies: object) -> dict[str, Any] | None:
    """Select the first cookie accepted by the shared ChatGPT auth policy.

    A qualifying cookie must carry a non-empty value and the chatgpt.com
    domain: a name-only match would report sessions Playwright itself
    rejects at load time (missing value) or that belong to another site.
    """
    if not isinstance(cookies, list):
        return None
    for cookie in cookies:
        if not isinstance(cookie, dict):
            continue
        name = cookie.get("name")
        if not (isinstance(name, str) and AUTH_COOKIE_NAME_PATTERN.match(name)):
            continue
        value = cookie.get("value")
        if not (isinstance(value, str) and value):
            continue
        if cookie.get("domain") not in ("chatgpt.com", ".chatgpt.com"):
            continue
        return cookie
    return None


def load_auth_cookie_expiry(path: Path) -> float | None:
    """Load the first ChatGPT authentication cookie's expiry timestamp."""
    state = _load_storage_state(path)
    auth_cookie = find_auth_cookie(state["cookies"])

    if auth_cookie is None:
        raise GptProSessionError(
            f"gptpro session file at {path} has no authentication cookie"
        )

    expires = auth_cookie.get("expires")
    if expires is None:
        return None
    if isinstance(expires, bool) or not isinstance(expires, (int, float)):
        raise GptProSessionError(
            f"gptpro session file at {path} has an invalid cookie expiry"
        )
    return float(expires)


def is_expired(expires: float | None, now: float) -> bool:
    """Return whether a persistent cookie expired under the plugin rule."""
    return expires is not None and expires > 0 and expires < now


async def save_storage_state(context: Any, path: Path) -> None:
    """Durably persist Playwright storage state with private permissions."""
    try:
        state = await context.storage_state()
    except Exception as exc:
        raise GptProSessionError("cannot read Playwright storage state") from exc
    if not isinstance(state, dict):
        raise GptProSessionError("Playwright returned an invalid storage state")

    try:
        write_private_json_atomic(path, state)
    except Exception as exc:
        raise GptProSessionError(
            f"cannot save gptpro session file at {path}"
        ) from exc


def session_status() -> dict[str, object]:
    """Return a static status snapshot without contacting ChatGPT."""
    path = paths.gptpro_session_file()
    if not path.exists():
        return {
            "path": path,
            "exists": False,
            "has_auth_cookie": False,
            "expired": None,
            "valid": False,
            "message": f"{_LOGIN_COMMAND}; no gptpro session was found",
        }

    try:
        expires = load_auth_cookie_expiry(path)
    except GptProSessionError:
        return {
            "path": path,
            "exists": True,
            "has_auth_cookie": False,
            "expired": None,
            "valid": False,
            "message": f"{_LOGIN_COMMAND}; the saved gptpro session is invalid",
        }

    expired = is_expired(expires, time.time())
    if expired:
        message = f"{_LOGIN_COMMAND}; the saved gptpro session is expired"
    else:
        message = f"found a valid gptpro session at {path}"
    return {
        "path": path,
        "exists": True,
        "has_auth_cookie": True,
        "expired": expired,
        "valid": not expired,
        "message": message,
    }
