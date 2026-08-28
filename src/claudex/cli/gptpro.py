"""Implementation of the gptpro command family."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import importlib.util
import sys
import time
from collections.abc import Callable

from claudex import locking, paths
from claudex.gptpro import login as gptpro_login
from claudex.gptpro import runtime as gptpro_runtime
from claudex.gptpro import session as gptpro_session

_LOGIN_COMMAND = "run claudex-gateway gptpro login"


def _build_gptpro_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claudex-gateway gptpro")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("login")
    subparsers.add_parser("status")
    subparsers.add_parser("doctor")
    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    return parser


def _gptpro_login() -> int:
    result = asyncio.run(gptpro_login.run_login(on_status=print))
    if result.success:
        print(result.message)
        return 0

    failure = result.failure or "error"
    print(f"gptpro login failed [{failure}]: {result.message}", file=sys.stderr)
    if failure in {"login_timeout", "session_rejected"}:
        print(
            "run claudex-gateway gptpro login to create a fresh session",
            file=sys.stderr,
        )
    return 1


def _gptpro_status() -> int:
    status = gptpro_session.session_status()
    print(status["message"])
    return 0


def _format_remaining_time(seconds: float) -> str:
    total_minutes = max(0, int(seconds) // 60)
    days, remaining_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(remaining_minutes, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _check_gptpro_session() -> tuple[str, str]:
    status = gptpro_session.session_status()
    if not status["valid"]:
        return "FAIL", str(status["message"])

    expires = gptpro_session.load_auth_cookie_expiry(paths.gptpro_session_file())
    if expires is None or expires <= 0:
        expiry_description = "authentication cookie has no fixed expiry"
    else:
        remaining = _format_remaining_time(expires - time.time())
        expiry_description = f"authentication cookie expires in {remaining}"
    return "OK", f'{status["message"]}; {expiry_description}'


def _check_gptpro_chrome_profile() -> tuple[str, str]:
    profile_dir = paths.gptpro_chrome_profile_dir()
    if profile_dir.exists():
        return "OK", f"found at {profile_dir}"
    return (
        "WARN",
        f"not found at {profile_dir}; created by the first ask or login",
    )


def _check_gptpro_profile_lock() -> tuple[str, str]:
    handle = locking.try_file_lock(paths.gptpro_profile_lock())
    if handle is None:
        return (
            "OK",
            "held by another process "
            "(a running gateway daemon holds it while its runtime is active)",
        )
    handle.release()
    return "OK", "available"


def _check_gptpro_playwright() -> tuple[str, str]:
    if importlib.util.find_spec("playwright") is None:
        return "FAIL", "not installed; run uv sync --extra gptpro"

    try:
        version = importlib.metadata.version("playwright")
    except Exception:
        return "OK", "installed"
    return "OK", f"installed ({version})"


def _gptpro_doctor() -> int:
    checks: tuple[tuple[str, Callable[[], tuple[str, str]]], ...] = (
        ("Session", _check_gptpro_session),
        ("Chrome profile", _check_gptpro_chrome_profile),
        ("Profile lock", _check_gptpro_profile_lock),
        ("Playwright dependency", _check_gptpro_playwright),
    )
    passed = 0
    failed = 0
    warned = 0

    for name, check in checks:
        try:
            result, description = check()
        except Exception as exc:
            result = "FAIL"
            description = f"check raised {type(exc).__name__}"
        print(f"{name}: {result} - {description}")
        if result == "OK":
            passed += 1
        elif result == "WARN":
            warned += 1
        else:
            failed += 1

    print(f"Summary: {passed} passed, {failed} failed, {warned} warnings")
    return 1 if failed else 0


def _print_ask_status(message: str) -> None:
    print(message, file=sys.stderr)


async def _execute_runtime_ask(question: str) -> gptpro_runtime.AskOutcome:
    ask_runtime = gptpro_runtime.AskRuntime()
    primary_failure: BaseException | None = None
    try:
        return await ask_runtime.ask(
            question,
            callbacks=gptpro_runtime.AskCallbacks(on_status=_print_ask_status),
        )
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        try:
            await ask_runtime.aclose()
        except BaseException as exc:
            if primary_failure is not None:
                primary_failure.add_note(
                    f"gptpro runtime cleanup failed with {type(exc).__name__}"
                )
            else:
                raise


def _print_ask_failure(error: gptpro_runtime.GptProAskError) -> None:
    print(f"gptpro ask failed [{error.failure}]: {error}", file=sys.stderr)
    if error.failure == "session_expired":
        if _LOGIN_COMMAND not in str(error):
            print(_LOGIN_COMMAND, file=sys.stderr)
    elif error.failure == "challenge":
        print(
            "complete the ChatGPT browser challenge, then retry the ask",
            file=sys.stderr,
        )
    elif error.failure in {"timeout", "rate_limited_timeout", "echo_timeout"}:
        print(
            "the ask timed out; retry after checking ChatGPT and the network",
            file=sys.stderr,
        )


def _gptpro_ask(question: str) -> int:
    try:
        outcome = asyncio.run(_execute_runtime_ask(question))
    except gptpro_runtime.GptProAskError as exc:
        _print_ask_failure(exc)
        return 1

    print(outcome.text)
    return 0


def _gptpro_main(argv: list[str]) -> int:
    parser = _build_gptpro_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    try:
        if args.command == "login":
            return _gptpro_login()
        if args.command == "status":
            return _gptpro_status()
        if args.command == "doctor":
            return _gptpro_doctor()
        return _gptpro_ask(args.question)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"unexpected error: {type(exc).__name__}", file=sys.stderr)
        return 1
