"""Implementation of the gptpro command family."""

from __future__ import annotations

import argparse
import asyncio
import sys

from claudex.gptpro import login as gptpro_login
from claudex.gptpro import runtime as gptpro_runtime
from claudex.gptpro import session as gptpro_session

_LOGIN_COMMAND = "run claudex-gateway gptpro login"


def _build_gptpro_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claudex-gateway gptpro")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("login")
    subparsers.add_parser("status")
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


def _print_ask_status(message: str) -> None:
    print(message, file=sys.stderr)


async def _execute_runtime_ask(question: str) -> str:
    ask_runtime = gptpro_runtime.AskRuntime()
    primary_failure: BaseException | None = None
    try:
        return await ask_runtime.ask(question, on_status=_print_ask_status)
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
        answer = asyncio.run(_execute_runtime_ask(question))
    except gptpro_runtime.GptProAskError as exc:
        _print_ask_failure(exc)
        return 1

    print(answer)
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
        return _gptpro_ask(args.question)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"unexpected error: {type(exc).__name__}", file=sys.stderr)
        return 1
