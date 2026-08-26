"""Implementation of the gptpro setup command family."""

from __future__ import annotations

import argparse
import asyncio
import sys

from claudex.gptpro import login as gptpro_login
from claudex.gptpro import session as gptpro_session


def _build_gptpro_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claudex-gateway gptpro")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("login")
    subparsers.add_parser("status")
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


def _gptpro_main(argv: list[str]) -> int:
    parser = _build_gptpro_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    try:
        if args.command == "login":
            return _gptpro_login()
        return _gptpro_status()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"unexpected error: {type(exc).__name__}", file=sys.stderr)
        return 1
