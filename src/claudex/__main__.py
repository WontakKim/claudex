"""Entrypoint for `claudex-gateway` / `python -m claudex`."""

from __future__ import annotations

import sys

from claudex.cli import accounts, compact, daemon, gptpro


def main() -> None:
    arguments = sys.argv[1:]
    # stop must work even when the current configuration is broken.
    if arguments == ["stop"]:
        daemon._stop_background()
        return

    # Account management only touches ~/.claudex account files, so it must
    # work even when the current gateway configuration is broken.
    if arguments and arguments[0] == "account":
        result = accounts._account_main(arguments[1:])
        if result != 0:
            raise SystemExit(result)
        return

    # compact validates its own argument (compact set claude:<id>) before
    # touching configuration, a daemon record, the network, or the settings
    # file, so it is dispatched ahead of the ordinary _load_config() below.
    if arguments and arguments[0] == "compact":
        result = compact._compact_main(arguments[1:])
        if result != 0:
            raise SystemExit(result)
        return

    # gptpro setup only touches its browser profile and session file, so it
    # must remain available when the gateway configuration is broken.
    if arguments and arguments[0] == "gptpro":
        result = gptpro._gptpro_main(arguments[1:])
        if result != 0:
            raise SystemExit(result)
        return

    config = daemon._load_config()
    if not arguments:
        daemon._start_background(config)
        return
    if arguments == ["--foreground"]:
        daemon._run_foreground(config)
        return

    print("usage: claudex-gateway [--foreground|stop]", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
