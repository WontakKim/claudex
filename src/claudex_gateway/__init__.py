"""Local gateway that runs Claude Code models on the Codex backend."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("claudex-gateway")
except importlib.metadata.PackageNotFoundError:  # running from a bare source tree
    __version__ = "unknown"
