"""Settings JSON file reading and updates."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schema import ConfigError, SETTINGS_KEYS


def read_settings_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read settings file {path}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"settings file {path} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"settings file {path} must contain a JSON object")
    group_names = {
        key.split(".", 1)[0] for key in SETTINGS_KEYS if "." in key
    }
    unknown = sorted(set(parsed) - set(SETTINGS_KEYS) - group_names)
    if unknown:
        raise ConfigError(
            f"settings file {path} has unknown keys: {', '.join(unknown)}; "
            f"valid keys: {', '.join(SETTINGS_KEYS)}"
        )

    settings = {
        key: value for key, value in parsed.items() if key in SETTINGS_KEYS
    }
    for group_name, group_value in parsed.items():
        if group_name in SETTINGS_KEYS:
            continue
        valid_group_keys = sorted(
            key for key in SETTINGS_KEYS if key.startswith(f"{group_name}.")
        )
        if not isinstance(group_value, dict):
            raise ConfigError(
                f'settings file {path} key "{group_name}" must contain a JSON object; '
                f"valid keys: {', '.join(valid_group_keys)}"
            )
        flattened = {
            f"{group_name}.{subkey}": value
            for subkey, value in group_value.items()
        }
        unknown_group_keys = sorted(set(flattened) - set(valid_group_keys))
        if unknown_group_keys:
            raise ConfigError(
                f"settings file {path} has unknown keys: "
                f"{', '.join(unknown_group_keys)}; "
                f"valid keys: {', '.join(valid_group_keys)}"
            )
        duplicated = sorted(set(flattened) & set(settings))
        if duplicated:
            raise ConfigError(
                f"settings file {path} has duplicate settings keys: "
                f"{', '.join(duplicated)}"
            )
        settings.update(flattened)
    return settings


def update_settings_file(
    path: Path,
    updates: dict[str, object],
    deletions: tuple[str, ...] = (),
) -> None:
    """Merge updates into the settings file, creating it if missing.

    The existing file is validated first so a corrupt or unknown-key file
    fails loudly instead of being silently overwritten, and the write is
    atomic so a crash cannot leave a half-written file behind.

    `deletions` removes keys from the settings dict before it is written, so
    a disabled feature is represented by the key's absence rather than a
    persisted JSON `null`. Deleting a key that is not present in the file is
    a no-op, not an error.
    """
    unknown = sorted(set(updates) - set(SETTINGS_KEYS))
    if unknown:
        raise ConfigError(f"cannot persist unknown settings keys: {', '.join(unknown)}")
    unknown_deletions = sorted(set(deletions) - set(SETTINGS_KEYS))
    if unknown_deletions:
        raise ConfigError(
            f"cannot delete unknown settings keys: {', '.join(unknown_deletions)}"
        )
    settings = read_settings_file(path)
    settings.update(updates)
    for key in deletions:
        settings.pop(key, None)

    rendered: dict[str, object] = {}
    for key, value in settings.items():
        group_name, separator, subkey = key.partition(".")
        if not separator:
            rendered[key] = value
            continue
        group = rendered.setdefault(group_name, {})
        if not isinstance(group, dict):
            raise ConfigError(
                f"cannot persist settings group {group_name!r} because it conflicts "
                "with a top-level settings key"
            )
        group[subkey] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    staging_file = path.with_name(path.name + ".tmp")
    staging_file.write_text(
        json.dumps(rendered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(staging_file, path)
