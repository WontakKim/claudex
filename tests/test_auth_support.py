"""Tests for credential reads and private text replacement."""

import io
import os
import stat
from pathlib import Path

import pytest

from claudex.providers import auth_support


def test_replace_credentials_file_is_private_at_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path / "auth.json"
    path.write_text('{"token": "dummy-original"}', encoding="utf-8")
    path.chmod(0o600)
    tmp_path.chmod(0o755)
    text = '{"token": "dummy-replacement"}'
    created_inodes: list[tuple[int, int]] = []
    real_open = os.open
    real_text_open = io.open

    def record_creation(file_descriptor: int) -> None:
        file_stat = os.fstat(file_descriptor)
        created_inodes.append((stat.S_IMODE(file_stat.st_mode), file_stat.st_size))

    def recording_open(file, flags, mode=0o777, *args, **kwargs):
        file_descriptor = real_open(file, flags, mode, *args, **kwargs)
        if flags & os.O_CREAT and Path(file).parent == tmp_path:
            record_creation(file_descriptor)
        return file_descriptor

    def recording_text_open(file, mode="r", *args, **kwargs):
        handle = real_text_open(file, mode, *args, **kwargs)
        if "w" in mode and isinstance(file, (str, os.PathLike)) and Path(file).parent == tmp_path:
            record_creation(handle.fileno())
        return handle

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(io, "open", recording_text_open)
    previous_umask = os.umask(0o022)
    try:
        auth_support.replace_credentials_file(
            path, path.with_name("auth.json.tmp"), text, "utf-8"
        )
    finally:
        os.umask(previous_umask)

    assert created_inodes == [(0o600, 0)]
    assert path.read_text(encoding="utf-8") == text
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o755
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("encoding", [None, "utf-8", "utf-16"])
def test_replace_credentials_file_preserves_text_encoding(
    tmp_path: Path, existing: bool, encoding: str | None,
) -> None:
    path = tmp_path / "auth.json"
    if existing:
        path.write_text("dummy-original", encoding="utf-8")
        path.chmod(0o600)
    text = '{"token": "dummy-é"}\n'
    reference = tmp_path / "expected.txt"
    reference.write_text(text, encoding=encoding)

    auth_support.replace_credentials_file(path, path.with_suffix(".json.tmp"), text, encoding)

    assert path.read_bytes() == reference.read_bytes()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert set(tmp_path.iterdir()) == {path, reference}


@pytest.mark.parametrize("is_symlink", [False, True])
def test_replace_credentials_file_does_not_reuse_preexisting_staging(
    tmp_path: Path, is_symlink: bool,
) -> None:
    path = tmp_path / "auth.json"
    path.write_text("dummy-original", encoding="utf-8")
    staging_file = path.with_suffix(".json.tmp")
    old_contents = "dummy-preexisting-staging"
    if is_symlink:
        target = tmp_path / "unrelated.json"
        target.write_text(old_contents, encoding="utf-8")
        staging_file.symlink_to(target)
    else:
        staging_file.write_text(old_contents, encoding="utf-8")
    staging_file.chmod(0o644)
    original_inode = staging_file.stat().st_ino
    original_paths = set(tmp_path.iterdir())

    with staging_file.open("r", encoding="utf-8") as existing_reader:
        auth_support.replace_credentials_file(
            path, staging_file, "dummy-replacement", "utf-8"
        )
        assert existing_reader.read() == old_contents

    assert path.read_text(encoding="utf-8") == "dummy-replacement"
    assert staging_file.read_text(encoding="utf-8") == old_contents
    assert staging_file.stat().st_ino == original_inode
    assert stat.S_IMODE(staging_file.stat().st_mode) == 0o644
    assert staging_file.is_symlink() == is_symlink
    assert set(tmp_path.iterdir()) == original_paths


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["open", "chmod", "fdopen", "write", "replace"])
def test_replace_credentials_file_preserves_destination_and_cleans_owned_staging_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, existing: bool, failure: str,
) -> None:
    path = tmp_path / "auth.json"
    if existing:
        path.write_text("dummy-original", encoding="utf-8")
        path.chmod(0o600)
    staging_file = path.with_suffix(".json.tmp")
    staging_file.write_text("dummy-unrelated-staging", encoding="utf-8")
    original_paths = set(tmp_path.iterdir())
    descriptors: list[int] = []
    real_open = os.open

    def recording_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def reject(*args, **kwargs):
        raise OSError("dummy persistence failure")

    monkeypatch.setattr(os, "open", recording_open)
    if failure == "open":
        monkeypatch.setattr(os, "open", reject)
    elif failure == "chmod":
        monkeypatch.setattr(os, "fchmod" if hasattr(os, "fchmod") else "chmod", reject)
    elif failure == "fdopen":
        monkeypatch.setattr(os, "fdopen", reject)
    elif failure == "replace":
        monkeypatch.setattr(os, "replace", reject)
    text = "dummy-é" if failure == "write" else "dummy-replacement"
    error = UnicodeEncodeError if failure == "write" else OSError

    with pytest.raises(error):
        auth_support.replace_credentials_file(path, staging_file, text, "ascii")

    if existing:
        assert path.read_text(encoding="utf-8") == "dummy-original"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    else:
        assert not path.exists()
    assert staging_file.read_text(encoding="utf-8") == "dummy-unrelated-staging"
    assert set(tmp_path.iterdir()) == original_paths
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_replace_credentials_file_does_not_remove_an_exclusive_creation_collision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    path = tmp_path / "auth.json"
    path.write_text("dummy-original", encoding="utf-8")
    collision = tmp_path / ".auth.json.tmp-fixed"
    collision.write_text("dummy-unrelated-collision", encoding="utf-8")
    collision.chmod(0o644)
    monkeypatch.setattr(auth_support.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed"))

    with pytest.raises(FileExistsError):
        auth_support.replace_credentials_file(
            path, path.with_suffix(".json.tmp"), "dummy-replacement", "utf-8"
        )

    assert path.read_text(encoding="utf-8") == "dummy-original"
    assert collision.read_text(encoding="utf-8") == "dummy-unrelated-collision"
    assert stat.S_IMODE(collision.stat().st_mode) == 0o644
    assert set(tmp_path.iterdir()) == {path, collision}


def test_replace_credentials_file_supports_path_chmod_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path / "auth.json"
    real_chmod = os.chmod
    observations: list[tuple[int, int]] = []

    def recording_chmod(staging_file: Path, mode: int) -> None:
        file_stat = staging_file.stat()
        observations.append((stat.S_IMODE(file_stat.st_mode), file_stat.st_size))
        real_chmod(staging_file, mode)

    monkeypatch.delattr(os, "fchmod", raising=False)
    monkeypatch.setattr(os, "chmod", recording_chmod)
    previous_umask = os.umask(0o022)
    try:
        auth_support.replace_credentials_file(
            path, path.with_suffix(".json.tmp"), "dummy-replacement", "utf-8"
        )
    finally:
        os.umask(previous_umask)

    assert observations == [(0o600, 0)]
    assert path.read_text(encoding="utf-8") == "dummy-replacement"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("reuse_closed_descriptor", [False, True])
def test_replace_credentials_file_preserves_encoding_error_and_owns_descriptor_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reuse_closed_descriptor: bool,
) -> None:
    import errno

    path = tmp_path / "auth.json"
    path.write_text("dummy-original", encoding="utf-8")
    path.chmod(0o600)
    unrelated_file = tmp_path / "dummy-unrelated.txt"
    unrelated_file.write_text("dummy-unrelated", encoding="utf-8")
    unrelated_stat = unrelated_file.stat()
    created_descriptors: list[tuple[int, int, int]] = []
    cleanup_observations: list[tuple[int, int | None, int | None]] = []
    unrelated_descriptors: list[int] = []
    real_open = os.open
    real_fdopen = os.fdopen
    real_close = os.close

    def recording_open(file, flags, mode=0o777, *args, **kwargs):
        descriptor = real_open(file, flags, mode, *args, **kwargs)
        if flags & os.O_CREAT and Path(file).parent == tmp_path:
            file_stat = os.fstat(descriptor)
            created_descriptors.append((descriptor, file_stat.st_dev, file_stat.st_ino))
        return descriptor

    def recording_fdopen(*args, **kwargs):
        try:
            return real_fdopen(*args, **kwargs)
        except LookupError:
            if reuse_closed_descriptor:
                unrelated_descriptors.append(real_open(unrelated_file, os.O_RDONLY))
            raise

    def recording_close(descriptor: int) -> None:
        try:
            file_stat = os.fstat(descriptor)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise
            cleanup_observations.append((descriptor, None, None))
        else:
            cleanup_observations.append((descriptor, file_stat.st_dev, file_stat.st_ino))
        real_close(descriptor)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "fdopen", recording_fdopen)
    monkeypatch.setattr(os, "close", recording_close)
    try:
        with pytest.raises(LookupError, match="synthetic-nonexistent-encoding"):
            auth_support.replace_credentials_file(
                path, path.with_suffix(".json.tmp"), "dummy-replacement",
                "synthetic-nonexistent-encoding",
            )

        assert len(created_descriptors) == 1
        assert cleanup_observations == created_descriptors
        descriptor = created_descriptors[0][0]
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == errno.EBADF
        if reuse_closed_descriptor:
            unrelated_descriptor = unrelated_descriptors[0]
            file_stat = os.fstat(unrelated_descriptor)
            assert (file_stat.st_dev, file_stat.st_ino) == (
                unrelated_stat.st_dev, unrelated_stat.st_ino,
            )
            assert os.read(unrelated_descriptor, 100) == b"dummy-unrelated"
        assert path.read_text(encoding="utf-8") == "dummy-original"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert set(tmp_path.iterdir()) == {path, unrelated_file}
    finally:
        for unrelated_descriptor in unrelated_descriptors:
            try:
                file_stat = os.fstat(unrelated_descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
            else:
                assert (file_stat.st_dev, file_stat.st_ino) == (
                    unrelated_stat.st_dev, unrelated_stat.st_ino,
                )
                real_close(unrelated_descriptor)


@pytest.mark.parametrize("provider", ["codex", "kimi", "grok"])
@pytest.mark.parametrize("failure", ["permission", "encoding"])
def test_load_json_credentials_types_read_failures_without_exposing_contents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str, failure: str,
) -> None:
    from claudex.providers.codex_auth import CodexAuthError
    from claudex.providers.grok_auth import GrokAuthError
    from claudex.providers.kimi_auth import KimiAuthError

    error_cls = {"codex": CodexAuthError, "kimi": KimiAuthError, "grok": GrokAuthError}[provider]
    path = tmp_path / "auth.json"
    contents = b"\xffdummy-sensitive-content"
    path.write_bytes(contents)
    read_text = Path.read_text
    if failure == "permission":
        def denied_read(file: Path, *args, **kwargs) -> str:
            if file == path:
                raise PermissionError("dummy-sensitive-content")
            return read_text(file, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied_read)
    expected_cause = PermissionError if failure == "permission" else UnicodeDecodeError

    with pytest.raises(error_cls, match="cannot read credentials file") as raised:
        auth_support.load_json_credentials(
            path, error_cls=error_cls, missing_message="dummy login guidance", encoding="utf-8"
        )

    assert isinstance(raised.value.__cause__, expected_cause)
    assert str(path) in str(raised.value)
    assert "dummy-sensitive-content" not in str(raised.value)
    assert path.read_bytes() == contents


def test_load_json_credentials_preserves_missing_file_guidance(tmp_path: Path) -> None:
    from claudex.providers.codex_auth import CodexAuthError

    with pytest.raises(CodexAuthError) as raised:
        auth_support.load_json_credentials(
            tmp_path / "missing.json", error_cls=CodexAuthError,
            missing_message="dummy login guidance", encoding="utf-8",
        )

    assert str(raised.value) == "dummy login guidance"
    assert isinstance(raised.value.__cause__, FileNotFoundError)


def test_load_json_credentials_preserves_valid_json_objects(tmp_path: Path) -> None:
    from claudex.providers.codex_auth import CodexAuthError

    path = tmp_path / "auth.json"
    path.write_text('{"token": "dummy-token", "native": [1, true, null]}', encoding="utf-8")

    assert auth_support.load_json_credentials(
        path, error_cls=CodexAuthError, missing_message="dummy login guidance", encoding="utf-8"
    ) == {"token": "dummy-token", "native": [1, True, None]}


@pytest.mark.parametrize(("contents", "message"), [
    ("not-json", "is not valid JSON"),
    ("[]", "has an unexpected format"),
    ('"dummy-string"', "has an unexpected format"),
    ("null", "has an unexpected format"),
])
def test_load_json_credentials_preserves_invalid_json_contracts(
    tmp_path: Path, contents: str, message: str,
) -> None:
    from claudex.providers.codex_auth import CodexAuthError

    path = tmp_path / "auth.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(CodexAuthError, match=message):
        auth_support.load_json_credentials(
            path, error_cls=CodexAuthError, missing_message="dummy login guidance", encoding="utf-8"
        )
