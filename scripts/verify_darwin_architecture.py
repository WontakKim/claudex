"""Verify Mach-O arm64 support for NUL-delimited build paths on stdin."""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from typing import BinaryIO

_CPU_TYPE_ARM64 = 0x0100000C
_THIN_BYTE_ORDERS = {b"\xcf\xfa\xed\xfe": "<", b"\xfe\xed\xfa\xcf": ">"}
_FAT_FORMATS = {
    b"\xca\xfe\xba\xbe": (">", "5I"),
    b"\xbe\xba\xfe\xca": ("<", "5I"),
    b"\xca\xfe\xba\xbf": (">", "IIQQII"),
    b"\xbf\xba\xfe\xca": ("<", "IIQQII"),
}


def _verify_thin_arm64(binary: BinaryIO, offset: int, size: int) -> None:
    if size < 32:
        raise ValueError("truncated Mach-O header")
    binary.seek(offset)
    header = binary.read(32)
    byte_order = _THIN_BYTE_ORDERS.get(header[:4])
    if len(header) != 32 or byte_order is None:
        raise ValueError("not a 64-bit Mach-O binary")
    _, cpu_type, _, _, _, load_commands_size, _, _ = struct.unpack(
        f"{byte_order}8I", header
    )
    if cpu_type != _CPU_TYPE_ARM64:
        raise ValueError("Mach-O binary lacks arm64 support")
    if load_commands_size > size - 32:
        raise ValueError("Mach-O load commands exceed the slice bounds")


def verify_arm64(path: Path) -> None:
    with path.open("rb") as binary:
        size = os.fstat(binary.fileno()).st_size
        magic = binary.read(4)
        if magic in _THIN_BYTE_ORDERS:
            _verify_thin_arm64(binary, 0, size)
            return
        if magic not in _FAT_FORMATS:
            raise ValueError("not a Mach-O binary")
        if size < 8:
            raise ValueError("truncated fat Mach-O header")
        byte_order, record_format = _FAT_FORMATS[magic]
        record_format = byte_order + record_format
        record_size = struct.calcsize(record_format)
        architecture_count = struct.unpack(f"{byte_order}I", binary.read(4))[0]
        if architecture_count > (size - 8) // record_size:
            raise ValueError("truncated fat Mach-O architecture table")
        table_end = 8 + architecture_count * record_size
        has_arm64 = False
        for index in range(architecture_count):
            binary.seek(8 + index * record_size)
            cpu_type, _, offset, slice_size, *_ = struct.unpack(
                record_format, binary.read(record_size)
            )
            if offset < table_end or slice_size == 0 or offset + slice_size > size:
                raise ValueError("invalid fat Mach-O slice bounds")
            if cpu_type == _CPU_TYPE_ARM64:
                _verify_thin_arm64(binary, offset, slice_size)
                has_arm64 = True
        if not has_arm64:
            raise ValueError("fat Mach-O binary lacks arm64 support")


def main() -> int:
    paths = sys.stdin.buffer.read().split(b"\0")
    if paths[-1] != b"" or len(paths) == 1:
        print("error: expected NUL-delimited native file paths", file=sys.stderr)
        return 1
    for raw_path in paths[:-1]:
        path = Path(os.fsdecode(raw_path))
        try:
            verify_arm64(path)
        except (OSError, ValueError) as error:
            print(
                f"error: native architecture verification failed: {path}: {error}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
