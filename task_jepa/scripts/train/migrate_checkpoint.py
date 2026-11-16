"""Migrate legacy Torch ZIP/pickle-2 class paths without unpickling or changing tensors."""

import argparse
import hashlib
import os
import pickletools
import stat
import tempfile
from copy import copy
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from stable_worldmodel.paths import resolve_path


def _rewrite_globals(data: bytes) -> tuple[bytes, int]:
    if data[:2] != b"\x80\x02":
        raise ValueError("Expected a Torch checkpoint saved with pickle protocol 2")
    chunks = []
    cursor = count = 0
    for opcode, argument, position in pickletools.genops(data):
        if opcode.name != "GLOBAL" or not isinstance(argument, str):
            continue
        module, name = argument.split(" ", 1)
        if not module.startswith("models."):
            continue
        assert isinstance(position, int)
        original = f"c{module}\n{name}\n".encode()
        replacement = f"ch_jepa.{module}\n{name}\n".encode()
        chunks.extend((data[cursor:position], replacement))
        cursor = position + len(original)
        count += 1
    if not count:
        return data, 0
    chunks.append(data[cursor:])
    return b"".join(chunks), count


def _digest(archive: ZipFile, info: ZipInfo) -> bytes:
    digest = hashlib.sha256()
    with archive.open(info) as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.digest()


def _migrate(checkpoint: Path) -> None:
    backup = checkpoint.with_name(checkpoint.name + ".legacy")
    pending: Path | None = None
    original_stat = checkpoint.stat()
    try:
        with ZipFile(checkpoint) as source:
            members = source.infolist()
            pickle_names = [info.filename for info in members if info.filename.endswith("/data.pkl")]
            if len(pickle_names) != 1:
                raise ValueError("Expected exactly one data.pkl in a Torch ZIP checkpoint")
            pickle_name = pickle_names[0]
            updated, count = _rewrite_globals(source.read(pickle_name))
            if not count:
                print(f"Already migrated; unchanged: {checkpoint}")
                return
            if backup.exists():
                raise FileExistsError(f"Refusing to overwrite the original backup: {backup}")
            digests = {}
            with tempfile.NamedTemporaryFile(prefix=f".{checkpoint.name}.", suffix=".tmp",
                                             dir=checkpoint.parent, delete=False) as stream:
                pending = Path(stream.name)
                with ZipFile(stream, "w") as destination:
                    destination.comment = source.comment
                    for info in members:
                        output_info = copy(info)
                        if info.filename == pickle_name:
                            destination.writestr(output_info, updated)
                            continue
                        digest = hashlib.sha256()
                        with source.open(info) as original, destination.open(output_info, "w") as output:
                            while block := original.read(1024 * 1024):
                                digest.update(block)
                                output.write(block)
                        digests[info.filename] = digest.digest()
                stream.flush()
                os.fsync(stream.fileno())
        with ZipFile(pending) as migrated:
            if migrated.namelist() != [info.filename for info in members]:
                raise RuntimeError("Migration changed the archive member list")
            if migrated.read(pickle_name) != updated:
                raise RuntimeError("Migrated pickle verification failed")
            for name, expected in digests.items():
                if _digest(migrated, migrated.getinfo(name)) != expected:
                    raise RuntimeError(f"Migration changed non-pickle data: {name}")
        pending.chmod(stat.S_IMODE(checkpoint.stat().st_mode))
        current_stat = checkpoint.stat()
        if (current_stat.st_ino, current_stat.st_mtime_ns, current_stat.st_size) != (
            original_stat.st_ino, original_stat.st_mtime_ns, original_stat.st_size
        ):
            raise RuntimeError("Checkpoint changed during migration; refusing to replace it")
        # A sibling hard link retains the original inode without copying tensor data.
        # Atomic replacement changes only the active checkpoint, never the backup.
        os.link(checkpoint, backup)
        os.replace(pending, checkpoint)
        print(f"Migrated {count} class references: {checkpoint}")
        print(f"Original backup: {backup}")
        print(f"Verified {len(digests)} non-pickle members unchanged (including all tensor storage)")
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file; relative to the project root")
    args = parser.parse_args()
    _migrate(resolve_path(args.checkpoint))


if __name__ == "__main__":
    main()
