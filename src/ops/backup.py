"""Offline, checksummed directory snapshots. Restore only into a new directory."""
import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def create(roots, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("backup destination must not exist")
    for name, raw in roots.items():
        if not name or Path(name).name != name or name in {".", "..", "manifest.json"} or ":" in name:
            raise ValueError("invalid root name")
        root = Path(raw).resolve()
        if not root.is_dir() or destination.is_relative_to(root):
            raise ValueError("source must exist and destination must be outside source")
    destination.mkdir(parents=True)
    files = []
    for name, raw in roots.items():
        root = Path(raw).resolve()
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("symlinks are not supported")
            if not path.is_file() or path.name.endswith(("-wal", "-shm")):
                continue
            target = destination / name / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as stream:
                is_sqlite = stream.read(16) == b"SQLite format 3\x00"
            if is_sqlite:
                with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as source:
                    with sqlite3.connect(target) as out:
                        source.backup(out)
                        if out.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                            raise ValueError("SQLite integrity check failed")
            else:
                shutil.copy2(path, target)
            files.append({"path": target.relative_to(destination).as_posix(), "sha256": digest(target), "bytes": target.stat().st_size})
    manifest = {"version": 1, "requires_offline": True, "roots": sorted(roots), "files": files}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify(snapshot):
    snapshot = Path(snapshot).resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("unsupported manifest version")
    seen = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        path = (snapshot / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(snapshot) or path == snapshot or item["path"] in seen:
            raise ValueError("unsafe or duplicate manifest path")
        seen.add(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError("snapshot checksum mismatch: " + item["path"])
    return manifest


def restore(snapshot, destination):
    manifest = verify(snapshot)
    snapshot, destination = Path(snapshot).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("restore destination must not exist")
    destination.mkdir(parents=True)
    for item in manifest["files"]:
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / item["path"], target)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["create", "verify", "restore"])
    parser.add_argument("snapshot")
    parser.add_argument("--root", action="append", default=[], help="logical-name=directory; include every configured storage root")
    parser.add_argument("--destination")
    parser.add_argument("--offline", action="store_true", help="confirm all application workers and writers are stopped")
    args = parser.parse_args()
    if args.operation == "create":
        if not args.offline or not args.root:
            parser.error("create requires --offline and at least one --root")
        roots = dict(value.split("=", 1) for value in args.root)
        result = create(roots, args.snapshot)
    elif args.operation == "restore":
        if not args.offline or not args.destination:
            parser.error("restore requires --offline and --destination")
        result = restore(args.snapshot, args.destination)
    else:
        result = verify(args.snapshot)
    print(json.dumps({"ok": True, "files": len(result["files"])}))


if __name__ == "__main__":
    main()
