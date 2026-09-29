"""Capture the installed dependency closure for the current Python/platform."""
import argparse
import importlib.metadata as metadata
import platform
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def capture(files, output):
    pending = []
    for file in files:
        for line in Path(file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                pending.append(Requirement(line))
    seen, pinned = set(), {}
    while pending:
        req = pending.pop()
        if req.marker and not req.marker.evaluate():
            continue
        key = (canonicalize_name(req.name), tuple(sorted(req.extras)))
        if key in seen:
            continue
        seen.add(key)
        distribution = metadata.distribution(req.name)
        if req.specifier and not req.specifier.contains(distribution.version, prereleases=True):
            raise ValueError(f"installed {req.name} does not satisfy {req.specifier}")
        pinned[key[0]] = distribution.version
        for raw in distribution.requires or []:
            child = Requirement(raw)
            if child.marker and not any(child.marker.evaluate({"extra": extra}) for extra in ({""} | set(req.extras))):
                continue
            child.marker = None
            pending.append(child)
    header = f"# Installed closure: Python {platform.python_version()}, {platform.system()} {platform.machine()}\n# Platform-specific constraints, not a cross-platform wheel/hash lock.\n"
    Path(output).write_text(header + "".join(f"{name}=={version}\n" for name, version in sorted(pinned.items())), encoding="utf-8")
    return len(pinned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Pinned {capture(['requirements.txt', 'requirements-dev.txt'], args.output)} distributions")
