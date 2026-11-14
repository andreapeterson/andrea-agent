"""Print the portable evaluation manifest or check an existing canonical JSON file."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from evals.portable_assets import (
    build_portable_eval_manifest,
    format_portable_eval_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Compare against an existing manifest without rewriting it")
    args = parser.parse_args(argv)
    try:
        current = format_portable_eval_manifest(build_portable_eval_manifest())
    except (OSError, RuntimeError, TypeError, ValueError):
        # Existing loader errors may contain controlled case text or local file paths.
        print("Unable to build the portable evaluation manifest.", file=sys.stderr)
        return 1
    if args.check is None:
        print(current, end="")
        return 0
    try:
        committed = args.check.read_bytes()
    except OSError:
        print("Unable to read the portable evaluation manifest for checking.", file=sys.stderr)
        return 1
    if committed != current.encode("utf-8"):
        print("Portable evaluation manifest is stale.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
