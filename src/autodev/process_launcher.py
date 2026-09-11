"""A tiny owned parent process for managed application trees.

Its UUID stays in the parent command line, allowing crash recovery to prove
ownership before taskkill /T touches a persisted PID.
"""

from __future__ import annotations

import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3 or args[1] != "--":
        return 2
    child = subprocess.Popen(args[2:])
    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
