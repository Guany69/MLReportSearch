"""`reportfinder-eval` -- run the reproducible benchmark.

This module was empty while `pyproject.toml` advertised it as a console script, so
`reportfinder-eval` failed with an ImportError on every invocation. It is a thin
alias rather than a second implementation: the benchmark runner already owns
argument parsing, config resolution and report writing, and duplicating any of
that would create a second place for the two to disagree.
"""

from __future__ import annotations

from ..evaluation.benchmark import main as _benchmark_main


def main() -> int:
    return _benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
