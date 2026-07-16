"""reportfinder: unsupervised report retrieval by product of experts."""

from .config import DEFAULT, Config
from .model import Candidate, ReportFinder, Result, why_matched
from .represent import Representation, load_or_build

__all__ = [
    "Config",
    "DEFAULT",
    "ReportFinder",
    "Result",
    "Candidate",
    "Representation",
    "load_or_build",
    "why_matched",
    "find",
]

__version__ = "0.1.0"


def find(query: str, cfg: Config | None = None, rebuild: bool = False) -> Result:
    """Convenience one-liner: build/load the index and answer a single query."""
    cfg = cfg or DEFAULT
    rep = load_or_build(cfg, rebuild=rebuild, verbose=False)
    return ReportFinder(rep, cfg).query(query)
