from reportfinder.config import DEFAULT
from reportfinder.ingest import build_corpus
from reportfinder.query.expansion.engine import ExpansionEngine

from .conftest import requires_real_estate


@requires_real_estate
def test_every_lexicon_emission_resolves_against_the_real_estate():
    """No lexicon rule may emit a canonical the corpus does not contain.

    The lexicon lives in YAML and the workbook is only the resolution target, so
    this lints equally well against the legacy estate -- and does: all 84
    emissions resolve against `Reports.xlsx`. A dead rule is a phrase users can
    type that expands to nothing.
    """
    cfg = DEFAULT.with_overrides(ingest_mode="legacy_single_file")
    corpus, _ = build_corpus(cfg, verbose=False)
    engine = ExpansionEngine(corpus.frame)
    assert engine.lexicon.dead_rules == ()
