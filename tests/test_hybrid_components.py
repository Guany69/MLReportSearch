from __future__ import annotations

import pandas as pd

from reportfinder.confidence.answerability import AnswerabilityOutcome, decide_answerability
from reportfinder.query.intent import IntentParser
from reportfinder.ranking.features import FEATURE_NAMES, extract_features
from reportfinder.retrieval import (
    BM25FIndex,
    CharNgramIndex,
    FieldIndex,
    reciprocal_rank_fusion,
)


def frame():
    return pd.DataFrame([
        {"title":"Worker Terminations", "description":"departed employees", "fields":["Termination Date","Manager"], "prompts":[], "category":"Worker", "data_source":"Workers", "tags":"", "field_meta":[], "runs":10, "last_updated_missing":False, "family_size":1},
        {"title":"Compensation", "description":"salary", "fields":["Base Pay"], "prompts":[], "category":"Pay", "data_source":"Compensation", "tags":"", "field_meta":[], "runs":2, "last_updated_missing":True, "family_size":1},
    ])

def test_bm25f_and_field_index_retrieve_structured_evidence():
    data=frame(); bm=BM25FIndex(data,{"title":4,"description":1,"fields":3})
    assert bm.search("termination date")[0][0] == 0
    scores, detected=FieldIndex(data).scores("show termination date by manager")
    assert detected == ["termination date", "manager"] and scores[0] > scores[1]

def test_char_index_handles_typo_and_rrf_deduplicates():
    char=CharNgramIndex(["termination date manager", "base compensation pay"])
    assert char.search("terminaton manger")[0][0] == 0
    fused=reciprocal_rank_fusion({"a":[(0,1),(1,.5)],"b":[(0,.8)]})
    assert set(fused)=={0,1} and fused[0] > fused[1]

def test_intent_and_features_keep_required_fields_auditable():
    data=frame(); intent=IntentParser(data).parse("need Termination Date and Manager")
    assert set(intent.mandatory_fields)=={"Termination Date","Manager"}
    features=extract_features(data.iloc[0], intent, {"rrf":.1})
    assert tuple(features.values) != () and len(features.vector()) == len(FEATURE_NAMES)
    assert features.values["mandatory_coverage"] == 1

def test_answerability_has_all_three_outcomes():
    high=decide_answerability(top_score=1,margin=.3,field_coverage=1,retriever_agreement=1,parser_confidence=1,unsupported=[])
    multiple=decide_answerability(top_score=.6,margin=.01,field_coverage=1,retriever_agreement=.5,parser_confidence=.5,unsupported=[])
    insufficient=decide_answerability(top_score=.9,margin=.4,field_coverage=0,retriever_agreement=1,parser_confidence=1,unsupported=["Missing"])
    assert high.outcome is AnswerabilityOutcome.HIGH_CONFIDENCE
    assert multiple.outcome is AnswerabilityOutcome.MULTIPLE_PLAUSIBLE_RESULTS
    assert insufficient.outcome is AnswerabilityOutcome.INSUFFICIENT_MATCH
    assert not high.calibrated
