from reportfinder.evaluation.production import Judgment, acceptance_gate, agreement, query_split


def judgment(q, report, rel, annotator):
    return Judgment(q,q,report,rel,True,annotator,"2026-07-17T00:00:00Z")

def test_query_split_is_stable_and_query_level():
    assert query_split("q1") == query_split("q1")
    assert query_split("q1",seed="x") in {"train","validation","test"}

def test_agreement_uses_independent_double_labels():
    items=[judgment("q1","r1",2,"a"),judgment("q1","r1",2,"b"),judgment("q2","r2",0,"a"),judgment("q2","r2",1,"b")]
    assert agreement(items) == .5

def test_gate_refuses_small_or_weak_evidence():
    ok,reasons=acceptance_gate({"ndcg@10":1,"mrr@10":1,"no_answer_precision":1,"no_answer_recall":1,"ece":0},[judgment("q1","r1",2,"a"),judgment("q1","r1",2,"b")])
    assert not ok and any("500 human queries" in reason for reason in reasons)
