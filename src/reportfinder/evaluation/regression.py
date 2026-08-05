def assert_no_regression(baseline, improved, tolerances=None):
    tolerances = tolerances or {"hit@1": .02, "mrr@10": .02, "ndcg@10": .02}
    drops = {m: baseline[m]-improved[m] for m in tolerances if baseline.get(m, 0)-improved.get(m, 0) > tolerances[m]}
    if drops: raise AssertionError(f"Retrieval regression exceeds tolerance: {drops}")
