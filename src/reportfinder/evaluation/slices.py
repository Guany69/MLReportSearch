def group_by_slice(qrels):
    out = {}
    for qrel in qrels: out.setdefault(qrel.slice, []).append(qrel)
    return out
