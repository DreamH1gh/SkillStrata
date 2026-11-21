from __future__ import annotations

import math


def evaluate(scores, targets: list[str], candidate_index: dict[str, int], top_k: list[int]):
    hits = {k: 0 for k in top_k}
    reciprocal_rank = 0.0
    ndcg = {k: 0.0 for k in top_k}
    for row, target in enumerate(targets):
        target_score = scores[row, candidate_index[target]]
        rank = int((scores[row] > target_score).sum()) + 1
        reciprocal_rank += 1.0 / rank
        for k in top_k:
            if rank <= k:
                hits[k] += 1
                ndcg[k] += 1.0 / math.log2(rank + 1)
    count = len(targets)
    return {
        **{f"Hit@{k}": round(hits[k] / count, 4) for k in top_k},
        **{f"nDCG@{k}": round(ndcg[k] / count, 4) for k in top_k},
        "MRR": round(reciprocal_rank / count, 4),
        "n": count,
    }


def evaluate_subset(scores, indices, targets, candidate_index, top_k):
    if not indices:
        return None
    return evaluate(
        scores[indices],
        [targets[index] for index in indices],
        candidate_index,
        top_k,
    )
