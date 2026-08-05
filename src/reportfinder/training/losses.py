"""Losses for listwise fusion and the three-class decision.

Listwise rather than pointwise because the task is to *order* the shortlist for one
query, not to predict an absolute relevance value. A pointwise loss would spend most
of its capacity learning the base rate of irrelevance, which is ~99% here and tells
us nothing useful.

Multi-positive is the case that matters. A vague request legitimately has several
right answers -- "why are we losing people" is answered by any of a dozen attrition
reports -- so a loss that assumes exactly one positive would actively punish the
model for ranking the others highly. Both losses below accept a graded target
distribution instead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def listnet_loss(
    scores: torch.Tensor,
    grades: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy between the predicted and target top-1 distributions.

    The target is built from graded relevance, so several documents can share the
    probability mass. Padding is masked to -inf before the softmax so padded slots
    contribute nothing rather than competing for it.
    """
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        grades = grades.masked_fill(~mask, float("-inf"))

    target = F.softmax(grades, dim=-1)
    predicted = F.log_softmax(scores, dim=-1)
    if mask is not None:
        # The target is already 0 at padded slots, but `predicted` is -inf there
        # and 0 * -inf is NaN, not 0. Zeroing the log-probabilities first keeps
        # the padded terms out of the sum without poisoning it.
        predicted = predicted.masked_fill(~mask, 0.0)
    per_query = -(target * predicted).sum(dim=-1)
    return per_query.mean()


def multi_positive_infonce(
    query_vectors: torch.Tensor,
    document_vectors: torch.Tensor,
    positives: torch.Tensor,
    *,
    temperature: float = 0.05,
    co_positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE where a query may have several correct documents.

    `positives` is a boolean (queries x documents) matrix. A query with no positive
    is an error rather than something to skip quietly -- silently dropping it would
    shrink the training set in a way nobody notices.

    `co_positive_mask` marks pairs that must not be treated as negatives: other
    valid answers for the same query, duplicate instances of the same report, and
    anything whose relevance is simply unknown. Leaving them in the denominator
    teaches the model to push apart documents that are both correct.
    """
    if not bool(positives.any(dim=-1).all()):
        empty = (~positives.any(dim=-1)).nonzero().flatten().tolist()
        raise ValueError(
            f"queries {empty[:10]} have no positive document. A query with no "
            "valid positive cannot contribute a contrastive signal; fix the data "
            "rather than dropping it silently."
        )

    logits = (query_vectors @ document_vectors.T) / temperature
    if co_positive_mask is not None:
        # Excluded from the denominator entirely: not a negative, not a positive.
        logits = logits.masked_fill(co_positive_mask & ~positives, float("-inf"))

    log_probabilities = F.log_softmax(logits, dim=-1)
    # Average the log-probability over each query's positives, so a query with
    # five right answers is not weighted five times more than one with a single
    # right answer.
    positive_log = (log_probabilities * positives).sum(dim=-1) / positives.sum(dim=-1)
    return -positive_log.mean()


def class_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Three-class loss with class weights.

    The decision labels available here are heavily imbalanced (8350 / 950 / 700),
    so an unweighted loss reaches 83% accuracy by always answering
    RETURN_RESULTS -- and never learns to abstain, which is the behaviour that
    actually matters.
    """
    return F.cross_entropy(logits, labels, weight=weights)


def balanced_class_weights(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights
