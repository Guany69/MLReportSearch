"""Training for the fusion model and the decision head.

Both trainers run. Neither ships an artifact: every label in this repository is
synthetic, so an artifact trained here demonstrates that the path executes, not
that the model is good. Artifacts are written with `approved: false` and the
loaders refuse to serve them until that is set deliberately.
"""

from __future__ import annotations

from .features import (
    DECISION_FEATURE_HASH,
    DECISION_FEATURE_NAMES,
    decision_features,
)
from .losses import (
    balanced_class_weights,
    class_weighted_cross_entropy,
    listnet_loss,
    multi_positive_infonce,
)
from .nets import (
    DECISION_ARCHITECTURE,
    FUSION_ARCHITECTURE,
    DecisionHead,
    FusionMLP,
    load_decision_model,
    load_fusion_model,
    save_model,
)

__all__ = [
    "DECISION_ARCHITECTURE",
    "DECISION_FEATURE_HASH",
    "DECISION_FEATURE_NAMES",
    "FUSION_ARCHITECTURE",
    "DecisionHead",
    "FusionMLP",
    "balanced_class_weights",
    "class_weighted_cross_entropy",
    "decision_features",
    "listnet_loss",
    "load_decision_model",
    "load_fusion_model",
    "multi_positive_infonce",
    "save_model",
]
