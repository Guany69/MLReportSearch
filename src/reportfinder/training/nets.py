"""The two PyTorch models, and strict artifact loading.

Deliberately small. Fusion combines a few dozen bounded features over a shortlist
of at most 200 candidates, and the decision head reads a handful of summary
statistics; neither problem has the data to support anything larger. With no human
judgements yet, a bigger model would mostly memorise the synthetic label generator.

Loading is strict on purpose. `ranking/ltr.py` already established the pattern:
an artifact whose feature schema no longer matches the serving code produces
plausible-looking scores computed from the wrong columns, which is far worse than
a refusal. Both loaders check the feature hash and refuse on mismatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

FUSION_ARCHITECTURE = "pytorch_mlp_64_32_1"
DECISION_ARCHITECTURE = "pytorch_mlp_32_16_3"
ARTIFACT_SCHEMA_VERSION = "1"


class FusionMLP(nn.Module):
    """input -> Linear(in, 64) -> act -> dropout -> Linear(64, 32) -> act -> Linear(32, 1)."""

    def __init__(self, input_dim: int, hidden_1: int = 64, hidden_2: int = 32,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # One unnormalized score per candidate. Not a probability: turning these
        # into one with a softmax would produce a confident-looking number that
        # nothing has calibrated.
        return self.net(features).squeeze(-1)


class DecisionHead(nn.Module):
    """input -> Linear(in, 32) -> act -> Linear(32, 16) -> act -> Linear(16, 3)."""

    CLASSES = ("RETURN_RESULTS", "ASK_CLARIFICATION", "NO_CONFIDENT_MATCH")

    def __init__(self, input_dim: int, hidden_1: int = 32, hidden_2: int = 16) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Logits. Calibration happens outside, on a held-out split."""
        return self.net(features)


def weights_sha256(state_dict) -> str:
    """A stable digest of the weights alone.

    Deliberately not a hash of the file: approving an artifact rewrites its
    metadata, which would change a file digest and break every reference to it.
    The weights are what an evaluation report actually describes, so they are what
    is hashed -- an eval report and the model it measured stay bound together
    across restamping.
    """
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x1f")
        tensor = state_dict[name].detach().to(torch.float64).cpu().contiguous()
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\x1e")
    return digest.hexdigest()[:32]


def save_model(model: nn.Module, path: Path, metadata: dict) -> None:
    """Write weights and metadata together.

    The metadata is not optional decoration: without the feature hash and the
    provenance of the training labels, a future reader cannot tell whether the
    artifact is safe to serve or what its numbers mean.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    metadata = {**metadata, "weights_sha256": weights_sha256(state_dict)}
    torch.save(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "state_dict": state_dict,
            "input_dim": model.input_dim,
            "metadata": metadata,
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps({"input_dim": model.input_dim, **metadata}, indent=2, sort_keys=True)
        + "\n"
    )


def _load(path: Path, builder, expected_hash: str, kind: str):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"{kind} artifact at {path} has schema version "
            f"{payload.get('schema_version')!r}; expected {ARTIFACT_SCHEMA_VERSION!r}"
        )
    metadata = payload.get("metadata", {})
    actual = metadata.get("feature_hash")
    if actual != expected_hash:
        raise ValueError(
            f"{kind} artifact at {path} was trained against feature space "
            f"{actual!r}, but this build computes {expected_hash!r}. Refusing to "
            "serve it: the scores would be computed from the wrong columns."
        )
    if not metadata.get("approved", False):
        raise ValueError(
            f"{kind} artifact at {path} is not marked approved "
            f"(training_label_source={metadata.get('training_label_source')!r}). "
            "Earn it with `reportfinder-train approve <artifact> --evaluation "
            "<report.json>`, which requires the model to beat its fallback on a "
            "split it was not fitted on."
        )
    # An approval must say what it was earned on. Without this, `approved: true`
    # is one hand-edit away from meaning nothing, which is what it used to be.
    if not (metadata.get("approval") or {}).get("basis"):
        raise ValueError(
            f"{kind} artifact at {path} is marked approved but carries no "
            "approval basis. Approval is granted by `reportfinder-train approve`, "
            "which records the evaluation it was earned on; a bare flag is not "
            "evidence and will not be served."
        )
    model = builder(payload["input_dim"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


def load_fusion_model(path: Path):
    from ..pipeline.fuse import FUSION_FEATURE_HASH

    return _load(path, FusionMLP, FUSION_FEATURE_HASH, "fusion")


def load_decision_model(path: Path):
    from .features import DECISION_FEATURE_HASH

    return _load(path, DecisionHead, DECISION_FEATURE_HASH, "decision")
