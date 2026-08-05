"""Serializable learned ranker with a deterministic fallback."""

from __future__ import annotations

import logging

import joblib
import numpy as np

from .features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION, linear_score

logger = logging.getLogger(__name__)

class Ranker:
    def __init__(self, model=None, metadata=None, fallback_reason=None):
        self.model, self.metadata, self.fallback_reason = model, metadata or {}, fallback_reason
    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        if payload.get("schema_version", payload.get("schema")) != FEATURE_SCHEMA_VERSION or tuple(payload.get("feature_names", payload.get("features", ()))) != FEATURE_NAMES or payload.get("feature_hash") != FEATURE_HASH:
            raise ValueError("LTR artifact feature schema is incompatible")
        if payload.get("model_type") not in {"lightgbm_lambdamart", "hist_gradient_boosting"}:
            raise ValueError("LTR artifact model type is unsupported")
        return cls(payload["model"], payload)
    @classmethod
    def load_or_fallback(cls, path):
        try: return cls.load(path)
        except Exception as exc:
            logger.warning("ltr_fallback", extra={"fallback_reason": str(exc), "artifact": str(path)})
            return cls(fallback_reason=str(exc))
    def score(self, features):
        value = float(self.model.predict([features.vector()])[0]) if self.model is not None else linear_score(features)
        if not np.isfinite(value): raise ValueError("LTR model returned a non-finite prediction")
        return value
