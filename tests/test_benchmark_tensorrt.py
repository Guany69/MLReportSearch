from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from scripts.benchmark_tensorrt import make_result, parse_args, require_gpu_environment


def test_argument_defaults_and_boolean_optional() -> None:
    defaults = parse_args([])
    assert defaults.queries == 40
    assert defaults.shortlist == 12
    assert defaults.fp16 is True
    assert defaults.output == Path("artifacts/evals/tensorrt_eval.json")
    assert parse_args(["--no-fp16"]).fp16 is False


def test_mocked_result_schema(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.fp16.engine"
    onnx_path.write_bytes(b"onnx")
    engine_path.write_bytes(b"engine")
    args = argparse.Namespace(
        queries=1, shortlist=2, fp16=True, max_discordant=0.02
    )
    unavailable = float("nan")
    result = make_result(
        args=args, config={"path": None}, onnx_path=onnx_path, engine_path=engine_path,
        versions={"torch": "mock", "onnxruntime": "mock", "tensorrt": "mock"},
        gpu_name="mock GPU",
        latencies={
            "torch": [unavailable], "onnxruntime": [unavailable],
            "tensorrt": [unavailable],
        },
        discordant_rates=[unavailable], top1_flips=int(False),
    )
    assert result["schema_version"] == "1"
    assert len(result["artifact"]["onnx_sha256"]) == 64
    assert result["dataset"]["label_source"] == "torch_fp32_cross_encoder_scores"
    assert set(result["metrics"]) == {"latency_ms_per_query", "agreement"}
    assert result["workload"]["queries"] == 1


def test_environment_guard_names_runbook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="docs/tensorrt_benchmark.md"):
        require_gpu_environment()
