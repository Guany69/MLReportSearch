# TensorRT cross-encoder benchmark

This benchmark continues the latency investigation documented in
[`scripts/export_cross_encoder_onnx.py`](../scripts/export_cross_encoder_onnx.py): the
serving path can be dominated by the torch cross-encoder, and a faster runtime is useful
only if it preserves the shortlist ordering. The benchmark compiles the verified ONNX
artifact to TensorRT, compares TensorRT ordering with a torch FP32 reference, and measures
per-query torch, ONNX Runtime CUDA, and TensorRT latency on identical shortlists.

## Colab T4 runbook

Create a new Google Colab notebook, select **Runtime → Change runtime type → T4 GPU**, and
run this shell cell. It clones the repository, installs the GPU-only extra and ONNX Runtime
CUDA, verifies the ONNX export, runs the ordering gate and benchmark, saves the complete
console log, and commits the two measured artifacts locally in the Colab clone.

```bash
%%bash
set -euo pipefail
cd /content
git clone https://github.com/Guany69/MLReportSearch.git
cd MLReportSearch
pip install -e ".[trt]" onnxruntime-gpu
mkdir -p docs/benchmarks
{
  python scripts/export_cross_encoder_onnx.py --verify
  python scripts/benchmark_tensorrt.py
} 2>&1 | tee docs/benchmarks/tensorrt_t4.log
git add artifacts/evals/tensorrt_eval.json docs/benchmarks/tensorrt_t4.log
git commit -m "Record TensorRT GPU benchmark"
```

Push or otherwise retrieve the Colab commit after reviewing the ordering result. A failed
ordering gate returns nonzero and does not reach the commit command.

## Results

| Backend | p50 per query | p95 per query |
| --- | --- | --- |
| torch FP32 | &lt;MEASURED — pending GPU run&gt; | &lt;MEASURED — pending GPU run&gt; |
| ONNX Runtime CUDA | &lt;MEASURED — pending GPU run&gt; | &lt;MEASURED — pending GPU run&gt; |
| TensorRT | &lt;MEASURED — pending GPU run&gt; | &lt;MEASURED — pending GPU run&gt; |

| Ordering check | Result |
| --- | --- |
| Mean discordant rate | &lt;MEASURED — pending GPU run&gt; |
| Maximum discordant rate | &lt;MEASURED — pending GPU run&gt; |
| Top-1 flips | &lt;MEASURED — pending GPU run&gt; |
