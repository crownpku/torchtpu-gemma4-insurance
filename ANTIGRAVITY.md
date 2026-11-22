# ANTIGRAVITY.md

## Project Context
- **Repository**: [https://github.com/crownpku/torchtpu-gemma4-insurance](https://github.com/crownpku/torchtpu-gemma4-insurance) (`TorchTPU_sprint`)
- **Predecessor Baseline**: [`crownpku/tunix-gemma4-tpu`](https://github.com/crownpku/tunix-gemma4-tpu) (`Tunix_Gemma4_TPU` JAX/Flax/Tunix implementation)
- **Stack**: Native PyTorch (`torch`) + `torch_tpu` (TorchTPU) + Hugging Face `transformers >= 5.0` (`Gemma4ForCausalLM`) + `peft` (LoRA) + Python 3.12 (managed via `uv`)
- **Model**: `google/gemma-4-E2B-it` (5.1B total parameters including the ~4.7 GB Per-Layer Embedding table; 2.3B active compute parameters; 128k context window)
- **Dataset**: `deccan-ai/insuranceQA-v2` on Hugging Face (`21.3k` train, `3.35k` validation, `3.31k` test prompt-response pairs)
- **Deployment & Hardware**: Local development (macOS CPU dry-run) + remote GCP Cloud TPU VM (`v6e-1` Trillium 32 GB HBM or `v5litepod-4` / `v5e-1` 16 GB HBM)

## Developer Guidelines & Agent Rules

### 1. Environment & Package Management
- **Rule**: Always use `uv` to manage Python environments and install packages locally and remotely.
- **Rule**: Always search the web for official documentation and verify API specifications before deep-diving into code.
- **Rule**: Do not modify installed official packages inside `.venv`. Focus on our own project code.

### 2. Multi-Host / Cloud Development Workflow
- **Rule**: **Do NOT modify code directly on the Cloud/TPU VM.**
- **Workflow**:
  1. Make and test code changes in the local workspace: `/Users/guanwang/Workspace/agy_CLI_Workspace/TorchTPU_sprint`.
  2. Run `uv run train_dry_run.py` locally on CPU to verify model config, LoRA attachment, and training/eval loop integrity.
  3. Package the project code into `project_code.tar.gz` and sync to the remote GCP TPU VM via `sync_and_run_tpu.sh`.
  4. Execute remote training/benchmarks and pull `results/` back to the local workspace.
  This maintains strict code consistency between local and cloud environments.

### 3. Gemma 4 PLE Weight Merging (LoRA Exporting)
- **Critical Limitation**: Gemma 4 contains a very large **Per-Layer Embedding (PLE) table (`embed_tokens_per_layer`, ~4.7 GB)**. On 16 GB HBM TPU chips (`v5e`), merging LoRA weights on-chip can trigger HBM exhaustion, whereas 32 GB `v6e-1` Trillium chips can merge in HBM or on host CPU RAM.
- **Rule**: Default weight merging and exporting to CPU (`--device cpu`) using the TPU VM's high-RAM host memory (or benchmark `--device tpu` on 32 GB `v6e-1` with automatic CPU fallback):
  ```bash
  uv run export_model.py --device cpu
  ```

### 4. Validation, Authentic Data Integrity & Pre-Teardown Backup
- **Strict Rule (Zero Simulated/Fake Data)**: **NEVER fake, hardcode, or simulate any metrics, loss curves, perplexities, or evaluation outputs for reporting or blogging.** Every single number, table, and chart in `README.md` and `blog/` must come 100% verbatim from real executions on the TPU VM (`results/metrics.json`, `results/execution_modes_bench.json`, `results/baseline_eval.json`, `results/finetuned_eval.json`, `base_perplexity.log`, `perplexity.log`, `training.log`).
- **Strict Rule (Local Sync Before Teardown)**: **NEVER delete the Cloud TPU VM until all logs, JSON metrics, evaluation outputs, and plots have been downloaded and verified in the local workspace (`/Users/guanwang/Workspace/agy_CLI_Workspace/TorchTPU_sprint/results/`).**
- **Rule**: Always validate merged weights using quantitative perplexity analysis and weight distribution checks (mean/std check on `embed_tokens`, `embed_tokens_per_layer`, and `q_proj` before vs. after merge) to confirm authentic LoRA updates and zero degradation:
  ```bash
  uv run eval_perplexity.py models/gemma-4-E2B-it-finetuned
  ```
