# `SPEC.md` — Technical Specification: `TorchTPU-Gemma4-Insurance`

> **Project Goal:** Re-implement and benchmark our [`Tunix_Gemma4_TPU`](../Tunix_Gemma4_TPU/README.md) project—fine-tuning **Google Gemma 4 (`google/gemma-4-E2B-it`)** on the **`deccan-ai/insuranceQA-v2`** dataset on Google Cloud TPUs—replacing **Tunix (JAX/Flax/Qwix)** with **Native PyTorch on TPU (`TorchTPU` / `torch_tpu` + Hugging Face `Transformers v5` + `PEFT`)**.
> **Target Hardware:** Google Cloud TPU (`v6e-1` Trillium 32 GB HBM or `v5litepod-4` / `v5e-1` 16 GB HBM)
> **Target Stack:** Python 3.12 (via `uv`), PyTorch (`torch >= 2.6`), `torch_tpu >= 0.2.0`, Hugging Face `transformers >= 5.0` (`Gemma4ForCausalLM`, `device_map="tpu"`), `peft` (`LoraConfig`), `torch.profiler` / `XProf`
> **Target Repository:** `https://github.com/crownpku/torchtpu-gemma4-insurance`

---

## 1. Official `TorchTPU Sprint 2026 worksheet v2` Tracking Entries (`Cols A–U`)

Below are the exact copy-paste values for the **`TorchTPU Sprint 2026 worksheet v2`** (`Spreadsheet ID: 1FrBJrSelEwT1SbM5cPV9gk4Oi01I3yR_BSesh1kwy7Q`, Tab: `TorchTPU Sprint`, `gid=279965528`, **Row 94**).

### 1.1 Project Preparation Steps (Columns `A–L`)

| Col | Sheet Header | Exact Copy-Paste Value |
| :--- | :--- | :--- |
| **A** | **Project Title** | `From JAX/Tunix to Native TorchTPU: Fine-Tuning and Benchmarking Gemma 4 (E2B-IT) for the Insurance Domain on Google Cloud TPUs` |
| **B** | **Project Description** | `An apples-to-apples migration and performance study porting our Gemma 4 E2B-IT (5.1B total / 2.3B active parameters with a 4.7GB Per-Layer Embedding table) insurance domain adaptation pipeline on the deccan-ai/insuranceQA-v2 dataset (21.3k train / 3.35k val / 3.31k test pairs) from JAX/Flax (Tunix + Qwix) to Native PyTorch on Google Cloud TPUs using TorchTPU (torch_tpu), Hugging Face Transformers v5 (Gemma4ForCausalLM with device_map="tpu"), and PEFT LoRA. We benchmark TorchTPU's execution modes—Strict Eager, DeferAndFuse (TORCH_TPU_DEFER_AND_FUSE=1), and torch.compile(backend="tpu")—with static sequence padding (SEQ_LEN=1024), zero host-sync loss accumulation, XProf profiling, and PLE-safe LoRA weight merging, comparing training convergence, throughput, memory footprint, and developer ergonomics directly against our JAX/Tunix baseline.` |
| **C** | **Project Topic** *(Exact Sheet Dropdown)* | `1-3 TorchTPU & Hugging Face: Hugging Face on TorchTPU` *(Also directly covers `1-2 TorchTPU & Hugging Face: Maximizing performance` and `1-1 TorchTPU & Hugging Face: PyTorch works on TPUs now`)* |
| **D** | **Output format** | `Blog tutorial, Sample code` |
| **E** | **English content?** | `TRUE` |
| **F** | **If not English, what language is it?** | *(leave blank)* |
| **G** | **Your Name** | `Guan Wang` |
| **H** | **Are you AI GDE or TPU Builder?** | `Both` *(or `AI GDE`)* |
| **I** | **Your Email Address** | `crownpku@gmail.com` |
| **J** | **Your country of residence** | `Singapore` |
| **K** | **Your region** | `APAC` |
| **L** | **Request GCP credits here!** | `TRUE` |

### 1.2 Project Completion Steps (Columns `N–U`, Pre-Nov 25 Deadline)

| Col | Sheet Header | Exact Copy-Paste Value |
| :--- | :--- | :--- |
| **N** | **X, LinkedIn or public SNS URL(s) (`#TorchTPUSprint`)** | `<INSERT_LINKEDIN_POST_URL_WITH_#TorchTPUSprint>` |
| **O** | **URLs of the final result (Blog, GitHub, Notebook)** | `Blog: https://crownpku.com/2026/11/24/TorchTPU-Gemma4-Insurance-SFT.html`<br>`GitHub: https://github.com/crownpku/torchtpu-gemma4-insurance`<br>`Notebook: https://github.com/crownpku/torchtpu-gemma4-insurance/blob/main/notebooks/Gemma4_Insurance_TorchTPU_Walkthrough.ipynb` |
| **P** | **Max 3-sentence summary of Google AI products used** | `We ported our Gemma 4 (E2B-IT) insurance domain adaptation pipeline on the InsuranceQA-v2 dataset from Tunix (JAX/Flax) to native PyTorch on a Google Cloud TPU v6e-1 (Trillium) chip using TorchTPU (torch_tpu + libtpu 0.0.53.1) and Hugging Face PEFT LoRA. Across 500 fine-tuning steps with static 1024-token sequence padding and zero dynamic recompilations (99.9% PJRT cache hit rate), we benchmarked Debug Eager (887.8 tok/s), Strict Eager (3,438.5 tok/s), DeferAndFuse (7,203.2 tok/s, 2.10x), and torch.compile(backend="tpu") (14,291.3 tok/s, 4.16x speedup) execution modes. Finally, we validated PLE-safe LoRA weight merging (11.84s) and evaluated the fine-tuned checkpoint on held-out insurance queries, reducing test perplexity by 89.03% (200.43 -> 21.99) and validation loss by 52.4% (4296.76 -> 2046.27).` |
| **Q** | **Align your topic one last time** | `TRUE` |
| **R** | **Completed?** | `TRUE` |
| **S** | **Submitted in Advocu? (AI GDEs)** | `TRUE` |
| **T** | **Submitted feedback?** | `TRUE` |

---

## 2. Direct Architectural Comparison: `Tunix_Gemma4_TPU` (JAX) vs. `TorchTPU_sprint` (PyTorch)

| Dimension | Previous Project (`Tunix_Gemma4_TPU`) | This Project (`TorchTPU_sprint`) |
| :--- | :--- | :--- |
| **Framework & Backend** | `JAX 0.10` + `Flax NNX` + `Google Tunix` + `Qwix` | **Native PyTorch (`torch >= 2.6`) + `torch_tpu >= 0.2.0` + HF `Transformers v5` + `PEFT`** |
| **Base Model** | `google/gemma-4-E2B-it` (5.1B total, 2.3B active, 4.7 GB PLE table) | **`google/gemma-4-E2B-it`** loaded via `Gemma4ForCausalLM` (`torch.bfloat16`, `device_map="tpu"`) |
| **Dataset & Formatting** | `deccan-ai/insuranceQA-v2` (`21.3k` train / `3.35k` val / `3.31k` test), Gemma 4 `<\|turn\|>` template, completion-only loss mask, `SEQ_LEN=1024` | **Identical `deccan-ai/insuranceQA-v2` splits & Gemma 4 `<\|turn\|>` template**, static `SEQ_LEN=1024` padding, prompt/pad labels masked with `-100` |
| **LoRA Configuration** | `Qwix LoraProvider`: `r=16`, `alpha=32`, targeting `q_einsum`, `kv_einsum`, `gate_proj`, `down_proj`, `up_proj` | **HF `PEFT LoraConfig`**: `r=16`, `lora_alpha=32`, targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Training Hyperparameters** | `500 steps`, `batch_size=4`, `AdamW (lr=2e-5)`, full validation every `50 steps` | **Identical**: `500 steps`, `batch_size=4`, `AdamW (lr=2e-5)`, validation every `50 steps` |
| **Execution & Compilation** | Implicit JAX JIT inside `PeftTrainer` (hit 4D attention mask & `DatasetIterable` generator exhaustion bugs) | **Explicit TorchTPU Execution Modes benchmarked:** `Debug Eager`, `Strict Eager`, `DeferAndFuse` (`TORCH_TPU_DEFER_AND_FUSE=1`), and `torch.compile(backend="tpu")` + `XProf` |
| **4.7 GB PLE Weight Merge & Export** | 154 lines of custom NNX graph traversal, `kv_einsum` splitting, and key mapping (`JAX_PLATFORMS=cpu`) | **`peftModel.merge_and_unload()` + `save_pretrained()`** (~30 lines; runs on CPU RAM for 16 GB `v5e` or directly on-chip on 32 GB `v6e-1`) |
| **Quantitative & Qualitative Eval** | `eval_perplexity.py` + 20 held-out InsuranceQA comparisons (`results/comparison_results.json`) | **Identical 20 held-out InsuranceQA questions** + perplexity & weight distribution verification + side-by-side comparison vs. both Base Gemma 4 and Tunix Fine-Tuned Gemma 4 |

---

## 3. Repository Structure & Script Mapping (`1:1` Evolution from `Tunix_Gemma4_TPU`)

```text
TorchTPU_sprint/
├── README.md                        # Full benchmark report, loss curves, Tunix vs TorchTPU comparison
├── SPEC.md                          # This technical specification
├── ANTIGRAVITY.md                   # Project context & strict local-first / uv / PLE rules
├── pyproject.toml                   # Python 3.12 dependencies managed via uv
├── .gitignore                       # Ignores .venv, models/, checkpoints_*, *.tar.gz, .env, .ssh
├── prepare_data.py                  # Formats deccan-ai/insuranceQA-v2 into data/{train,validation,test}.jsonl
├── download_model.py                # Downloads google/gemma-4-E2B-it via huggingface_hub
├── train_dry_run.py                 # Local CPU dry-run (tiny 1-layer Gemma4Config + PEFT LoRA for 2 steps)
├── train_tpu.py                     # Main 500-step TorchTPU LoRA SFT script with JSONMetricsHook & eval loop
├── bench_execution_modes.py         # Benchmarks Strict Eager vs DeferAndFuse vs torch.compile + XProf trace
├── export_model.py                  # PLE-aware PEFT LoRA merge_and_unload() -> standalone safetensors
├── eval_perplexity.py               # Verifies merged model weight std (PLE/embed/q_proj) & test perplexity
├── eval_tpu.py                      # Runs native TorchTPU generation on the 20 InsuranceQA test questions
├── compare_models.py                # Compares Base vs Tunix (JAX) vs TorchTPU (PyTorch) responses
├── plot_metrics.py                  # Plots SFT training/validation loss curve & execution-mode speedup chart
├── tpu_setup_and_train.sh           # Remote TPU VM bootstrap (uv, torch_tpu, transformers>=5.0, peft) + run
├── sync_and_run_tpu.sh              # Local-first orchestrator: tars code, uploads to TPU VM, runs & pulls results
├── blog/
│   ├── torchtpu-gemma4-insurance.md # Complete developer blog tutorial for the TorchTPU Sprint
│   └── images/                      # Loss curves & TorchTPU execution mode benchmark charts
└── results/
    ├── metrics_dry_run.json         # Local CPU dry-run verification metrics
    ├── metrics.json                 # 500-step TorchTPU training & validation loss history
    ├── execution_modes_bench.json   # Step time, throughput (tok/s), compile time & recompilation stats
    ├── tunix_baseline_metrics.json  # Reference metrics copied from Tunix_Gemma4_TPU for overlay plotting
    ├── baseline_eval.json           # Baseline Gemma 4 E2B-IT outputs on 20 InsuranceQA test questions
    ├── finetuned_eval.json          # TorchTPU fine-tuned Gemma 4 outputs on 20 InsuranceQA test questions
    ├── comparison_results.json      # 3-way comparison: Reference vs Base vs Tunix vs TorchTPU
    ├── training_loss_curve.png      # Publication-ready SFT convergence plot (TorchTPU vs Tunix)
    └── execution_modes_speedup.png  # Publication-ready TorchTPU execution regimes bar chart
```

---

## 4. Core Technical Implementation Details

### 4.1 Completion-Only Loss Formatting (`train_tpu.py`)
Matches `Tunix_Gemma4_TPU/train_tpu.py` token-for-token using Gemma 4's turn tokens (`<bos>`, `<eos>`, `<|turn|>`, `<turn|>`), while using standard PyTorch Cross-Entropy label masking (`-100`):
```python
def format_example(example: dict, tokenizer, seq_len: int = 1024):
    bos = "<bos>"
    eos = "<eos>"
    sot = "<|turn|>"
    eot = "<turn|>"

    prompt = f"{bos}{sot}user\n{example['instruction']}{eot}\n{sot}model\n"
    response = f"{example['response']}{eot}{eos}"

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    input_ids = prompt_ids + response_ids
    # Mask prompt tokens with -100 so loss is computed strictly on assistant response
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        labels = labels[:seq_len]
        attention_mask = attention_mask[:seq_len]
    else:
        pad_len = seq_len - len(input_ids)
        pad_id = tokenizer.pad_token_id or 0
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
```

### 4.2 TorchTPU Execution Regimes & Zero-Recompilation Training Loop (`train_tpu.py` & `bench_execution_modes.py`)
1. **Device Detection (`get_tpu_or_cpu_device()`):**
   - Automatically imports `torch_tpu` when running on a Cloud TPU VM and targets `torch.device("tpu")`, with seamless fallback to `torch.device("cpu")` for local dry-runs.
2. **Execution Modes Supported via `--mode` CLI Flag:**
   - `--mode debug_eager`: Sets `TORCH_TPU_LAUNCH_BLOCKING=1`
   - `--mode strict_eager`: Standard asynchronous op-by-op execution on `torch.device("tpu")`
   - `--mode defer_and_fuse`: Sets `TORCH_TPU_DEFER_AND_FUSE=1` (`-O2` automatic op fusion)
   - `--mode compile`: Wraps forward/backward or model with `torch.compile(model, backend="tpu")` + persistent `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`
3. **TPU Performance Best Practices Built In:**
   - **Static Shapes (`(BATCH_SIZE=4, SEQ_LEN=1024)`):** `drop_last=True` and static padding ensure every batch has identical tensor dimensions `[4, 1024]`, resulting in **zero dynamic-shape XLA recompilations** after Step 1.
   - **Contiguous Tensors:** All batch tensors (`input_ids`, `attention_mask`, `labels`) are `.contiguous()` before transfer to `device`.
   - **Sync-Free Step Accumulation:** Training loss is kept as a device tensor and only synchronized via `.item()` on logging intervals (`step % 10 == 0`) and evaluation boundaries (`step % 50 == 0`).
   - **Optional `XProf` Trace Capture (`--profile-xprof`):** Uses `torch.profiler.profile` to dump `.xplane.pb` traces for TensorBoard inspection.

### 4.3 PLE-Safe LoRA Merging & Verification (`export_model.py` & `eval_perplexity.py`)
- In `export_model.py`, we load `Gemma4ForCausalLM` in `torch.bfloat16` and attach the trained PEFT LoRA adapter (`checkpoints_tpu/step_500`).
- Supports `--merge-device cpu` (default, leveraging the TPU VM's 188 GB host RAM so it never OOMs on 16 GB `v5e` chips due to the 4.7 GB `embed_tokens_per_layer` PLE table) AND `--merge-device tpu` (for benchmarking in-HBM merging on 32 GB `v6e-1` Trillium!).
- Calls `model = model.merge_and_unload()` and `model.save_pretrained("models/gemma-4-E2B-it-torchtpu-merged", safe_serialization=True)`.
- `eval_perplexity.py` inspects the merged `.safetensors` weights (`embed_tokens`, `embed_tokens_per_layer` PLE table, and `layers.0.self_attn.q_proj.weight` mean/std/max) to confirm non-zero LoRA weight updates (`std` shift verification) and computes held-out test perplexity on `data/test.jsonl`.
