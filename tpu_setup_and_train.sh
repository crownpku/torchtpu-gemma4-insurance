#!/usr/bin/env bash
set -euo pipefail

echo "[1/7] Extracting project_code.tar.gz..."
if [ -f project_code.tar.gz ]; then
    tar -xzf project_code.tar.gz
fi

echo "[2/7] Setting up uv & Python 3.12 environment..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d .venv ]; then
    uv python install 3.12
    uv venv --python 3.12 .venv
fi
source .venv/bin/activate

echo "[3/7] Verifying PyTorch + TorchTPU (torch_tpu) + Hugging Face PEFT..."
if ! python -c "import torch, torch_tpu, peft, safetensors, tokenizers, datasets, matplotlib" >/dev/null 2>&1; then
    uv pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch \
        torch_tpu \
        peft \
        safetensors \
        tokenizers \
        datasets \
        huggingface-hub \
        accelerate \
        matplotlib \
        numpy \
        tqdm \
        tpu-info \
        tensorboard-plugin-profile
fi

export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="$HOME/.cache/torchtpu_pjrt_cache"
mkdir -p "$TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT" results

echo "[4/7] Preparing InsuranceQA-v2 dataset & verifying Gemma 4 E2B-IT weights..."
if [ -n "${1:-}" ]; then
    export HUGGING_FACE_TOKEN="$1"
    export HF_TOKEN="$1"
fi
if [ ! -f data/train.jsonl ]; then
    python prepare_data.py
else
    echo "  -> data/train.jsonl already present."
fi
python download_model.py

echo "[5/7] Benchmarking TorchTPU Execution Modes (Debug Eager vs Strict Eager vs DeferAndFuse vs torch.compile)..."
if [ ! -f results/execution_modes_bench.json ]; then
    python bench_execution_modes.py --steps 10 --warmup 2 --profile-xprof 2>&1 | tee bench_modes.log
else
    echo "  -> results/execution_modes_bench.json already present; skipping redundant mode benchmark."
fi

echo "[6/7] Running 500-Step Gemma 4 LoRA SFT on TorchTPU..."
export TORCH_TPU_DEFER_AND_FUSE=1
if [ ! -f checkpoints_tpu/step_500/adapter_model.safetensors ]; then
    python train_tpu.py --mode defer_and_fuse 2>&1 | tee training.log
else
    echo "  -> checkpoints_tpu/step_500/adapter_model.safetensors already present; skipping redundant 500-step training."
fi

echo "[7/7] Merging LoRA weights (CPU PLE-safe export), verifying perplexity & running 20-sample evaluation..."
python export_model.py --device cpu 2>&1 | tee export.log
python eval_perplexity.py models/gemma-4-E2B-it tpu 2>&1 | tee base_perplexity.log
python eval_perplexity.py models/gemma-4-E2B-it-finetuned tpu 2>&1 | tee perplexity.log
python eval_tpu.py models/gemma-4-E2B-it results/baseline_eval.json 20 tpu
python eval_tpu.py models/gemma-4-E2B-it-finetuned results/finetuned_eval.json 20 tpu
python compare_models.py
python plot_metrics.py
echo "[Done] All TorchTPU training, evaluation, and plotting steps completed!"
