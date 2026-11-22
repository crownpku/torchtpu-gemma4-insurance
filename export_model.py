"""
PLE-Aware PEFT LoRA Weight Merging & Safetensors Export Script for TorchTPU.

Compare with `Tunix_Gemma4_TPU/export_model.py` (154 lines of manual NNX graph traversal,
`kv_einsum` splitting, and tensor transpose rules): in PyTorch + Hugging Face `peft`,
`PeftModel.from_pretrained` + `merge_and_unload()` merges all 35 layers cleanly in ~30 lines!
"""

import argparse
import os
import time
import torch
from peft import PeftModel
from gemma4_torchtpu import Gemma4ForCausalLMTorchTPU, get_torchtpu_device

BASE_MODEL_DIR = "models/gemma-4-E2B-it"
LORA_CHECKPOINT_DIR = "checkpoints_tpu/step_500"
MERGED_OUTPUT_DIR = "models/gemma-4-E2B-it-finetuned"


def main():
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA adapter into Gemma 4 E2B-IT and export safetensors")
    parser.add_argument("--base-model-dir", type=str, default=BASE_MODEL_DIR)
    parser.add_argument("--lora-dir", type=str, default=LORA_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=str, default=MERGED_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "tpu"],
        help="Default 'cpu' uses host RAM (188GB on TPU VM) to avoid 16GB HBM OOM from the 4.7GB PLE table; 'tpu' tests on-chip merge on 32GB v6e.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.lora_dir):
        raise FileNotFoundError(
            f"LoRA checkpoint directory '{args.lora_dir}' not found! Run `train_tpu.py` first."
        )

    t0 = time.perf_counter()
    target_device = get_torchtpu_device(args.device)
    print(f"[Export] 1. Loading base Gemma 4 E2B-IT model on {target_device}...")
    base_model = Gemma4ForCausalLMTorchTPU.from_safetensors(
        args.base_model_dir,
        device=target_device,
        dtype=torch.bfloat16,
    )

    # Record pre-merge Q_proj weight std for strict verification (preventing silent no-op merges)
    q0_before = base_model.language_model.layers[0].self_attn.q_proj.weight.detach().float().cpu().clone()

    print(f"[Export] 2. Attaching trained PEFT LoRA adapter from {args.lora_dir}...")
    # Explicit torch_device="cpu" prevents peft's infer_device() from passing device="tpu"
    # to Rust safetensors.torch.load_file(), which only recognizes standard PyTorch devices.
    peft_model = PeftModel.from_pretrained(base_model, args.lora_dir, torch_device="cpu")
    if args.device == "tpu":
        peft_model = peft_model.to(target_device)

    print("[Export] 3. Running PEFT merge_and_unload()...")
    merged_model: Gemma4ForCausalLMTorchTPU = peft_model.merge_and_unload()

    q0_after = merged_model.language_model.layers[0].self_attn.q_proj.weight.detach().float().cpu()
    max_diff = float((q0_after - q0_before).abs().max().item())
    print(f"[Export] Layer 0 q_proj max weight delta after LoRA merge: {max_diff:.6f}")
    if max_diff == 0.0:
        raise RuntimeError("Silent failure detected: merged weights are identical to base weights (max_diff == 0)!")

    print(f"[Export] 4. Saving merged safetensors to {args.output_dir}...")
    merged_model.save_merged_safetensors(args.base_model_dir, args.output_dir)
    elapsed = time.perf_counter() - t0
    print(f"[Export] Export complete in {elapsed:.2f}s! Merged model is in {args.output_dir}")


if __name__ == "__main__":
    main()
