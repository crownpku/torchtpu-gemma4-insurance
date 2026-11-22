"""
Local CPU dry-run verification script for TorchTPU Gemma 4 Insurance SFT.
Mirrors `Tunix_Gemma4_TPU/train_dry_run.py` using Native PyTorch + PEFT LoRA.
"""

import json
import os
import torch
from peft import LoraConfig, get_peft_model
from gemma4_torchtpu import Gemma4ForCausalLMTorchTPU, Gemma4TextConfig


def dry_run():
    print("[Dry-Run] 1. Initializing scaled-down Gemma 4 config for CPU verification...")
    torch.manual_seed(0)
    config = Gemma4TextConfig.dry_run_tiny()
    device = torch.device("cpu")

    print("[Dry-Run] 2. Initializing Gemma4ForCausalLMTorchTPU model...")
    model = Gemma4ForCausalLMTorchTPU(config, dtype=torch.bfloat16).to(device)

    print("[Dry-Run] 3. Applying Hugging Face PEFT LoRA adapter...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(
        f"[Dry-Run] Trainable params: {trainable_params:,} || All params: {all_param:,} "
        f"|| Trainable%: {100 * trainable_params / all_param:.2f}%"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    metrics = {"train_loss": [], "eval_loss": []}

    print("[Dry-Run] 4. Running 5 training steps + 2 validation checks on CPU...")
    batch_size = 2
    seq_len = 16
    fixed_input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    fixed_attn_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    fixed_labels = fixed_input_ids.clone()
    fixed_labels[:, :4] = -100  # Mask prompt tokens

    for step in range(1, 6):
        model.train()
        optimizer.zero_grad()
        out = model(input_ids=fixed_input_ids, attention_mask=fixed_attn_mask, labels=fixed_labels)
        loss = out.loss
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu().item())
        metrics["train_loss"].append({"step": step, "loss": loss_val})
        print(f"  [Step {step}/5] train_loss = {loss_val:.4f}")

        if step in (1, 5):
            model.eval()
            with torch.no_grad():
                eval_out = model(input_ids=fixed_input_ids, attention_mask=fixed_attn_mask, labels=fixed_labels)
                eval_val = float(eval_out.loss.detach().cpu().item())
                metrics["eval_loss"].append({"step": step, "loss": eval_val})
                print(f"  [Eval @ Step {step}] eval_loss = {eval_val:.4f}")

    os.makedirs("results", exist_ok=True)
    out_path = "results/metrics_dry_run.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[Dry-Run] 5. Verifying PEFT merge_and_unload()...")
    merged_model = model.merge_and_unload()
    with torch.no_grad():
        merged_out = merged_model(input_ids=fixed_input_ids, attention_mask=fixed_attn_mask, labels=fixed_labels)
        print(f"  [Merged Model Check] post-merge eval_loss = {float(merged_out.loss.item()):.4f}")

    print(f"[Dry-Run] Completed successfully! Metrics written to {out_path}")


if __name__ == "__main__":
    dry_run()
