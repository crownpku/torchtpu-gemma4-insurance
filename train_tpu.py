"""
Main 500-step Gemma 4 (E2B-IT) Supervised Fine-Tuning (SFT) script on Google Cloud TPUs
using Native PyTorch (`torch_tpu`) + Hugging Face `peft` LoRA.

Direct PyTorch/TorchTPU evolution of `Tunix_Gemma4_TPU/train_tpu.py`.
"""

import argparse
import json
import os
import time
from typing import Iterator, List, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer

from gemma4_torchtpu import Gemma4ForCausalLMTorchTPU, get_torchtpu_device

# Constants matching Tunix_Gemma4_TPU/train_tpu.py
MODEL_DIR = "models/gemma-4-E2B-it"
TRAIN_DATA = "data/train.jsonl"
VAL_DATA = "data/validation.jsonl"
OUTPUT_DIR = "checkpoints_tpu"
SEQ_LEN = 1024
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
LORA_RANK = 16
LORA_ALPHA = 32
MAX_STEPS = 500
EVAL_EVERY_N_STEPS = 50


def configure_torchtpu_mode(mode: str):
    """Configures TorchTPU environment variables for the requested execution regime."""
    cache_dir = os.environ.get(
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
        os.path.expanduser("~/.cache/torchtpu_pjrt_cache"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = cache_dir

    if mode == "debug_eager":
        os.environ["TORCH_TPU_LAUNCH_BLOCKING"] = "1"
        os.environ.pop("TORCH_TPU_DEFER_AND_FUSE", None)
    elif mode == "strict_eager":
        os.environ.pop("TORCH_TPU_LAUNCH_BLOCKING", None)
        os.environ.pop("TORCH_TPU_DEFER_AND_FUSE", None)
    elif mode == "defer_and_fuse":
        os.environ.pop("TORCH_TPU_LAUNCH_BLOCKING", None)
        os.environ["TORCH_TPU_DEFER_AND_FUSE"] = "1"
    elif mode == "compile":
        os.environ.pop("TORCH_TPU_LAUNCH_BLOCKING", None)
        os.environ.pop("TORCH_TPU_DEFER_AND_FUSE", None)
    else:
        raise ValueError(f"Unknown execution mode: {mode}")


def get_tokenizer(model_dir: str = MODEL_DIR) -> Tokenizer:
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        sib = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Tunix_Gemma4_TPU", "models", "gemma-4-E2B-it", "tokenizer.json")
        )
        if os.path.exists(sib):
            tok_path = sib
    return Tokenizer.from_file(tok_path)


def format_example(example: dict, tokenizer: Tokenizer, seq_len: int = SEQ_LEN) -> Tuple[List[int], List[int], List[int]]:
    """
    Formats an InsuranceQA pair using Gemma 4's turn template (`<|turn|>` / `<turn|>`),
    matching `Tunix_Gemma4_TPU/train_tpu.py` with static padding to `seq_len=1024`.
    """
    bos = "<bos>"
    eos = "<eos>"
    sot = "<|turn|>"
    eot = "<turn|>"

    prompt = f"{bos}{sot}user\n{example['instruction']}{eot}\n{sot}model\n"
    response = f"{example['response']}{eot}{eos}"

    prompt_ids = tokenizer.encode(prompt).ids
    response_ids = tokenizer.encode(response).ids

    input_tokens = prompt_ids + response_ids
    # Completion-only mask: 1 on response tokens, 0 on prompt/padding tokens (matches Tunix input_mask)
    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

    if len(input_tokens) > seq_len:
        input_tokens = input_tokens[:seq_len]
        loss_mask = loss_mask[:seq_len]
    else:
        pad_len = seq_len - len(input_tokens)
        input_tokens += [0] * pad_len
        loss_mask += [0] * pad_len

    attn_mask = [1 if tok != 0 else 0 for tok in input_tokens]
    return input_tokens, attn_mask, loss_mask


def ds_gen(
    data_path: str,
    tokenizer: Tokenizer,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    is_training: bool = True,
    max_eval_batches: int | None = None,
) -> Iterator[dict]:
    """
    Static-shape batch generator yielding contiguous PyTorch tensors of shape `[batch_size, seq_len]`.
    Enforces `len(batch_tokens) == batch_size` so XLA never recompiles for partial tail batches.
    """
    batches_yielded = 0
    while True:
        with open(data_path, "r") as f:
            while True:
                batch_tokens = []
                batch_attn = []
                batch_loss_mask = []

                for _ in range(batch_size):
                    line = f.readline()
                    if not line:
                        break
                    example = json.loads(line)
                    tokens, attn, lmask = format_example(example, tokenizer, seq_len=seq_len)
                    batch_tokens.append(tokens)
                    batch_attn.append(attn)
                    batch_loss_mask.append(lmask)

                # Drop partial tail batch to guarantee static [BATCH_SIZE, SEQ_LEN] shapes on TPU
                if len(batch_tokens) < batch_size:
                    break

                yield {
                    "input_ids": torch.tensor(batch_tokens, dtype=torch.long).contiguous(),
                    "attention_mask": torch.tensor(batch_attn, dtype=torch.long).contiguous(),
                    "loss_mask": torch.tensor(batch_loss_mask, dtype=torch.float32).contiguous(),
                }
                batches_yielded += 1
                if not is_training and max_eval_batches is not None and batches_yielded >= max_eval_batches:
                    return
        if not is_training:
            break


class DatasetIterable:
    """
    Reusable iterable wrapper spawning a fresh generator every time `iter()` is invoked
    on multi-step evaluation boundaries (carrying forward the key fix from `Tunix_Gemma4_TPU`).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        batch_size: int = BATCH_SIZE,
        seq_len: int = SEQ_LEN,
        is_training: bool = True,
        max_eval_batches: int | None = None,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.is_training = is_training
        self.max_eval_batches = max_eval_batches

    def __iter__(self) -> Iterator[dict]:
        return ds_gen(
            self.data_path,
            self.tokenizer,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            is_training=self.is_training,
            max_eval_batches=self.max_eval_batches,
        )


class JSONMetricsHook:
    """Persistent JSON logger matching `Tunix_Gemma4_TPU/train_tpu.py`."""

    def __init__(self, output_file: str = "results/metrics.json"):
        self.output_file = output_file
        self.metrics = {"train_loss": [], "eval_loss": [], "step_time_ms": []}

    def on_train_start(self):
        print("[Hook] Training started on TorchTPU.", flush=True)

    def on_train_step_end(self, step: int, train_loss: float, step_time_ms: float, max_steps: int = MAX_STEPS):
        self.metrics["train_loss"].append({"step": int(step), "loss": float(train_loss)})
        self.metrics["step_time_ms"].append({"step": int(step), "ms": float(step_time_ms)})
        if step % 10 == 0 or step == max_steps or step == 1:
            self.save_metrics()
            print(
                f"[Hook] Step {step}/{max_steps}: train_loss = {train_loss:.4f} ({step_time_ms:.1f} ms/step)",
                flush=True,
            )

    def on_eval_step_end(self, step: int, eval_loss: float):
        self.metrics["eval_loss"].append({"step": int(step), "loss": float(eval_loss)})
        self.save_metrics()
        print(f"[Hook] Eval at step {step}: eval_loss = {eval_loss:.4f}", flush=True)

    def on_train_end(self):
        self.save_metrics()
        print(f"[Hook] Training completed. Metrics saved to {self.output_file}", flush=True)

    def save_metrics(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w") as f:
            json.dump(self.metrics, f, indent=2)


def compute_completion_loss(logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes completion-only cross-entropy loss matching Tunix PeftTrainer:
    mean token cross-entropy over response tokens where `loss_mask == 1`.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)

    masked_loss = per_token_loss * shift_mask
    denom = shift_mask.sum().clamp(min=1.0)
    return masked_loss.sum() / denom


@torch.no_grad()
def evaluate_validation_set(
    model: torch.nn.Module,
    val_iterable: DatasetIterable,
    device: torch.device,
    full_val_batches: int = 838,
) -> float:
    """Evaluates validation loss over `val_iterable` scaled to full validation split."""
    model.eval()
    total_eval_loss = torch.zeros((), dtype=torch.float32, device=device)
    n_batches = 0
    for batch in val_iterable:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device, non_blocking=True)

        out = model(input_ids=input_ids, attention_mask=attn_mask, loss_mask=loss_mask, return_logits=False)
        batch_loss = out.loss
        total_eval_loss = total_eval_loss + batch_loss
        n_batches += 1

    model.train()
    raw_sum = float(total_eval_loss.cpu().item())
    if n_batches > 0 and n_batches < full_val_batches:
        return raw_sum * (float(full_val_batches) / float(n_batches))
    return raw_sum


def main():
    parser = argparse.ArgumentParser(description="TorchTPU Gemma 4 E2B-IT InsuranceQA LoRA SFT")
    parser.add_argument(
        "--mode",
        type=str,
        default="defer_and_fuse",
        choices=["debug_eager", "strict_eager", "defer_and_fuse", "compile"],
        help="TorchTPU execution regime",
    )
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--device", type=str, default=None, help="Force 'tpu' or 'cpu'")
    parser.add_argument("--output-metrics", type=str, default="results/metrics.json")
    args = parser.parse_args()

    configure_torchtpu_mode(args.mode)
    device = get_torchtpu_device(args.device)
    print(f"[TorchTPU] Using device: {device} | Execution mode: {args.mode}")

    tokenizer = get_tokenizer(args.model_dir)

    # 1. Load Gemma 4 (E2B-IT) directly from safetensors in bfloat16
    print(f"[TorchTPU] Loading Gemma 4 E2B-IT from {args.model_dir}...")
    base_model = Gemma4ForCausalLMTorchTPU.from_safetensors(
        args.model_dir,
        device=device,
        dtype=torch.bfloat16,
    )

    # 2. Apply Hugging Face PEFT LoRA matching Tunix_Gemma4_TPU (r=16, alpha=32, q/k/v/o + gate/up/down)
    print(f"[TorchTPU] Applying PEFT LoRA (r={LORA_RANK}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(
        f"[TorchTPU] Trainable params: {trainable_params:,} || All params: {all_param:,} "
        f"|| Trainable%: {100 * trainable_params / all_param:.4f}%"
    )

    if args.mode == "compile" and device.type == "tpu":
        print("[TorchTPU] Compiling model forward pass with torch.compile(backend='tpu')...")
        model = torch.compile(model, backend="tpu")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    train_iterable = DatasetIterable(
        TRAIN_DATA, tokenizer, batch_size=args.batch_size, seq_len=args.seq_len, is_training=True
    )
    val_iterable = DatasetIterable(
        VAL_DATA,
        tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        is_training=False,
        max_eval_batches=args.max_eval_batches,
    )

    hook = JSONMetricsHook(output_file=args.output_metrics)

    # Step 0 baseline validation evaluation (matching Tunix Step 0 eval)
    eval_0 = evaluate_validation_set(model, val_iterable, device)
    hook.on_eval_step_end(0, eval_0)
    hook.on_train_start()

    train_iter = iter(train_iterable)
    model.train()

    for step in range(1, args.max_steps + 1):
        t0 = time.perf_counter()
        batch = next(train_iter)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attn_mask, loss_mask=loss_mask, return_logits=False)
        loss = out.loss
        loss.backward()
        optimizer.step()

        # Synchronize scalar loss to host for logging
        loss_val = float(loss.detach().cpu().item())
        dt_ms = (time.perf_counter() - t0) * 1000.0
        hook.on_train_step_end(step, loss_val, dt_ms, max_steps=args.max_steps)

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_loss = evaluate_validation_set(model, val_iterable, device)
            hook.on_eval_step_end(step - 1 if step < args.max_steps else step, eval_loss)
            # Save periodic checkpoint at every evaluation boundary
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            step_ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{step}")
            unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
            unwrapped.save_pretrained(step_ckpt_dir)

    hook.on_train_end()

    # Save trained PEFT LoRA adapter checkpoint
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    adapter_dir = os.path.join(OUTPUT_DIR, f"step_{args.max_steps}")
    unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    unwrapped.save_pretrained(adapter_dir)
    print(f"[TorchTPU] Saved trained PEFT LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
