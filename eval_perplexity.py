"""
Quantitative Perplexity & Weight Distribution Verifier on TorchTPU / CPU.
Direct PyTorch evolution of `Tunix_Gemma4_TPU/eval_perplexity.py`.
"""

import json
import math
import os
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from gemma4_torchtpu import Gemma4ForCausalLMTorchTPU, get_torchtpu_device


@torch.no_grad()
def compute_perplexity(
    model: Gemma4ForCausalLMTorchTPU,
    tokenizer: Tokenizer,
    data_file: str,
    device: torch.device,
    num_samples: int = 10,
):
    print(f"Computing perplexity for {data_file} on {device}...")
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with open(data_file, "r") as f:
        lines = [ f.readline() for _ in range(num_samples) ]

    static_len = 512
    for idx, line in enumerate(lines):
        if not line:
            break
        example = json.loads(line)
        instruction = example["instruction"]
        response = example["response"]
        text = f"<bos><|turn|>user\n{instruction}<turn|>\n<|turn|>model\n{response}<turn|><eos>"

        token_ids = tokenizer.encode(text).ids[:static_len]
        cur_len = len(token_ids)
        padded_ids = token_ids + [0] * (static_len - cur_len)
        padded_mask = [1] * cur_len + [0] * (static_len - cur_len)

        input_ids = torch.tensor([padded_ids], dtype=torch.long, device=device)
        attn_mask = torch.tensor([padded_mask], dtype=torch.long, device=device)

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits

        if idx < 2:
            preds = torch.argmax(logits[0, : cur_len - 1, :], dim=-1).cpu().tolist()
            actual = token_ids[1:cur_len]
            print(f"DEBUG Sample {idx+1} Preds : {preds[:10]} -> {tokenizer.decode(preds[:10])!r}")
            print(f"DEBUG Sample {idx+1} Actual: {actual[:10]} -> {tokenizer.decode(actual[:10])!r}")

        shift_logits = logits[0, : cur_len - 1, :]
        shift_labels = input_ids[0, 1:cur_len]
        loss_sum = F.cross_entropy(shift_logits, shift_labels, reduction="sum")

        total_loss += float(loss_sum.cpu().item())
        total_tokens += int(shift_labels.numel())

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def main(model_dir: str, device_pref: str | None = None):
    device = get_torchtpu_device(device_pref)
    model = Gemma4ForCausalLMTorchTPU.from_safetensors(model_dir, device=device, dtype=torch.bfloat16)

    tok_path = os.path.join(model_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    print(f"DEBUG: Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    lm = model.language_model
    emb_w = lm.embed_tokens.weight.detach().float()
    print(f"DEBUG: Input embedding mean: {emb_w.mean().item():.6f}, std: {emb_w.std().item():.6f}, max: {emb_w.max().item():.6f}")

    ple_w = lm.embed_tokens_per_layer.weight.detach().float()
    print(f"DEBUG: PLE table mean:       {ple_w.mean().item():.6f}, std: {ple_w.std().item():.6f}, max: {ple_w.max().item():.6f}")

    q_w = lm.layers[0].self_attn.q_proj.weight.detach().float()
    print(f"DEBUG: Layer0 q_proj mean:   {q_w.mean().item():.6f}, std: {q_w.std().item():.6f}, max: {q_w.max().item():.6f}")

    ppl, loss = compute_perplexity(model, tokenizer, "data/test.jsonl", device=device, num_samples=10)
    print(f"MODEL: {model_dir}")
    print(f"PERPLEXITY: {ppl:.4f}")
    print(f"LOSS: {loss:.4f}")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "models/gemma-4-E2B-it"
    dev_arg = sys.argv[2] if len(sys.argv) > 2 else None
    main(target_dir, dev_arg)
