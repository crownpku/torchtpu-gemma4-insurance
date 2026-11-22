"""
Qualitative Evaluation Script on TorchTPU (`torch.device("tpu")`) or CPU.
Direct PyTorch evolution of `Tunix_Gemma4_TPU/eval_tpu_v2.py`.
"""

import json
import os
import sys
import time
import torch
from tokenizers import Tokenizer
from gemma4_torchtpu import Gemma4ForCausalLMTorchTPU, get_torchtpu_device


def run_eval(model_dir: str, output_file: str, num_samples: int = 20, device_pref: str | None = None):
    device = get_torchtpu_device(device_pref)
    print(f"[Eval-TorchTPU] Loading model from {model_dir} onto {device}...")
    model = Gemma4ForCausalLMTorchTPU.from_safetensors(model_dir, device=device, dtype=torch.bfloat16)
    model.eval()

    tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    eos_ids = (
        tokenizer.token_to_id("<eos>") or 1,
        tokenizer.token_to_id("<turn|>") or 106,
    )

    test_data = "data/test.jsonl"
    results = []

    print(f"[Eval-TorchTPU] Running generation on {num_samples} held-out InsuranceQA samples...")
    with open(test_data, "r") as f:
        lines = [f.readline() for _ in range(num_samples)]

    for idx, line in enumerate(lines):
        if not line:
            break
        example = json.loads(line)
        instruction = example["instruction"]
        prompt = f"<bos><|turn|>user\n{instruction}<turn|>\n<|turn|>model\n"
        prompt_ids = tokenizer.encode(prompt).ids

        t0 = time.perf_counter()
        out_ids = model.generate_response(
            prompt_ids=prompt_ids,
            max_new_tokens=128,
            eos_token_ids=eos_ids,
            temperature=0.0,
            device=device,
        )
        dt_s = time.perf_counter() - t0
        response = tokenizer.decode(out_ids).strip()

        results.append(
            {
                "instruction": instruction,
                "expected": example["response"],
                "generated": response,
                "num_output_tokens": len(out_ids),
                "latency_s": round(dt_s, 3),
            }
        )
        print(f"[{idx+1}/{num_samples}] Q: {instruction}\nA ({len(out_ids)} tok, {dt_s:.2f}s): {response}\n")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[Eval-TorchTPU] Results saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run eval_tpu.py <model_dir> <output_file> [num_samples] [device]")
        sys.exit(1)
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    dev = sys.argv[4] if len(sys.argv) > 4 else None
    run_eval(sys.argv[1], sys.argv[2], num_samples=n_samples, device_pref=dev)
