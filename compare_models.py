"""
Side-by-Side Model Evaluation & Comparison Script:
Compares Reference Gold Standard vs. Baseline Gemma 4 vs. Tunix (JAX) Fine-Tuned vs. TorchTPU (PyTorch) Fine-Tuned.
"""

import json
import os


def compare_outputs(
    base_file: str = "results/baseline_eval.json",
    torchtpu_file: str = "results/finetuned_eval.json",
    tunix_file: str = "../Tunix_Gemma4_TPU/results/comparison_results.json",
    output_file: str = "results/comparison_results.json",
):
    tunix_map = {}
    if os.path.exists(tunix_file):
        with open(tunix_file, "r") as f:
            for item in json.load(f):
                tunix_map[item["question"]] = item

    base_data = []
    if os.path.exists(base_file):
        with open(base_file, "r") as f:
            base_data = json.load(f)

    torchtpu_data = []
    if os.path.exists(torchtpu_file):
        with open(torchtpu_file, "r") as f:
            torchtpu_data = json.load(f)

    combined = []
    count = max(len(base_data), len(torchtpu_data))
    for i in range(count):
        q = (
            torchtpu_data[i]["instruction"]
            if i < len(torchtpu_data)
            else base_data[i]["instruction"]
        )
        ref = (
            torchtpu_data[i]["expected"]
            if i < len(torchtpu_data)
            else base_data[i]["expected"]
        )
        base_ans = (
            base_data[i]["generated"]
            if i < len(base_data)
            else tunix_map.get(q, {}).get("base_model", "")
        )
        tunix_ans = tunix_map.get(q, {}).get("finetuned_model", "")
        torchtpu_ans = torchtpu_data[i]["generated"] if i < len(torchtpu_data) else ""

        combined.append(
            {
                "id": i + 1,
                "question": q,
                "reference_answer": ref,
                "base_model": base_ans,
                "tunix_jax_finetuned": tunix_ans,
                "torchtpu_pytorch_finetuned": torchtpu_ans,
            }
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[Compare] Saved {len(combined)} side-by-side comparisons to {output_file}")


if __name__ == "__main__":
    compare_outputs()
