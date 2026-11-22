import json
import os
import shutil


def prepare_dataset():
    os.makedirs("data", exist_ok=True)
    sibling_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Tunix_Gemma4_TPU", "data"))

    # Fast local reuse if Tunix_Gemma4_TPU/data already has the formatted splits
    if all(os.path.exists(os.path.join(sibling_data_dir, f"{s}.jsonl")) for s in ("train", "validation", "test")):
        for s in ("train", "validation", "test"):
            src = os.path.join(sibling_data_dir, f"{s}.jsonl")
            dst = os.path.join("data", f"{s}.jsonl")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            print(f"[prepare_data] Ready: {dst} ({os.path.getsize(dst) / 1e6:.2f} MB)")
        return

    from datasets import load_dataset

    dataset = load_dataset("deccan-ai/insuranceQA-v2")
    for split in dataset.keys():
        output_file = f"data/{split}.jsonl"
        print(f"Processing {split} split...")
        with open(output_file, "w") as f:
            for item in dataset[split]:
                formatted_item = {
                    "instruction": item["input"].strip(),
                    "response": item["output"].strip(),
                }
                f.write(json.dumps(formatted_item) + "\n")
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    prepare_dataset()
