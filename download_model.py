import os
from huggingface_hub import snapshot_download


def download_gemma():
    model_id = "google/gemma-4-E2B-it"
    local_dir = "models/gemma-4-E2B-it"
    sibling_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Tunix_Gemma4_TPU", "models", "gemma-4-E2B-it")
    )

    if os.path.exists(os.path.join(local_dir, "model.safetensors")):
        print(f"[download_model] Model already present at {local_dir}")
        return

    if os.path.exists(os.path.join(sibling_dir, "model.safetensors")):
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(local_dir):
            os.symlink(sibling_dir, local_dir)
        print(f"[download_model] Symlinked local model from {sibling_dir} -> {local_dir}")
        return

    print(f"Downloading {model_id} to {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision="main",
        token=os.environ.get("HUGGING_FACE_TOKEN"),
    )
    print("Download complete.")


if __name__ == "__main__":
    download_gemma()
