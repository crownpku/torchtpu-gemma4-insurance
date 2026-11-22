#!/usr/bin/env bash
set -euo pipefail

# Local-First Orchestrator (Rule #4: Always edit locally, tar, upload to Cloud TPU VM, run, and pull results back)
export PROJECT_ID="${PROJECT_ID:-gemma4-tunix-insurance-202604}"
export ZONE="${ZONE:-us-east5-a}"
export TPU_NAME="${TPU_NAME:-torchtpu-gemma4}"
export ACCEL_TYPE="${ACCEL_TYPE:-v6e-1}"
export RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv6e}"

if [ -f .env ]; then
    HUGGING_FACE_TOKEN=$(grep HUGGING_FACE_TOKEN .env | cut -d '=' -f2 | tr -d ' ')
else
    HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-}"
fi

echo "[Local] 1. Packaging local workspace into project_code.tar.gz..."
tar -czf project_code.tar.gz \
    pyproject.toml \
    SPEC.md \
    ANTIGRAVITY.md \
    gemma4_torchtpu.py \
    prepare_data.py \
    download_model.py \
    train_dry_run.py \
    train_tpu.py \
    bench_execution_modes.py \
    export_model.py \
    eval_perplexity.py \
    eval_tpu.py \
    compare_models.py \
    plot_metrics.py \
    tpu_setup_and_train.sh

echo "[Cloud] 2. Ensuring TPU VM ($TPU_NAME in $ZONE) is provisioned..."
if ! gcloud compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
    echo "Provisioning $TPU_NAME ($ACCEL_TYPE) in $ZONE..."
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --accelerator-type="$ACCEL_TYPE" \
        --version="$RUNTIME_VERSION" \
        --spot || \
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --accelerator-type="$ACCEL_TYPE" \
        --version="$RUNTIME_VERSION"
fi

echo "[Cloud] 3. Uploading project_code.tar.gz to $TPU_NAME..."
gcloud compute tpus tpu-vm scp project_code.tar.gz "$TPU_NAME:~/project_code.tar.gz" \
    --project="$PROJECT_ID" \
    --zone="$ZONE"

echo "[Cloud] 4. Executing tpu_setup_and_train.sh on remote TPU VM..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --command="bash tpu_setup_and_train.sh '$HUGGING_FACE_TOKEN'"

echo "[Local] 5. Syncing results/ and logs back to local workspace..."
mkdir -p results
gcloud compute tpus tpu-vm scp --recurse "$TPU_NAME:~/results/*" ./results/ \
    --project="$PROJECT_ID" \
    --zone="$ZONE"
gcloud compute tpus tpu-vm scp "$TPU_NAME:~/training.log" ./remote_training.log \
    --project="$PROJECT_ID" \
    --zone="$ZONE" || true
