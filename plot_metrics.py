"""
Publication-Ready Chart Generator for the TorchTPU Gemma 4 Insurance SFT Project:
1. `results/training_loss_curve.png`: SFT Training & Validation Loss Trajectory (with optional Tunix overlay)
2. `results/execution_modes_speedup.png`: TorchTPU Execution Modes (Debug vs Strict Eager vs DeferAndFuse vs torch.compile)
"""

import json
import os
import matplotlib.pyplot as plt


def plot_training_curves(
    metrics_file: str = "results/metrics.json",
    tunix_metrics_file: str = "../Tunix_Gemma4_TPU/results/metrics.json",
    output_image: str = "results/training_loss_curve.png",
):
    if not os.path.exists(metrics_file):
        print(f"[Plot] {metrics_file} not found; skipping loss curve plot.")
        return

    with open(metrics_file, "r") as f:
        data = json.load(f)

    train_data = data.get("train_loss", [])
    eval_data = data.get("eval_loss", [])

    train_steps = [item["step"] for item in train_data]
    train_losses = [item["loss"] for item in train_data]
    eval_steps = [item["step"] for item in eval_data]
    eval_losses = [item["loss"] for item in eval_data]

    smoothed_train_losses = []
    ema = train_losses[0] if train_losses else 0.0
    alpha = 0.08
    for loss in train_losses:
        ema = alpha * loss + (1.0 - alpha) * ema
        smoothed_train_losses.append(ema)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(10.5, 6.2), dpi=300)

    color_train = "#2B6CB0"
    color_raw = "#CBD5E0"
    ax1.plot(train_steps, train_losses, color=color_raw, alpha=0.55, linewidth=1.0, label="TorchTPU Train Loss (Raw Steps)")
    ax1.plot(train_steps, smoothed_train_losses, color=color_train, linewidth=2.6, label="TorchTPU Train Loss (EMA Trend)")

    # Overlay Tunix (JAX) baseline EMA if available
    if os.path.exists(tunix_metrics_file):
        with open(tunix_metrics_file, "r") as tf:
            tdata = json.load(tf)
        t_train = tdata.get("train_loss", [])
        if t_train:
            t_steps = [x["step"] for x in t_train]
            t_vals = [x["loss"] for x in t_train]
            t_ema = []
            cur = t_vals[0]
            for v in t_vals:
                cur = alpha * v + (1.0 - alpha) * cur
                t_ema.append(cur)
            ax1.plot(t_steps, t_ema, color="#718096", linewidth=1.8, linestyle=":", label="Tunix (JAX) Train Loss Baseline (EMA)")

    ax1.set_xlabel("Training Step", fontsize=12, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Training Cross-Entropy Loss", fontsize=12, fontweight="bold", color=color_train, labelpad=10)
    ax1.tick_params(axis="y", labelcolor=color_train)

    if eval_losses:
        ax2 = ax1.twinx()
        color_eval = "#D53F8C"
        ax2.plot(
            eval_steps,
            eval_losses,
            color=color_eval,
            marker="o",
            markersize=6,
            linewidth=2.2,
            linestyle="--",
            label="TorchTPU Validation Loss",
        )
        ax2.set_ylabel("Validation Loss (Summed over Eval Split)", fontsize=12, fontweight="bold", color=color_eval, labelpad=10)
        ax2.tick_params(axis="y", labelcolor=color_eval)
        ax2.grid(False)

    plt.title(
        "Gemma 4 (E2B-IT) Supervised Fine-Tuning on InsuranceQA-v2\n(Native PyTorch + PEFT LoRA on Google Cloud TorchTPU)",
        fontsize=13.5,
        fontweight="bold",
        pad=14,
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    if eval_losses:
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=True, framealpha=0.92)
    else:
        ax1.legend(lines_1, labels_1, loc="upper right", frameon=True, framealpha=0.92)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"[Plot] Loss curve saved to {output_image}")


def plot_execution_modes(
    bench_file: str = "results/execution_modes_bench.json",
    output_image: str = "results/execution_modes_speedup.png",
):
    if not os.path.exists(bench_file):
        print(f"[Plot] {bench_file} not found; skipping execution modes chart.")
        return

    with open(bench_file, "r") as f:
        rows = json.load(f)

    label_map = {
        "debug_eager": "Debug Eager\n(LAUNCH_BLOCKING=1)",
        "strict_eager": "Strict Eager\n(Default -O1)",
        "defer_and_fuse": "DeferAndFuse\n(DEFER_AND_FUSE=1)",
        "compile": "torch.compile\n(backend='tpu')",
    }
    labels = [label_map.get(r["mode"], r["mode"]) for r in rows]
    tok_s = [r["tokens_per_sec"] for r in rows]
    step_ms = [r["median_step_time_ms"] for r in rows]
    speedups = [r.get("speedup_vs_strict_eager", 1.0) for r in rows]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    colors = ["#A0AEC0", "#4A5568", "#3182CE", "#38A169"]
    bars = ax1.bar(labels, tok_s, color=colors[: len(labels)], width=0.55, edgecolor="#1A202C", linewidth=1.0)
    ax1.set_ylabel("Training Throughput (tokens / sec)", fontsize=12, fontweight="bold", labelpad=10)
    ax1.set_title(
        "TorchTPU Execution Modes Benchmark — Gemma 4 E2B-IT LoRA SFT\n(Static Shape [B=4, L=1024], Zero Dynamic Recompilations)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    max_y = max(tok_s) if tok_s else 1.0
    ax1.set_ylim(0, max_y * 1.25)

    for bar, ms, sp in zip(bars, step_ms, speedups):
        height = bar.get_height()
        ax1.annotate(
            f"{height:,.0f} tok/s\n({ms:,.1f} ms | {sp:.2f}x)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"[Plot] Execution modes chart saved to {output_image}")


if __name__ == "__main__":
    plot_training_curves()
    plot_execution_modes()
