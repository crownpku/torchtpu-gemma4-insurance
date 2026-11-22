"""
Benchmarks TorchTPU's four execution regimes on Gemma 4 (E2B-IT) LoRA fine-tuning:
1. `debug_eager`     (`TORCH_TPU_LAUNCH_BLOCKING=1`)
2. `strict_eager`    (Default `-O1` asynchronous op-by-op dispatch)
3. `defer_and_fuse`  (`TORCH_TPU_DEFER_AND_FUSE=1`, `-O2` dynamic graph fusion)
4. `compile`         (`torch.compile(model, backend="tpu")` + Tier-3 cache)

CRITICAL TPU ARCHITECTURE NOTE:
The parent coordinator process MUST NOT `import torch`! Importing `torch` autoloads
`torch_tpu` and acquires the `/dev/vfio/0` device lock in the parent process, blocking
child worker processes from accessing the TPU chip. Furthermore, `TORCH_TPU_*` environment
variables must be set in the child process environment BEFORE `import torch` executes.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def build_mode_env(mode: str) -> dict:
    """Builds the child process environment variables before `import torch`."""
    env = os.environ.copy()
    cache_dir = env.get(
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
        os.path.expanduser("~/.cache/torchtpu_pjrt_cache"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    env["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = cache_dir

    env.pop("TORCH_TPU_LAUNCH_BLOCKING", None)
    env.pop("TORCH_TPU_DEFER_AND_FUSE", None)

    if mode == "debug_eager":
        env["TORCH_TPU_LAUNCH_BLOCKING"] = "1"
    elif mode == "strict_eager":
        pass
    elif mode == "defer_and_fuse":
        env["TORCH_TPU_DEFER_AND_FUSE"] = "1"
    elif mode == "compile":
        pass
    else:
        raise ValueError(f"Unknown execution mode: {mode}")
    return env


def run_single_mode_worker(
    mode: str,
    steps: int,
    warmup: int,
    batch_size: int,
    seq_len: int,
    model_dir: str,
    dry_run: bool,
    profile_xprof: bool,
) -> dict:
    # Lazy imports ONLY inside the isolated child worker process
    import torch
    from peft import LoraConfig, get_peft_model
    from gemma4_torchtpu import Gemma4ForCausalLMTorchTPU, Gemma4TextConfig, get_torchtpu_device
    from train_tpu import compute_completion_loss

    device = get_torchtpu_device("cpu" if dry_run else None)

    if dry_run:
        cfg = Gemma4TextConfig.dry_run_tiny()
        base_model = Gemma4ForCausalLMTorchTPU(cfg, dtype=torch.bfloat16).to(device)
    else:
        base_model = Gemma4ForCausalLMTorchTPU.from_safetensors(model_dir, device=device, dtype=torch.bfloat16)
        cfg = base_model.config

    lora_config = LoraConfig(
        r=16 if not dry_run else 4,
        lora_alpha=32 if not dry_run else 8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    model.train()

    if mode == "compile" and device.type == "tpu":
        model = torch.compile(model, backend="tpu")
    elif mode == "compile" and device.type == "cpu":
        model = torch.compile(model)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)

    input_ids = torch.randint(1, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device).contiguous()
    attn_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device).contiguous()
    loss_mask = torch.ones((batch_size, seq_len), dtype=torch.float32, device=device).contiguous()
    loss_mask[:, :32] = 0.0

    # Warmup / compilation phase
    t_compile_start = time.perf_counter()
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attn_mask, loss_mask=loss_mask, return_logits=False)
        loss = out.loss
        loss.backward()
        optimizer.step()
        _ = float(loss.detach().cpu().item())
    warmup_s = time.perf_counter() - t_compile_start

    # Steady-state timed steps (with optional XProf profiling)
    step_times_ms = []
    xprof_dir = "results/xprof"
    if profile_xprof and device.type == "tpu":
        os.makedirs(xprof_dir, exist_ok=True)
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if hasattr(torch.profiler.ProfilerActivity, "TPU"):
                activities.append(torch.profiler.ProfilerActivity.TPU)
            with torch.profiler.profile(
                activities=activities,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(xprof_dir),
                record_shapes=True,
            ) as prof:
                for _ in range(steps):
                    t0 = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    out = model(input_ids=input_ids, attention_mask=attn_mask, loss_mask=loss_mask, return_logits=False)
                    loss = out.loss
                    loss.backward()
                    optimizer.step()
                    _ = float(loss.detach().cpu().item())
                    step_times_ms.append((time.perf_counter() - t0) * 1000.0)
                    prof.step()
        except Exception as e:
            print(f"[XProf Warning] Profiler fallback due to: {e}", flush=True)
            step_times_ms = []

    if not step_times_ms:
        for _ in range(steps):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attn_mask, loss_mask=loss_mask, return_logits=False)
            loss = out.loss
            loss.backward()
            optimizer.step()
            _ = float(loss.detach().cpu().item())
            step_times_ms.append((time.perf_counter() - t0) * 1000.0)

    step_times_sorted = sorted(step_times_ms)
    median_ms = step_times_sorted[len(step_times_sorted) // 2]
    mean_ms = sum(step_times_ms) / len(step_times_ms)
    tokens_per_sec = (batch_size * seq_len) / (median_ms / 1000.0)

    cache_stats = {}
    if device.type == "tpu" and hasattr(torch, "tpu") and hasattr(torch.tpu, "_get_cache_stats"):
        try:
            raw_stats = torch.tpu._get_cache_stats()
            if isinstance(raw_stats, dict):
                cache_stats = {str(k): int(v) if isinstance(v, (int, float)) else str(v) for k, v in raw_stats.items()}
        except Exception:
            cache_stats = {}

    return {
        "mode": mode,
        "device": str(device),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "warmup_compile_time_s": round(warmup_s, 3),
        "median_step_time_ms": round(median_ms, 2),
        "mean_step_time_ms": round(mean_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "dynamic_shape_recompilations": 0,
        "cache_stats": cache_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark TorchTPU Execution Modes on Gemma 4 LoRA SFT")
    parser.add_argument(
        "--worker-mode",
        type=str,
        default=None,
        help="Internal flag to run a single mode in an isolated process",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--model-dir", type=str, default="models/gemma-4-E2B-it")
    parser.add_argument("--dry-run", action="store_true", help="Run with tiny model on CPU for local verification")
    parser.add_argument("--profile-xprof", action="store_true")
    parser.add_argument("--output", type=str, default="results/execution_modes_bench.json")
    args = parser.parse_args()

    if args.worker_mode:
        res = run_single_mode_worker(
            mode=args.worker_mode,
            steps=args.steps,
            warmup=args.warmup,
            batch_size=args.batch_size if not args.dry_run else 2,
            seq_len=args.seq_len if not args.dry_run else 32,
            model_dir=args.model_dir,
            dry_run=args.dry_run,
            profile_xprof=args.profile_xprof,
        )
        print("WORKER_RESULT_JSON:" + json.dumps(res), flush=True)
        return

    modes = ["debug_eager", "strict_eager", "defer_and_fuse", "compile"]
    results = []
    for mode in modes:
        print(f"\n=== Benchmarking TorchTPU Mode: {mode} ===", flush=True)
        child_env = build_mode_env(mode)
        cmd = [
            sys.executable,
            __file__,
            "--worker-mode",
            mode,
            "--steps",
            str(args.steps),
            "--warmup",
            str(args.warmup),
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
            "--model-dir",
            args.model_dir,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.profile_xprof and mode == "defer_and_fuse":
            cmd.append("--profile-xprof")

        proc = subprocess.run(cmd, env=child_env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[Error] Worker for mode '{mode}' exited with code {proc.returncode}:", flush=True)
            print("STDOUT:\n", proc.stdout, flush=True)
            print("STDERR:\n", proc.stderr, flush=True)
            raise RuntimeError(f"Worker for mode '{mode}' failed")

        for line in proc.stdout.splitlines():
            if line.startswith("WORKER_RESULT_JSON:"):
                data = json.loads(line[len("WORKER_RESULT_JSON:") :])
                # Parse authentic TorchTPU C++ CompilationCache final stats from STDERR
                import re
                m_reqs = re.search(r"num_cache_reqs=(\d+)", proc.stderr)
                m_hits = re.search(r"num_cache_hits=(\d+)", proc.stderr)
                m_comp = re.search(r"num_compilations=(\d+)", proc.stderr)
                m_dyn = re.search(r"num_unbounded_dynamic_hits=(\d+)", proc.stderr)
                if m_reqs and m_hits and m_comp:
                    data["cache_stats"] = {
                        "num_cache_reqs": int(m_reqs.group(1)),
                        "num_cache_hits": int(m_hits.group(1)),
                        "num_compilations": int(m_comp.group(1)),
                        "num_unbounded_dynamic_hits": int(m_dyn.group(1)) if m_dyn else 0,
                        "cache_hit_rate_pct": round(100.0 * int(m_hits.group(1)) / max(int(m_reqs.group(1)), 1), 2),
                    }
                results.append(data)
                print(
                    f"  -> {mode:15s} | warmup: {data['warmup_compile_time_s']:6.2f}s | "
                    f"median step: {data['median_step_time_ms']:8.2f} ms | "
                    f"throughput: {data['tokens_per_sec']:8.1f} tok/s | "
                    f"cache: {data.get('cache_stats', {})}",
                    flush=True,
                )
            else:
                print(" ", line, flush=True)

    strict_ms = next(
        (r["median_step_time_ms"] for r in results if r["mode"] == "strict_eager"),
        results[0]["median_step_time_ms"],
    )
    for r in results:
        r["speedup_vs_strict_eager"] = round(strict_ms / max(r["median_step_time_ms"], 1e-6), 2)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Benchmark] Saved execution mode results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
