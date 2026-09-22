"""Paired MLX source-precision / 4-bit evaluation, with isolated worker processes."""

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(args):
    import mlx.core as mx
    import psutil
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from decision_mlx import configure, load_baseline, predict, sha256
    from verify_decision_exports import CASES

    configure()
    if args.worker == "baseline":
        model, tokenizer, _ = load_baseline("semif", {"base": args.source})
    else:
        manifest = json.loads((args.model / "decision_manifest.json").read_text())
        if manifest["system"] != "semif":
            raise ValueError("Expected a SemIf export")
        for name, expected in manifest["files_sha256"].items():
            if sha256(args.model / name) != expected:
                raise ValueError(f"Export checksum mismatch: {name}")
        model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": False})
    rows = json.loads((args.output / "tasks.json").read_text())
    warmup = SimpleNamespace(**{k: CASES[0][k] for k in ("id", "state", "question")})
    predict(model, tokenizer, "semif", warmup)
    mx.reset_peak_memory()
    results = []
    print(f"{args.worker}: {len(rows)} tasks, MLX Metal, batch 1", flush=True)
    for index, row in enumerate(rows, 1):
        task = SimpleNamespace(**{k: row[k] for k in ("id", "state", "question")})
        mx.synchronize()
        start = time.perf_counter()
        probs, metadata = predict(model, tokenizer, "semif", task)
        elapsed = time.perf_counter() - start
        if set(probs) != set(row["labels"]) or not all(np.isfinite(p) for p in probs.values()):
            raise ValueError(f"Invalid probabilities for {row['id']}")
        if abs(sum(probs.values()) - 1) > 1e-5 or any(p < 0 or p > 1 for p in probs.values()):
            raise ValueError("Invalid probability distribution")
        predicted = max(probs, key=probs.get)
        results.append(dict(id=row["id"], expected=str(row["expected"]), predicted=predicted,
                            correct=predicted == str(row["expected"]), probabilities=probs,
                            latency_ms=elapsed * 1000, **metadata))
        if index % max(1, len(rows) // 8) == 0 or index == len(rows):
            correct = sum(r["correct"] for r in results)
            print(f"  {index}/{len(rows)}: {correct/index:.1%} running accuracy", flush=True)
    params = tree_flatten(model.parameters())
    save(args.output / f"{args.worker}.json", {
        "variant": args.worker, "platform": platform.platform(), "python": platform.python_version(),
        "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "physical_memory_gib": psutil.virtual_memory().total / 2**30,
        "versions": {name: version(name) for name in ("mlx", "mlx-lm", "transformers", "numpy")},
        "parameter_bytes": sum(v.nbytes for _, v in params),
        "parameter_dtypes": sorted({str(v.dtype) for _, v in params}),
        "peak_mlx_gib": mx.get_peak_memory() / 2**30,
        "post_run_rss_gib": psutil.Process().memory_info().rss / 2**30,
        "results": results,
    })


def report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [json.loads((output / f"{name}.json").read_text()) for name in ("baseline", "quantized")]
    a, b = [run["results"] for run in runs]
    for left, right in zip(a, b, strict=True):
        for key in ("id", "expected", "token_ids_sha256", "prompt_sha256"):
            if left[key] != right[key]:
                raise ValueError(f"Paired input mismatch: {key}")
    delta = np.array([int(y["correct"]) - int(x["correct"]) for x, y in zip(a, b)])
    rng = np.random.default_rng(42)
    boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1) * 100
    summaries = []
    for run in runs:
        rows = run["results"]
        summaries.append({"variant": run["variant"], "correct": sum(r["correct"] for r in rows),
                          "total": len(rows), "accuracy_percent": 100 * np.mean([r["correct"] for r in rows]),
                          "median_latency_ms": float(np.median([r["latency_ms"] for r in rows])),
                          "input_tokens_per_second": sum(r["input_tokens"] for r in rows) / (sum(r["latency_ms"] for r in rows) / 1000),
                          "decisions_per_second": len(rows) / (sum(r["latency_ms"] for r in rows) / 1000),
                          "peak_mlx_gib": run["peak_mlx_gib"],
                          "parameter_gib": run["parameter_bytes"] / 2**30})
    paired = {"quantized_minus_baseline_pp": float(delta.mean() * 100),
              "paired_bootstrap_95_pp": np.percentile(boot, [2.5, 97.5]).tolist(),
              "regressions": [x["id"] for x, d in zip(a, delta) if d == -1],
              "improvements": [x["id"] for x, d in zip(a, delta) if d == 1]}
    save(output / "summary.json", {"measured": summaries, "paired": paired})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, label in zip(axes, ("accuracy_percent", "median_latency_ms", "peak_mlx_gib"),
                              ("Accuracy (%)", "Median latency (ms)", "Peak MLX allocation (GiB)")):
        ax.bar(["Source precision", "4-bit"], [r[key] for r in summaries], color=["#536B8E", "#238A75"])
        ax.set_ylabel(label)
        ax.set_title(label)
        if key == "accuracy_percent":
            ax.set_ylim(0, 100)
    fig.suptitle(f"SemIf — paired MLX evaluation ({len(a)} public JevBench items)")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# Measured paired MLX results", "", "| Variant | Correct / total | Accuracy | Median latency | Peak MLX | Input tokens/s | Decisions/s |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summaries:
        lines.append(f"| {r['variant']} | {r['correct']}/{r['total']} | {r['accuracy_percent']:.2f}% | "
                     f"{r['median_latency_ms']:.1f} ms | {r['peak_mlx_gib']:.3f} GiB | "
                     f"{r['input_tokens_per_second']:.2f} | {r['decisions_per_second']:.3f} |")
    lines += ["", f"Quantized minus baseline: {paired['quantized_minus_baseline_pp']:.2f} pp; "
              f"paired bootstrap 95% interval: {paired['paired_bootstrap_95_pp']} pp.", "",
              f"Regressions: {len(paired['regressions'])}; improvements: {len(paired['improvements'])}.", "",
              "![Paired MLX accuracy and resources](comparison.png)", "",
              "Public development data, not a held-out test. Related items are treated as independent by the bootstrap.",
              "Timing includes prompt construction, tokenization and synchronized inference; excludes loading and warmup.",
              "Peak MLX includes live weights; RSS is a post-run snapshot. Do not add overlapping memory counters."]
    (output / "benchmark_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"measured": summaries, "paired": paired}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/semif-4b-mlx-4bit")
    parser.add_argument("--source", help="Pinned source snapshot; otherwise download it")
    parser.add_argument("--output", type=Path, default=ROOT / "results/paired")
    parser.add_argument("--samples", type=int, default=231, help="First N examples in the fixed shuffled order")
    parser.add_argument("--report-only", action="store_true", help="Build the report from two completed worker JSON files")
    parser.add_argument("--worker", choices=("baseline", "quantized"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.report_only:
        report(args.output)
        return
    if args.worker:
        worker(args)
        return
    from create_model import checked_source, source_path
    from decision_mlx import BASE_REVISION, MLX_LM_REVISION, sha256

    if args.output.exists():
        parser.error("Output already exists; choose a fresh directory")
    rows = json.loads((ROOT / "data/tasks.json").read_text())
    if not 1 <= args.samples <= len(rows):
        parser.error(f"samples must be between 1 and {len(rows)}")
    args.source = checked_source(args.source) if args.source else source_path()
    manifest = json.loads((args.model / "decision_manifest.json").read_text())
    if manifest["system"] != "semif" or manifest["quantization"]["bits"] != 4:
        raise ValueError("This benchmark expects the SemIf MLX 4-bit export")
    if manifest["base_revision"] != BASE_REVISION or manifest["mlx_lm_revision"] != MLX_LM_REVISION:
        raise ValueError("Export revisions do not match the evaluation runtime/source")
    save(args.output / "tasks.json", rows[:args.samples])
    save(args.output / "protocol.json", {
        "base_revision": BASE_REVISION, "mlx_lm_revision": MLX_LM_REVISION,
        "tasks_sha256": sha256(ROOT / "data/tasks.json"), "export_manifest": manifest,
        "seed": 42, "n": args.samples, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": args.source, "scope": "Public JevBench, not held-out; no tuning or fitting in this run.",
        "protocol": "Same token IDs and readout; sequential isolated processes; source BF16/FP32 then 4-bit; one independent warmup; batch 1.",
    })
    for variant in ("baseline", "quantized"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", variant,
                        "--source", args.source, "--model", str(args.model.resolve()),
                        "--output", str(args.output.resolve())], check=True)
    report(args.output)


if __name__ == "__main__":
    main()
