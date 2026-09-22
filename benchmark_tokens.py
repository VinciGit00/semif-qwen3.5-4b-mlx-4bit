"""Measure MLX prompt processing and text generation tokens/s separately from Jev decisions."""

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

PROMPTS = [
    "Explain how a refrigerator moves heat in three detailed paragraphs.",
    "Describe how to organize a small software project in three detailed paragraphs.",
    "Explain the water cycle and its main stages in three detailed paragraphs.",
]


def worker(args):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from decision_mlx import configure, load_baseline, sha256

    configure()
    if args.worker == "baseline":
        model, tokenizer, _ = load_baseline("semif", {"base": args.source})
    else:
        manifest = json.loads((args.model / "decision_manifest.json").read_text())
        for name, expected in manifest["files_sha256"].items():
            if sha256(args.model / name) != expected:
                raise ValueError(f"Checksum mismatch: {name}")
        model, tokenizer = load(args.model)

    def generate(prompt, limit):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        text = ""
        start = time.perf_counter()
        for response in stream_generate(model, tokenizer, encoded, max_tokens=limit,
                                        sampler=make_sampler(temp=0)):
            text += response.text
        mx.synchronize()
        return {"prompt": prompt, "output": text, "prompt_token_ids": encoded,
                "prompt_tokens": response.prompt_tokens, "prompt_tps": response.prompt_tps,
                "generation_tokens": response.generation_tokens,
                "generation_tps": response.generation_tps,
                "wall_seconds": time.perf_counter() - start,
                "finish_reason": response.finish_reason}

    generate("Say hello.", 16)
    rows = []
    for i, prompt in enumerate(PROMPTS, 1):
        rows.append(generate(prompt, args.max_tokens))
        print(f"{args.worker} {i}/{len(PROMPTS)}: {rows[-1]['generation_tps']:.2f} generation tokens/s", flush=True)
    result = {"variant": args.worker, "platform": platform.platform(),
              "versions": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
              "rows": rows}
    (args.output / f"{args.worker}.json").write_text(json.dumps(result, indent=2) + "\n")


def report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [json.loads((output / f"{v}.json").read_text()) for v in ("baseline", "quantized")]
    for a, b in zip(runs[0]["rows"], runs[1]["rows"], strict=True):
        if a["prompt_token_ids"] != b["prompt_token_ids"]:
            raise ValueError("Prompt tokens differ between variants")
    summaries = []
    for run in runs:
        row = {"variant": run["variant"]}
        for kind in ("prompt", "generation"):
            total = sum(r[f"{kind}_tokens"] for r in run["rows"])
            seconds = sum(r[f"{kind}_tokens"] / r[f"{kind}_tps"] for r in run["rows"])
            row[f"{kind}_tokens"] = total
            row[f"{kind}_tokens_per_second"] = total / seconds
        summaries.append(row)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kind in zip(axes, ("prompt", "generation")):
        values = [row[f"{kind}_tokens_per_second"] for row in summaries]
        bars = ax.bar(["Source precision", "4-bit"], values, color=["#536B8E", "#238A75"])
        ax.bar_label(bars, fmt="%.1f")
        ax.set_ylabel("Tokens per second")
        ax.set_title(f"{kind.title()} throughput")
        ax.set_ylim(0, max(values) * 1.2)
    fig.suptitle("Qwen3.5-4B MLX — warmed greedy generation, 3 fixed prompts")
    fig.tight_layout()
    fig.savefig(output / "tokens_per_second.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/semif-4b-mlx-4bit"))
    parser.add_argument("--source")
    parser.add_argument("--output", type=Path, default=Path("results/tokens"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--worker", choices=("baseline", "quantized"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        from create_model import checked_source, source_path
        from decision_mlx import BASE_REVISION, MLX_LM_REVISION

        if args.output.exists() or args.max_tokens < 1:
            parser.error("Choose a new output directory and a positive max-tokens value")
        args.source = checked_source(args.source) if args.source else source_path()
        manifest = json.loads((args.model / "decision_manifest.json").read_text())
        if manifest["base_revision"] != BASE_REVISION or manifest["mlx_lm_revision"] != MLX_LM_REVISION:
            raise ValueError("Checkpoint revisions do not match the runtime/source")
        args.output.mkdir(parents=True)
        (args.output / "protocol.json").write_text(json.dumps({
            "base_revision": BASE_REVISION, "mlx_lm_revision": MLX_LM_REVISION,
            "export_manifest": manifest, "prompts": PROMPTS, "max_tokens": args.max_tokens,
            "seed": 42, "temperature": 0, "thinking": False,
            "scope": "Generation speed only, not SemIf decision accuracy. Separate from public JevBench.",
            "timing": "MLX-LM prompt_tps and generation_tps; weighted by token counts over three warmed requests. Model load, warmup and chat formatting excluded; output lengths can differ.",
        }, indent=2) + "\n")
        for variant in ("baseline", "quantized"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", variant,
                            "--source", args.source, "--model", str(args.model.resolve()),
                            "--output", str(args.output.resolve()), "--max-tokens", str(args.max_tokens)], check=True)
        report(args.output)
