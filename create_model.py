"""Download the pinned source and export SemIf MLX 4-bit with a reload check."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import snapshot_download

from decision_mlx import BASE_REVISION, export_quantized
from verify_decision_exports import CASES


def source_path():
    return snapshot_download(
        "Qwen/Qwen3.5-4B", revision=BASE_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
    )


def checked_source(path):
    """Accept only the pinned HF cache snapshot when bypassing the downloader."""
    source = Path(path).resolve()
    if source.name != BASE_REVISION or not (source / "config.json").is_file():
        raise ValueError(f"Expected a Hugging Face snapshot directory named {BASE_REVISION}")
    return str(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/semif-4b-mlx-4bit"))
    parser.add_argument("--source", type=Path, help="Existing pinned HF snapshot; avoids download")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; select a new directory")
    source = checked_source(args.source) if args.source else source_path()
    checks = [SimpleNamespace(**{key: row[key] for key in ("id", "state", "question")})
              for row in CASES]
    print("Converting SemIf: MLX affine 4-bit, group size 64", flush=True)
    manifest = export_quantized("semif", {"base": source}, args.output, checks)
    print(json.dumps(manifest, indent=2), flush=True)
