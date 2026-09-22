"""Verify exported weights and four fixed functional cases; not an accuracy benchmark."""

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from mlx_lm import load

from decision_mlx import configure, predict, sha256


CASES = [
    {"id": "damaged-parcel", "state": "The parcel arrived broken.",
     "question": {"type": "choice", "instructions": "Which condition describes the parcel?",
                  "criteria": {"damaged": "The parcel is broken or damaged.",
                               "intact": "The parcel is undamaged."}}, "expected": "damaged"},
    {"id": "explicit-yes", "state": "The shop is open on Monday.",
     "question": {"type": "noul", "instructions": "Is the shop open on Monday?"},
     "expected": "yes"},
    {"id": "explicit-no", "state": "The shop is closed on Sunday.",
     "question": {"type": "noul", "instructions": "Is the shop open on Sunday?"},
     "expected": "no"},
    {"id": "high-satisfaction", "state": "I am extremely satisfied. Everything was perfect!",
     "question": {"type": "score", "instructions": "Rate the customer's satisfaction.",
                  "criteria": ["Dissatisfied", "Neutral", "Satisfied"]}, "expected": "2"},
]


def verify(system, export, output):
    configure()
    manifest_path = export / "decision_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["system"] != system:
        raise ValueError("The export belongs to a different system")
    for name, expected in manifest["files_sha256"].items():
        if sha256(export / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    model, tokenizer = load(export, tokenizer_config={"trust_remote_code": False})
    calibration = export / "calibration.json" if system == "reflex" else None
    rows = []
    for case in CASES:
        # The reference label is deliberately excluded from the inference request.
        task = SimpleNamespace(**{key: case[key] for key in ("id", "state", "question")})
        probabilities, metadata = predict(model, tokenizer, system, task, calibration)
        valid = (all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
                 and abs(sum(probabilities.values()) - 1) < 1e-5)
        predicted = max(probabilities, key=probabilities.get)
        row = dict(case, probabilities=probabilities, predicted=predicted,
                   valid_distribution=valid, passed=valid and predicted == case["expected"],
                   **metadata)
        rows.append(row)
        print(f"{system}: {case['id']}: {'PASS' if row['passed'] else 'FAIL'} {probabilities}", flush=True)
    report = {"system": system, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Four fixed synthetic functional checks; not a held-out accuracy estimate.",
              "platform": platform.platform(), "python": platform.python_version(),
              "versions": {name: version(name) for name in ("mlx", "mlx-lm", "transformers", "numpy")},
              "checksum_verification": "passed", "manifest": manifest,
              "weight_bytes": sum(p.stat().st_size for p in export.glob("*.safetensors")),
              "passed": sum(row["passed"] for row in rows), "total": len(rows), "cases": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{system}_functional.json").write_text(json.dumps(report, indent=2) + "\n")
    return report["passed"] == report["total"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("semif", "reflex"), required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/decision_exports"))
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.system, args.export, args.output) else 1)
