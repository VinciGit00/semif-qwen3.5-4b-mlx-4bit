"""Text-only MLX conversion and typed readout for pinned SemIf/reflex models."""

import gc
import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import quantize_model, save_config, save_model

BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER_REVISION = "9df1cbbbc1837ba494530a59d3d9d345688ed250"
MLX_LM_REVISION = "a63e24c389382619eb6d9af656e3b46024be217a"


def configure():
    if not mx.metal.is_available():
        raise RuntimeError("Apple Silicon Metal is required; conversion must run on the Mac mini.")
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 2**20)
    mx.set_memory_limit(12 * 2**30)
    mx.random.seed(42)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_baseline(system, paths):
    """Keep the original BF16/FP32 dtypes; fuse reflex's FP32 LoRA before quantizing."""
    configure()
    model, tokenizer = load(paths["base"], tokenizer_config={"trust_remote_code": False})
    config = json.loads((Path(paths["base"]) / "config.json").read_text())
    if system == "reflex":
        adapter_dir = Path(paths["adapter"])
        spec = json.loads((adapter_dir / "adapter_config.json").read_text())
        if any(spec.get(k) for k in ("use_dora", "use_rslora", "fan_in_fan_out", "rank_pattern", "alpha_pattern")):
            raise ValueError("This converter only supports the pinned ordinary LoRA adapter.")
        tensors = mx.load(str(adapter_dir / "adapter_model.safetensors"))
        params = dict(tree_flatten(model.parameters()))
        merged = 0
        for name in sorted(tensors):
            if not name.endswith(".lora_A.weight"):
                continue
            stem = name.removesuffix(".lora_A.weight")
            target = stem.removeprefix("base_model.model.")
            target = target.replace("model.language_model.", "language_model.model.") + ".weight"
            a, b = tensors[name], tensors[stem + ".lora_B.weight"]
            weight = params[target]
            assert a.shape[0] == spec["r"] and (b.shape[0], a.shape[1]) == weight.shape
            # Match PEFT: add the FP32 update, then round once to the base dtype.
            fused = (weight.astype(mx.float32) + (b @ a) * (spec["lora_alpha"] / spec["r"])).astype(weight.dtype)
            mx.eval(fused)
            model.load_weights([(target, fused)], strict=False)
            params[target] = fused
            merged += 1
        if merged != 32 or len(tensors) != 64:
            raise ValueError(f"Unexpected LoRA coverage: {merged} layers, {len(tensors)} tensors")
    model.eval()
    mx.eval(model.parameters())
    return model, tokenizer, config


def last_logits(model, ids, cache=None):
    # Project only the final hidden state, avoiding a sequence x vocabulary allocation.
    lm = model.language_model
    hidden = lm.model(mx.array([ids]), cache=cache)[:, -1:, :]
    logits = (lm.model.embed_tokens.as_linear(hidden) if lm.args.tie_word_embeddings
              else lm.lm_head(hidden))
    return logits[0, 0].astype(mx.float32)


def predict(model, tokenizer, system, task, calibration_path=None):
    if system == "semif":
        from jevbench.adapters.semif_direct import SemIfDirectAdapter
        from semif_phase1.core import softmax
        from semif_phase1.direct import encode_prompt

        row = SemIfDirectAdapter().build_request(task)
        ids, slots, prompt_hash = encode_prompt(tokenizer, row, 4096)
        selected = last_logits(model, ids)[mx.array(slots)].tolist()
        probs = dict(zip([o["id"] for o in row["options"]], softmax(selected)))
        if task.question["type"] == "noul":
            probs = {"yes": probs["true"], "no": probs["false"]}
        token_groups = [ids]
    else:
        from reflex import SystemOneRequest
        from reflex.prompt import PromptFormat, build_branches
        from reflex.readout import Calibration, merge_branches, to_answer

        request = SystemOneRequest(state=task.state, questions={"q": task.question})
        fmt = PromptFormat(chat=True, no_think=True)
        branch = build_branches("q", request.questions["q"], fmt, permutations=1)[0]
        prefix_text = fmt.prefix(task.state)
        prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix = tokenizer.encode(branch.text, add_special_tokens=False)
        if len(prefix) + len(suffix) > 4096:
            raise ValueError("Input exceeds 4096 tokens; no truncation is allowed.")
        slots = [tokenizer.encode(label, add_special_tokens=False) for label in branch.labels]
        if not all(len(slot) == 1 for slot in slots):
            raise ValueError("Every readout label must be one token.")
        cache = make_prompt_cache(model)
        model.language_model.model(mx.array([prefix]), cache=cache)
        mx.eval([entry.state for entry in cache])
        selected = last_logits(model, suffix, cache)[mx.array([s[0] for s in slots])].tolist()
        cal = Calibration.load(calibration_path)
        answer = to_answer(branch.kind, merge_branches(branch.kind, [(branch, np.array(selected))], cal),
                           request.questions["q"]).model_dump()
        probs = ({"yes": answer["noul"], "no": 1 - answer["noul"]}
                 if branch.kind == "noul" else answer["probabilities"])
        token_groups = [prefix, suffix]
        prompt_hash = hashlib.sha256((prefix_text + branch.text).encode()).hexdigest()
    mx.synchronize()
    return probs, {"prompt_sha256": prompt_hash, "input_tokens": sum(map(len, token_groups)),
                   "token_ids_sha256": hashlib.sha256(json.dumps(token_groups).encode()).hexdigest(),
                   "option_logits": selected}


def export_quantized(system, paths, destination, checks):
    """Save standard MLX weights, reload in a fresh model, and verify predictions."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an existing export: {destination}")
    model, tokenizer, config = load_baseline(system, paths)
    baseline_bytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    dtypes = sorted({str(v.dtype) for _, v in tree_flatten(model.parameters())})
    model, config = quantize_model(model, config, group_size=64, bits=4, mode="affine")
    mx.eval(model.parameters())
    destination.mkdir(parents=True)
    save_model(destination, model)
    save_config(config, destination / "config.json")
    tokenizer.save_pretrained(destination)
    calibration = Path(paths["adapter"]) / "calibration.json" if system == "reflex" else None
    if calibration:
        shutil.copyfile(calibration, destination / "calibration.json")
    expected = [predict(model, tokenizer, system, task, calibration)[0] for task in checks]
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    model, tokenizer = load(destination, tokenizer_config={"trust_remote_code": False})
    actual = [predict(model, tokenizer, system, task, calibration)[0] for task in checks]
    error = max(abs(a[k] - b[k]) for a, b in zip(expected, actual) for k in a)
    if error > 1e-5:
        raise AssertionError(f"Export/reload probability mismatch: {error}")
    manifest = {"system": system, "format": "MLX Safetensors", "text_only": True,
                "base_revision": BASE_REVISION, "adapter_revision": ADAPTER_REVISION if calibration else None,
                "mlx_lm_revision": MLX_LM_REVISION, "baseline_dtypes": dtypes,
                "baseline_parameter_bytes": baseline_bytes, "quantization": config["quantization"],
                "reload_max_probability_error": error,
                "files_sha256": {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()}}
    (destination / "decision_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
