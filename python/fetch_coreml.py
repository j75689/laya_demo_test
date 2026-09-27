"""Download a pre-converted Core ML Laya model and set up models/coreml/ for both runners.

Model: https://huggingface.co/FluidInference/laya-coreml, an independent Core ML conversion of
laya-multilingual (mmBERT-base, 322M). It is a different checkpoint from the English
ModernBERT-large model used by the ONNX path, so its decisions differ from that path.

Writes:
    models/coreml/model.mlmodelc   symlink to the downloaded bucket
    models/coreml/tokenizer.json   multilingual tokenizer
    models/coreml/meta.json        same schema as models/meta.json, plus max_options

Usage:
    python python/fetch_coreml.py                   # L256 fp16 bucket (fits the snake prompts)
    python python/fetch_coreml.py --verify          # also compare against PyTorch laya-multilingual
    python python/fetch_coreml.py --bucket 512 --precision e8
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "models", "coreml")
COREML_REPO = "FluidInference/laya-coreml"
COREML_REVISION = "7b8d7a2b7e28e746c6ecaad44bbcd5cf251a4fcc"
SOURCE_REPO = "convaiinnovations/laya"


def main():
    parser = argparse.ArgumentParser(description="Set up the FluidInference Core ML Laya model")
    parser.add_argument("--bucket", type=int, choices=[128, 256, 512, 1024], default=256,
                        help="fixed sequence length of the model")
    parser.add_argument("--precision", choices=["fp16", "e8"], default="fp16",
                        help="e8 stores the embedding table as int8 (smaller, near-identical accuracy)")
    parser.add_argument("--verify", action="store_true",
                        help="compare against PyTorch laya-multilingual (downloads its ~640 MB weights)")
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    from huggingface_hub import snapshot_download

    name = "laya_multilingual_%s_L%d_options32.mlmodelc" % (args.precision, args.bucket)
    print("Downloading %s ..." % name)
    coreml_dir = snapshot_download(COREML_REPO, revision=COREML_REVISION,
                                   allow_patterns=["config.json", "tokenizer.json", name + "/**"])
    with open(os.path.join(coreml_dir, "config.json")) as f:
        coreml_cfg = json.load(f)

    patterns = ["multilingual/rl_agent_config.json", "multilingual/tokenizer/*"]
    if args.verify:
        patterns += ["multilingual/model.safetensors", "multilingual/encoder/*"]
    source_dir = os.path.join(snapshot_download(SOURCE_REPO, allow_patterns=patterns), "multilingual")

    from laya.agent import _load_tokenizer
    from laya.common import clamp_temperature

    with open(os.path.join(source_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = _load_tokenizer(os.path.join(source_dir, "tokenizer"), cfg)

    os.makedirs(OUT, exist_ok=True)
    link = os.path.join(OUT, "model.mlmodelc")
    if os.path.islink(link):
        os.unlink(link)
    os.symlink(os.path.join(coreml_dir, name), link)
    # copyfile, not copy: the HF cache file is read-only and its mode must not come along
    tok_path = os.path.join(OUT, "tokenizer.json")
    if os.path.exists(tok_path):
        os.remove(tok_path)
    shutil.copyfile(os.path.join(coreml_dir, "tokenizer.json"), tok_path)
    meta = {
        "cls_id": tok.cls_token_id,
        "sep_id": tok.sep_token_id,
        "mask_id": tok.mask_token_id,
        "pad_id": tok.pad_token_id,
        "mask_token": tok.mask_token,
        # The model has a fixed length, so sequences are built (and truncated) to the bucket size
        "max_len": args.bucket,
        "head_max_len": cfg.get("head_max_len", 256),
        "max_options": coreml_cfg["max_options"],
        "temperature": [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])],
        "temperature_by_options": {k: clamp_temperature(v)
                                   for k, v in cfg.get("temperature_by_options", {}).items()},
        "source": "%s@%s/%s" % (COREML_REPO, COREML_REVISION[:12], name),
    }
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Wrote models/coreml/ (model.mlmodelc -> %s)" % name)

    if args.verify:
        verify(source_dir)


def verify(source_dir: str) -> None:
    import numpy as np
    from laya.agent import Agent

    from coreml_laya import CoreMLLaya
    from snake import Game, rule_move, state_text

    with open(os.path.join(ROOT, "shared", "question.json"), encoding="utf-8") as f:
        questions = json.load(f)
    states = []
    for seed in range(40):
        g = Game(10, seed)
        while g.alive and g.steps < 60:
            states.append(state_text(g))
            g.step(rule_move(g))
    states = states[::25][:40]

    print("Loading PyTorch laya-multilingual ...")
    ref = Agent(source_dir, device="cpu", compile=False)
    ref.model.float().eval()
    model = CoreMLLaya(OUT, units="all")
    qid = next(iter(questions))
    agree, worst, times = 0, 0.0, []
    for s in states:
        want = ref.predict(s, questions)["answers"][qid]["probabilities"]
        t = time.perf_counter()
        got = model.predict(s, questions)[qid]["probabilities"]
        times.append((time.perf_counter() - t) * 1000)
        agree += int(max(want, key=want.get) == max(got, key=got.get))
        worst = max(worst, max(abs(want[k] - got[k]) for k in want))
    print("CoreML vs PyTorch: same decision %d/%d, max prob error %.4f, p50 %.1f ms" % (
        agree, len(states), worst, float(np.median(times))))


if __name__ == "__main__":
    main()
