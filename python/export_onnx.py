"""Export the Laya checkpoint to ONNX and write the tokenizer.json / meta.json the Rust port needs.

Adapted from the official scripts/export_onnx.py
(https://github.com/NandhaKishorM/laya/blob/main/scripts/export_onnx.py):
- Uses the torch.export (dynamo) exporter. The official TorchScript trace bakes the example
  sequence length into the head's attention reshape, so any other input length fails.
- Uses a real snake-game input as the example instead of 16 random tokens.
- Verifies ONNX against PyTorch on inputs of several lengths after exporting.
- --int8 additionally writes a dynamically quantized model.
- --static N additionally writes a fixed-shape model, which is what CoreML needs to run fast.

Usage:
    python python/export_onnx.py            # writes models/laya.onnx (+ laya.onnx.data)
    python python/export_onnx.py --int8     # also writes models/laya.int8.onnx
    python python/export_onnx.py --static 192   # also writes models/laya.static192.onnx (CoreML)
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from laya.agent import Agent  # noqa: E402
from laya.common import QTYPES, build_sequence, collate_items  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402

from snake import Game, rule_move, state_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]


def local_model_dir() -> str:
    from huggingface_hub import snapshot_download

    patterns = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
    try:
        return snapshot_download("convaiinnovations/laya", allow_patterns=patterns, local_files_only=True)
    except Exception:
        return snapshot_download("convaiinnovations/laya", allow_patterns=patterns)


def load_questions() -> dict:
    with open(os.path.join(ROOT, "shared", "question.json"), encoding="utf-8") as f:
        return json.load(f)


def make_batch(agent: Agent, state: str, questions: dict) -> dict:
    items = []
    for qdef in questions.values():
        q = ONNXAgent._to_internal(qdef)
        seq, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    return collate_items([items], agent.tok.pad_token_id)


def pad_batch(b: dict, pad_to: int, pad_id: int) -> dict:
    """Right-pad input_ids/attention_mask to a fixed length. Padding is masked out, so logits do not change."""
    n = b["input_ids"].shape[1]
    if n > pad_to:
        raise ValueError("input is %d tokens, longer than the static length %d" % (n, pad_to))
    ids = torch.full((1, pad_to), pad_id, dtype=torch.long)
    att = torch.zeros((1, pad_to), dtype=torch.long)
    ids[:, :n], att[:, :n] = b["input_ids"], b["attention_mask"]
    return dict(b, input_ids=ids, attention_mask=att)


def sample_states() -> list:
    """A few real game states plus one long input, to prove the sequence length is not baked in."""
    g = Game(size=10, seed=7)
    states = []
    while g.alive and len(states) < 6:
        states.append(state_text(g))
        for _ in range(3):
            if g.alive:
                g.step(rule_move(g))
    states.append(" ".join([states[0]] * 5))
    return states


def verify(agent: Agent, onnx_path: str, questions: dict, label: str, pad_to: int = 0) -> None:
    """Compare ONNX logits with PyTorch. With pad_to, inputs are padded for a static-shape model."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst, agree, total = 0.0, 0, 0
    for state in sample_states():
        b = make_batch(agent, state, questions)
        if pad_to and b["input_ids"].shape[1] > pad_to:
            continue
        with torch.no_grad():
            ref, _ = agent.model(*[b[k] for k in INPUT_NAMES])
        ref = ref.numpy()
        fed = pad_batch(b, pad_to, agent.tok.pad_token_id) if pad_to else b
        got = sess.run(["logits"], {k: fed[k].numpy() for k in INPUT_NAMES})[0]
        k = int(b["marker_mask"].sum())
        worst = max(worst, float(np.abs(ref[0, :k] - got[0, :k]).max()))
        agree += int(ref[0, :k].argmax() == got[0, :k].argmax())
        total += 1
        print("  seq_len=%3d  torch=%s  onnx=%s" % (
            b["input_ids"].shape[1], np.round(ref[0, :k], 3), np.round(got[0, :k], 3)))
    print("%s: max logit error %.5f, same decision %d/%d" % (label, worst, agree, total))


def export_fp32(agent: Agent, questions: dict, path: str) -> None:
    b = make_batch(agent, sample_states()[0], questions)
    print("Exporting ONNX (about 30 s)...")
    t0 = time.time()
    batch = torch.export.Dim("batch", min=1, max=64)
    seq = torch.export.Dim("seq", min=16, max=agent.cfg.get("max_len", 512))
    markers = torch.export.Dim("markers", min=2, max=64)
    torch.onnx.export(
        agent.model,
        tuple(b[name] for name in INPUT_NAMES),
        path,
        opset_version=18,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamo=True,
        dynamic_shapes={
            "input_ids": {0: batch, 1: seq},
            "attention_mask": {0: batch, 1: seq},
            "marker_pos": {0: batch, 1: markers},
            "marker_mask": {0: batch, 1: markers},
            "qtype": {0: batch},
        },
        external_data=True,
    )
    print("Wrote %s (%.0f s)" % (path, time.time() - t0))


def export_static(agent: Agent, questions: dict, path: str, pad_to: int) -> None:
    """Fixed-shape export (batch 1, this question's option count, pad_to tokens).

    CoreML only takes over most of the graph when every shape is static; with dynamic shapes
    ONNX Runtime splits the model into many small CoreML/CPU pieces and ends up slower than CPU.
    """
    b = pad_batch(make_batch(agent, sample_states()[0], questions), pad_to, agent.tok.pad_token_id)
    print("Exporting static ONNX, seq_len=%d (about 30 s)..." % pad_to)
    t0 = time.time()
    torch.onnx.export(
        agent.model,
        tuple(b[name] for name in INPUT_NAMES),
        path,
        opset_version=18,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamo=True,
        external_data=True,
    )
    print("Wrote %s (%.0f s)" % (path, time.time() - t0))


def export_int8(fp32_path: str, int8_path: str) -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("Quantizing to INT8 (1-2 min)...")
    # torch.export writes one value_info entry with a wrong shape, which makes the quantizer's
    # shape inference fail. value_info is only a hint, so dropping it does not change the math.
    model = onnx.load(fp32_path)
    del model.graph.value_info[:]
    tmp_path = os.path.join(MODELS, "_prequant.onnx")
    onnx.save(model, tmp_path, save_as_external_data=True, location="_prequant.onnx.data")
    del model
    try:
        # Per-channel MatMul weights only: per-tensor INT8 over every op flipped most decisions.
        quantize_dynamic(tmp_path, int8_path, weight_type=QuantType.QInt8,
                         per_channel=True, op_types_to_quantize=["MatMul"])
    finally:
        for p in (tmp_path, tmp_path + ".data"):
            if os.path.exists(p):
                os.remove(p)


def main():
    parser = argparse.ArgumentParser(description="Export Laya to ONNX for the Python/Rust snake demo")
    parser.add_argument("--int8", action="store_true", help="also write a dynamically quantized INT8 model")
    parser.add_argument("--skip-export", action="store_true", help="reuse an existing models/laya.onnx")
    parser.add_argument("--static", type=int, default=0, metavar="N",
                        help="also write models/laya.staticN.onnx with inputs fixed to N tokens (for CoreML)")
    args = parser.parse_args()

    os.makedirs(MODELS, exist_ok=True)
    model_dir = local_model_dir()
    questions = load_questions()
    fp32_path = os.path.join(MODELS, "laya.onnx")

    print("Loading PyTorch checkpoint: %s" % model_dir)
    agent = Agent(model_dir, device="cpu", compile=False)
    agent.model.float().eval()

    if not args.skip_export:
        export_fp32(agent, questions, fp32_path)

    # The Rust port has no transformers library, so everything it needs goes into plain files
    shutil.copy(os.path.join(model_dir, "tokenizer", "tokenizer.json"), os.path.join(MODELS, "tokenizer.json"))
    tok = agent.tok
    meta = {
        "cls_id": tok.cls_token_id,
        "sep_id": tok.sep_token_id,
        "mask_id": tok.mask_token_id,
        "pad_id": tok.pad_token_id,
        "mask_token": tok.mask_token,
        "max_len": agent.cfg.get("max_len", 512),
        "head_max_len": agent.cfg.get("head_max_len", 192),
        # Same as the official runtime: only temperatures clamped to [0.5, 5.0] are applied
        "temperature": agent.temperature,
        "temperature_by_options": agent.temperature_by_options,
    }
    with open(os.path.join(MODELS, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Wrote models/tokenizer.json and models/meta.json")

    verify(agent, fp32_path, questions, "FP32")

    if args.int8:
        int8_path = os.path.join(MODELS, "laya.int8.onnx")
        export_int8(fp32_path, int8_path)
        verify(agent, int8_path, questions, "INT8")

    if args.static:
        static_path = os.path.join(MODELS, "laya.static%d.onnx" % args.static)
        if not os.path.exists(static_path):
            export_static(agent, questions, static_path, args.static)
        verify(agent, static_path, questions, "STATIC%d" % args.static, pad_to=args.static)


if __name__ == "__main__":
    main()
