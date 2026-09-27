"""Convert a Laya checkpoint (e.g. the fine-tuned models/finetuned) to a Core ML model for --backend coreml.

Produces the same fixed-shape FP16 ML Program format as FluidInference/laya-coreml, through the
same adapter (python/coreml_export.py), so both runners load it with --coreml-dir.

Needs the conversion environment: coremltools 9.0 only works with torch 2.7, not the torch in .venv.
    python3 -m venv .venv-coreml && .venv-coreml/bin/pip install -r requirements-coreml.txt

Usage:
    .venv-coreml/bin/python python/convert_coreml.py                        # models/finetuned -> models/finetuned_coreml
    .venv-coreml/bin/python python/convert_coreml.py --model-dir models/finetuned --out models/finetuned_coreml --length 256
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Convert a Laya checkpoint to Core ML")
    parser.add_argument("--model-dir", default="models/finetuned", help="Laya checkpoint, relative to the project root")
    parser.add_argument("--out", default="models/finetuned_coreml", help="output directory for --coreml-dir")
    parser.add_argument("--length", type=int, default=256, help="fixed sequence length")
    parser.add_argument("--max-options", type=int, default=32, help="option slots")
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    import coremltools as ct
    import numpy as np
    import torch
    from laya.agent import Agent

    from coreml_export import LayaExport

    model_dir = os.path.join(ROOT, args.model_dir)
    out = os.path.join(ROOT, args.out)
    length, slots = args.length, args.max_options

    torch.backends.mha.set_fastpath_enabled(False)
    agent = Agent(model_dir, device="cpu", compile=False)
    agent.model.float().eval()
    export = LayaExport(agent.model, length, slots).eval()

    # Any valid input works for tracing: the graph has no data-dependent control flow
    marker_map = torch.zeros(1, slots, length)
    marker_map[0, 0, 5] = marker_map[0, 1, 9] = 1.0
    example = (torch.zeros(1, length, dtype=torch.int32), torch.ones(1, length, dtype=torch.int32),
               marker_map, torch.tensor([[1.0, 0.0, 0.0]]))
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(export, example)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, length), dtype=np.int32),
            ct.TensorType(name="marker_map", shape=(1, slots, length), dtype=np.float32),
            ct.TensorType(name="question_type", shape=(1, 3), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="probabilities", dtype=np.float32),
            ct.TensorType(name="action_probabilities", dtype=np.float32),
        ],
    )
    print("converted in %.0f s" % (time.time() - t0))

    os.makedirs(out, exist_ok=True)
    compiled = os.path.join(out, "model.mlmodelc")
    if os.path.exists(compiled):
        shutil.rmtree(compiled)
    with tempfile.TemporaryDirectory() as tmp:
        package = os.path.join(tmp, "model.mlpackage")
        converted.save(package)
        ct.utils.compile_model(package, compiled)

    tok = agent.tok
    meta = {
        "cls_id": tok.cls_token_id,
        "sep_id": tok.sep_token_id,
        "mask_id": tok.mask_token_id,
        "pad_id": tok.pad_token_id,
        "mask_token": tok.mask_token,
        "max_len": length,
        "head_max_len": agent.cfg.get("head_max_len", 256),
        "max_options": slots,
        "temperature": agent.temperature,
        "temperature_by_options": agent.temperature_by_options,
        "source": "%s converted with coremltools %s" % (args.model_dir, ct.__version__),
    }
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    shutil.copyfile(os.path.join(model_dir, "tokenizer", "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    print("wrote %s (use --backend coreml --coreml-dir %s)" % (args.out, args.out))


if __name__ == "__main__":
    main()
