"""Laya on a pre-compiled Core ML model (models/coreml/, set up by fetch_coreml.py).

Sequences are built with laya's own build_sequence, then mapped onto the model's fixed-shape inputs:
    input_ids      int32   [1, L]
    attention_mask int32   [1, L]
    marker_map     float32 [1, max_options, L]   one-hot [MASK] position per option
    question_type  float32 [1, 3]                one-hot choice / score / noul
Probabilities are recomputed from the logits with laya's temperature rule, like ONNXAgent does.
"""

import json
import os

import numpy as np

UNITS = {"all": "ALL", "gpu": "CPU_AND_GPU", "ane": "CPU_AND_NE", "cpu": "CPU_ONLY"}


class _Tokenizer:
    """The slice of a transformers tokenizer that laya's build_sequence uses, on plain HF tokenizers.

    Special-token ids come from meta.json, the same way the Rust port does it.
    """

    def __init__(self, path: str, meta: dict):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(path)
        self._tok.no_truncation()
        self._tok.no_padding()
        self.mask_token = meta["mask_token"]
        self.mask_token_id = meta["mask_id"]
        self.cls_token_id = meta["cls_id"]
        self.sep_token_id = meta["sep_id"]
        self.pad_token_id = meta["pad_id"]

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = self._tok.encode(text, add_special_tokens=add_special_tokens).ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


class CoreMLLaya:
    def __init__(self, model_dir: str, units: str = "all"):
        import coremltools as ct

        with open(os.path.join(model_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.tok = _Tokenizer(os.path.join(model_dir, "tokenizer.json"), self.meta)
        self.model = ct.models.CompiledMLModel(
            os.path.join(model_dir, "model.mlmodelc"), compute_units=getattr(ct.ComputeUnit, UNITS[units]))
        self.length = self.meta["max_len"]

    def _run(self, state, qdef):
        from laya.common import QTYPES, build_sequence, temp_bucket
        from laya.onnx_agent import ONNXAgent

        q = ONNXAgent._to_internal(qdef)
        ids, markers = build_sequence(self.tok, state, q, self.length, self.meta["head_max_len"])
        k = len(markers)
        if k > self.meta["max_options"]:
            raise ValueError("%d options, the model supports at most %d" % (k, self.meta["max_options"]))
        n, qt = len(ids), QTYPES[q["t"]]
        input_ids = np.full((1, self.length), self.meta["pad_id"], np.int32)
        input_ids[0, :n] = ids
        attention = np.zeros((1, self.length), np.int32)
        attention[0, :n] = 1
        marker_map = np.zeros((1, self.meta["max_options"], self.length), np.float32)
        marker_map[0, np.arange(k), markers] = 1.0
        question_type = np.zeros((1, 3), np.float32)
        question_type[0, qt] = 1.0
        out = self.model.predict({"input_ids": input_ids, "attention_mask": attention,
                                  "marker_map": marker_map, "question_type": question_type})
        t = self.meta["temperature_by_options"].get(temp_bucket(qt, k), self.meta["temperature"][qt])
        z = out["logits"][0, :k].astype(np.float64) / t
        p = np.exp(z - z.max())
        return q, p / p.sum()

    def predict(self, state, questions: dict) -> dict:
        """Returns {qid: {"choice": ..., "probabilities": {...}}}, like the answers of laya's predict()."""
        answers = {}
        for qid, qdef in questions.items():
            q, p = self._run(state, qdef)
            keys = list(q["crit"].keys()) if q["t"] == "choice" else [str(i) for i in range(len(p))]
            answers[qid] = {
                "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
            }
        return answers
