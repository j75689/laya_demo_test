"""Python runner: snake game driven by Laya decisions.

Backends:
    torch  official laya.Agent (PyTorch), --device cpu|mps
    onnx   official laya.ONNXAgent (ONNX Runtime), --provider cpu|coreml
    rule   rule-based baseline (no model)

Examples:
    python python/run.py --backend onnx --render
    python python/run.py --backend torch --device mps --episodes 5
    python python/run.py --backend rule --render --delay 0.05
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snake import ORDER, Game, render, rule_move, state_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_questions() -> dict:
    with open(os.path.join(ROOT, "shared", "question.json"), encoding="utf-8") as f:
        return json.load(f)


def local_model_dir() -> str:
    from huggingface_hub import snapshot_download

    patterns = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
    try:
        return snapshot_download("convaiinnovations/laya", allow_patterns=patterns, local_files_only=True)
    except Exception:
        return snapshot_download("convaiinnovations/laya", allow_patterns=patterns)


class LayaDecider:
    def __init__(self, agent, questions: dict):
        self.agent = agent
        self.questions = questions
        self.qid = next(iter(questions))

    def decide(self, g: Game):
        ans = self.agent.predict(state_text(g), self.questions)["answers"][self.qid]
        return ans["choice"], ans["probabilities"]


class PaddedSession:
    """Wraps an ORT session for a static-shape model: right-pads input_ids/attention_mask to its fixed length."""

    def __init__(self, session, pad_to: int, pad_id: int):
        self.session = session
        self.pad_to = pad_to
        self.pad_id = pad_id

    def run(self, output_names, feeds):
        import numpy as np

        n = feeds["input_ids"].shape[1]
        if n > self.pad_to:
            raise ValueError("input is %d tokens, longer than the model's static length %d" % (n, self.pad_to))
        extra = ((0, 0), (0, self.pad_to - n))
        feeds = dict(feeds,
                     input_ids=np.pad(feeds["input_ids"], extra, constant_values=self.pad_id),
                     attention_mask=np.pad(feeds["attention_mask"], extra, constant_values=0))
        return self.session.run(output_names, feeds)


class RuleDecider:
    def decide(self, g: Game):
        return rule_move(g), None


def build_decider(args):
    """Return (decider, display name)."""
    if args.backend == "rule":
        return RuleDecider(), "rule"

    import warnings

    warnings.filterwarnings("ignore")
    import laya

    questions = load_questions()
    model_dir = local_model_dir()
    if args.backend == "torch":
        if args.threads:
            import torch

            torch.set_num_threads(args.threads)
        agent = laya.Agent(model_dir, device=args.device)
        return LayaDecider(agent, questions), "torch-%s" % args.device

    import onnxruntime as ort
    from laya.onnx_agent import ONNXAgent

    onnx_path = os.path.join(ROOT, args.onnx)
    opts = ort.SessionOptions()
    if args.threads:
        opts.intra_op_num_threads = args.threads
    providers = ["CPUExecutionProvider"]
    if args.provider == "coreml":
        providers.insert(0, ("CoreMLExecutionProvider", coreml_options(onnx_path, args.coreml_units)))

    # The official ONNXAgent builds its own session and only picks CUDA or CPU. Swap in ours while it
    # initialises, so the model is loaded once with the providers and threads asked for.
    real_session = ort.InferenceSession
    ort.InferenceSession = lambda path, **_: real_session(path, opts, providers=providers)
    try:
        agent = ONNXAgent(model_dir, onnx_path=onnx_path)
    finally:
        ort.InferenceSession = real_session

    name = "onnx-%s(%s)" % (args.provider, os.path.basename(args.onnx))
    seq_dim = agent.session.get_inputs()[0].shape[1]
    if isinstance(seq_dim, int):
        agent.session = PaddedSession(agent.session, seq_dim, agent.tok.pad_token_id)
    return LayaDecider(agent, questions), name


COREML_UNITS = {"all": "ALL", "gpu": "CPUAndGPU", "ane": "CPUAndNeuralEngine", "cpu": "CPUOnly"}


def coreml_options(onnx_path: str, units: str) -> dict:
    """MLProgram is the CoreML format that takes nearly the whole Laya graph; NeuralNetwork leaves a third on CPU.

    Compiling takes about a minute, so the result is cached. ONNX Runtime does not notice a re-exported
    model, so the cache directory name includes the model's mtime.
    """
    stem = os.path.basename(onnx_path).rsplit(".onnx", 1)[0]
    cache = os.path.join(ROOT, "models", "coreml_cache",
                         "%s-%d-%s" % (stem, int(os.path.getmtime(onnx_path)), COREML_UNITS[units]))
    os.makedirs(cache, exist_ok=True)
    return {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": COREML_UNITS[units],
        "ModelCacheDirectory": cache,
    }


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = max(0, math.ceil(p / 100 * len(sorted_vals)) - 1)
    return sorted_vals[idx]


def fmt_probs(probs):
    if not probs:
        return ""
    return "(" + " | ".join("%s %.2f" % (d, probs[d]) for d in ORDER) + ")"


def main():
    parser = argparse.ArgumentParser(description="Python snake + Laya")
    parser.add_argument("--backend", choices=["torch", "onnx", "rule"], default="onnx")
    parser.add_argument("--device", default="cpu", help="torch backend: cpu or mps")
    parser.add_argument("--onnx", default="models/laya.onnx", help="onnx backend: model path, relative to the project root")
    parser.add_argument("--provider", choices=["cpu", "coreml"], default="cpu")
    parser.add_argument("--coreml-units", choices=sorted(COREML_UNITS), default="all",
                        help="coreml provider: compute units (gpu = CPU+GPU, ane = CPU+Neural Engine)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for inference (0 = library default)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=3, help="untimed inferences before measuring")
    parser.add_argument("--render", action="store_true", help="draw the game in the terminal")
    parser.add_argument("--delay", type=float, default=0.0, help="extra pause per step in seconds, for watching")
    parser.add_argument("--trace", help="write every decision as JSONL, to diff against the Rust runner")
    parser.add_argument("--json", action="store_true", help="print a one-line JSON summary at the end")
    args = parser.parse_args()

    t0 = time.perf_counter()
    decider, name = build_decider(args)
    load_s = time.perf_counter() - t0

    warm = Game(args.size, args.seed)
    for _ in range(args.warmup):
        decider.decide(warm)

    trace = open(args.trace, "w", encoding="utf-8") if args.trace else None
    latencies, scores, deaths = [], [], Counter()

    for ep in range(args.episodes):
        g = Game(args.size, args.seed + ep)
        while g.alive and g.steps < args.max_steps:
            t = time.perf_counter()
            move, probs = decider.decide(g)
            ms = (time.perf_counter() - t) * 1000
            latencies.append(ms)
            if trace:
                trace.write(json.dumps({"ep": ep, "step": g.steps, "move": move, "probs": probs}) + "\n")
            g.step(move)
            if args.render:
                avg = sum(latencies) / len(latencies)
                sys.stdout.write("\x1b[H\x1b[2J")
                print("[python / %s]  episode %d/%d  step %d  score %d" % (
                    name, ep + 1, args.episodes, g.steps, g.score))
                print(render(g))
                print("decision: %-5s %s" % (move, fmt_probs(probs)))
                print("latency : %.2f ms (avg %.2f ms)" % (ms, avg))
                if not g.alive:
                    print("GAME OVER: %s" % g.death)
                    time.sleep(1.0)
            if args.delay:
                time.sleep(args.delay)
        scores.append(g.score)
        deaths[g.death or "max_steps"] += 1

    if trace:
        trace.close()

    s = sorted(latencies)
    mean = sum(s) / len(s) if s else 0.0
    print()
    print("=== python / %s ===" % name)
    print("model load   : %.2f s" % load_s)
    print("episodes     : %d   scores %s   avg %.2f" % (args.episodes, scores, sum(scores) / len(scores)))
    print("game over    : %s" % ", ".join("%s %d" % kv for kv in sorted(deaths.items())))
    print("decisions    : %d" % len(s))
    print("latency (ms) : mean %.3f   p50 %.3f   p95 %.3f   min %.3f" % (
        mean, percentile(s, 50), percentile(s, 95), s[0] if s else 0.0))
    print("throughput   : %.1f decisions/s" % (1000 / mean if mean else 0.0))
    if args.json:
        print(json.dumps({
            "impl": "python", "backend": name, "load_s": round(load_s, 3), "scores": scores,
            "decisions": len(s), "mean_ms": round(mean, 6), "p50_ms": round(percentile(s, 50), 6),
            "p95_ms": round(percentile(s, 95), 6),
        }))


if __name__ == "__main__":
    main()
