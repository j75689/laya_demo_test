"""Run every runner/backend combination on the same seeds and print one comparison table.

Also diffs each Rust trace against its Python counterpart, to show the Rust port makes
exactly the same decisions.

Usage:
    python python/bench.py                  # all combinations, 3 episodes each
    python python/bench.py --episodes 5 --skip-int8 --skip-coreml
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RUST_BIN = os.path.join(ROOT, "rust", "target", "release", "snake-laya")


def run(label, cmd, trace):
    print("-> %s" % label, flush=True)
    out = subprocess.run(cmd + ["--json", "--trace", trace], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-2000:] + out.stderr[-2000:])
        return None
    return json.loads(out.stdout.strip().splitlines()[-1])


def identical(a, b):
    out = subprocess.run([PY, "python/compare_traces.py", a, b], cwd=ROOT, capture_output=True, text=True)
    return out.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Benchmark every runner/backend combination")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--skip-coreml", action="store_true")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--threads", type=int, default=0, help="passed to both runners (0 = library default)")
    args = parser.parse_args()

    if not os.path.exists(RUST_BIN):
        subprocess.run(["cargo", "build", "--release"], cwd=os.path.join(ROOT, "rust"), check=True)
    os.makedirs(os.path.join(ROOT, "traces"), exist_ok=True)

    common = ["--episodes", str(args.episodes), "--seed", str(args.seed), "--threads", str(args.threads)]
    py = [PY, "python/run.py"] + common
    rs = [RUST_BIN] + common

    # (label, python cmd or None, rust cmd or None, trace name)
    plan = [("rule baseline", py + ["--backend", "rule"], rs + ["--backend", "rule"], "rule")]
    if not args.skip_torch:
        plan.append(("laya torch cpu", py + ["--backend", "torch", "--device", "cpu"], None, "torch_cpu"))
        plan.append(("laya torch mps", py + ["--backend", "torch", "--device", "mps"], None, "torch_mps"))
    plan.append(("laya onnx cpu fp32", py + ["--backend", "onnx"], rs, "onnx_cpu"))
    if not args.skip_coreml:
        plan.append(("laya onnx coreml fp32", py + ["--backend", "onnx", "--provider", "coreml"],
                     rs + ["--provider", "coreml"], "onnx_coreml"))
    static = "models/laya.static192.onnx"
    if not args.skip_coreml and os.path.exists(os.path.join(ROOT, static)):
        plan.append(("laya onnx coreml static192", py + ["--backend", "onnx", "--onnx", static, "--provider", "coreml"],
                     rs + ["--onnx", static, "--provider", "coreml"], "onnx_coreml_static"))
    if not args.skip_coreml and os.path.exists(os.path.join(ROOT, "models", "coreml", "meta.json")):
        plan.append(("laya-multilingual coreml (Fluid)", py + ["--backend", "coreml"],
                     rs + ["--backend", "coreml"], "coreml_fluid"))
        for flags in (["--hints"], ["--safe", "--hints"], ["--safe", "--hints", "--sample"]):
            label = "laya-multilingual coreml (Fluid) " + " ".join(flags)
            plan.append((label, py + ["--backend", "coreml"] + flags, rs + ["--backend", "coreml"] + flags,
                         "coreml_fluid" + "".join(f.strip("-") for f in flags)))
    ft = "models/finetuned"
    if os.path.exists(os.path.join(ROOT, ft, "rl_agent_config.json")):
        if not args.skip_torch:
            for flags in ([], ["--safe"]):
                plan.append(("fine-tuned torch mps " + " ".join(flags),
                             py + ["--backend", "torch", "--device", "mps", "--model-dir", ft] + flags, None,
                             "ft_torch_mps" + "".join(f.strip("-") for f in flags)))
        ft_static = ft + "/laya.static256.onnx"
        if not args.skip_coreml and os.path.exists(os.path.join(ROOT, ft_static)):
            plan.append(("fine-tuned onnx coreml static256 --safe",
                         py + ["--backend", "onnx", "--onnx", ft_static, "--provider", "coreml", "--safe"],
                         rs + ["--onnx", ft_static, "--provider", "coreml", "--safe"], "ft_onnx_coreml_safe"))
    if not args.skip_int8 and os.path.exists(os.path.join(ROOT, "models", "laya.int8.onnx")):
        plan.append(("laya onnx cpu int8", py + ["--backend", "onnx", "--onnx", "models/laya.int8.onnx"],
                     rs + ["--onnx", "models/laya.int8.onnx"], "onnx_int8"))

    rows = []
    for label, py_cmd, rs_cmd, name in plan:
        py_trace, rs_trace = "traces/py_%s.jsonl" % name, "traces/rs_%s.jsonl" % name
        res_py = run("python " + label, py_cmd, py_trace) if py_cmd else None
        res_rs = run("rust   " + label, rs_cmd, rs_trace) if rs_cmd else None
        same = identical(py_trace, rs_trace) if res_py and res_rs else None
        for impl, res in (("python", res_py), ("rust", res_rs)):
            if res:
                rows.append((label, impl, res, same))

    print()
    print("| setup | runner | model load (s) | mean ms | p50 ms | p95 ms | decisions/s | avg score | same moves as python |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for label, impl, r, same in rows:
        avg = sum(r["scores"]) / max(1, len(r["scores"]))
        same_txt = "-" if impl == "python" or same is None else ("yes" if same else "NO")
        print("| %s | %s | %.2f | %.4f | %.4f | %.4f | %.0f | %.2f | %s |" % (
            label, impl, r["load_s"], r["mean_ms"], r["p50_ms"], r["p95_ms"],
            1000 / r["mean_ms"] if r["mean_ms"] else 0.0, avg, same_txt))


if __name__ == "__main__":
    main()
