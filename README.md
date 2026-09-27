# Laya snake demo: Python vs Rust

A snake game where every move is decided by [Laya](https://github.com/NandhaKishorM/laya),
run through two runners so they can be compared:

- **Python** (`python/run.py`) uses the official `laya` package. It can run on PyTorch (CPU or MPS) or on ONNX Runtime (CPU or CoreML).
- **Rust** (`rust/`) is a port of Laya's inference that needs no Python. It uses `ort` (ONNX Runtime) and HF `tokenizers`.

A rule-based player is included as a baseline.

Both runners share the same game engine, RNG and state text. With the same seed they feed
Laya identical input, so their decisions can be compared step by step.

## Layout

```
shared/question.json     Laya question used by both runners
python/snake.py          game engine, state text, rule baseline
python/run.py            Python runner (torch / onnx / rule backends)
python/export_onnx.py    checkpoint -> models/laya.onnx (+ tokenizer.json, meta.json for Rust)
python/compare_traces.py step-by-step diff of two --trace files
python/bench.py          runs every combination and prints one table
rust/src/snake.rs        game engine (line-for-line port of snake.py)
rust/src/laya.rs         Laya inference port (build_sequence, collate, temperature, softmax)
rust/src/main.rs         Rust runner (same flags as run.py)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Downloads the ~800 MB checkpoint on first run, then writes models/
.venv/bin/python python/export_onnx.py          # add --int8 for the quantized model too
.venv/bin/python python/export_onnx.py --skip-export --static 192   # fixed-shape model for CoreML

cd rust && cargo build --release && cd ..
```

`export_onnx.py` checks the ONNX output against PyTorch on inputs of 175 to 512 tokens.
It prints `FP32: max logit error 0.00004, same decision 7/7`.

## Run

```bash
# Watch a game in the terminal
.venv/bin/python python/run.py --backend onnx --render
./rust/target/release/snake-laya --render

.venv/bin/python python/run.py --backend torch --device mps --render

# Apple GPU / Neural Engine through CoreML (first load compiles for ~1 min, then it is cached)
./rust/target/release/snake-laya --onnx models/laya.static192.onnx --provider coreml --render
.venv/bin/python python/run.py --backend onnx --onnx models/laya.static192.onnx --provider coreml --render
./rust/target/release/snake-laya --backend rule --render --delay 0.05

# Prove the Rust port makes the same decisions as the official Python package
.venv/bin/python python/run.py --backend onnx --trace traces/py.jsonl
./rust/target/release/snake-laya --trace traces/rs.jsonl
.venv/bin/python python/compare_traces.py traces/py.jsonl traces/rs.jsonl

# Everything at once
.venv/bin/python python/bench.py
```

Common flags for both runners:

| Flag | Meaning |
|---|---|
| `--episodes` | number of games |
| `--seed` | random seed |
| `--size` | board size |
| `--max-steps` | step limit per game |
| `--warmup` | untimed inferences before measuring |
| `--threads` | CPU threads for inference |
| `--render` | draw the game in the terminal |
| `--delay` | extra pause per step, in seconds |
| `--trace` | write every decision to a JSONL file |
| `--json` | print a one-line JSON summary |

The ONNX backends also take:

| Flag | Meaning |
|---|---|
| `--onnx` | model path |
| `--provider cpu\|coreml` | execution provider |
| `--coreml-units all\|gpu\|ane\|cpu` | CoreML compute units |

A static-shape model (`laya.staticN.onnx`) is detected automatically, and inputs are padded to N tokens.

## Results

These numbers come from an Apple M3 (4 performance + 4 efficiency cores, 24 GB), running 3 games with seed 42.
The machine was not idle during the run: load average was about 10, and macOS `mediaanalysisd` was using two cores.
On a quiet machine an earlier run measured ONNX CPU at about 515 ms (Python) and 548 ms (Rust).
Use absolute numbers only as a rough guide. The Python vs Rust comparison is still fair, because each pair ran back to back under the same load.

| setup | runner | model load (s) | mean ms | p50 ms | p95 ms | avg score | same moves as python |
|---|---|---:|---:|---:|---:|---:|---|
| rule baseline | python | 0.00 | 0.002 | 0.002 | 0.002 | 25.00 | - |
| rule baseline | rust | 0.00 | <0.001 | <0.001 | <0.001 | 25.00 | yes |
| laya torch cpu | python | 6.13 | 496 | 381 | 989 | 0.00 | - |
| laya torch mps | python | 6.79 | 203 | 188 | 272 | 0.00 | - |
| laya onnx cpu fp32 | python | 7.44 | 1315 | 1304 | 1614 | 0.00 | - |
| laya onnx cpu fp32 | rust | 1.28 | 1230 | 1215 | 1401 | 0.00 | yes |
| laya onnx coreml fp32 | python | 13.71 | 2373 | 1919 | 4123 | 0.00 | - |
| laya onnx coreml fp32 | rust | 6.24 | 2037 | 1781 | 4128 | 0.00 | yes |
| laya onnx cpu int8 | python | 5.31 | 441 | 442 | 456 | 0.00 | - |
| laya onnx cpu int8 | rust | 0.91 | 446 | 443 | 475 | 0.00 | yes |

All FP32 backends make identical moves: torch cpu, torch mps, onnx cpu, onnx coreml, and the Rust runner.

## Findings

1. **The Rust port is exact.** It matches the official Python package on every step, with a probability difference of 0.0000 on CPU. Game parity also holds: over 20 rule-based games, both runners produced the same 3958 moves.
2. **Rust does not make Laya faster.** Per decision, Python and Rust are within noise of each other. Almost all the time is spent inside the ModernBERT-large forward pass, which is the same ONNX Runtime C++ code either way.
   Rust's real wins are elsewhere:
   - faster startup (about 1 s vs about 7 s to load)
   - a single binary, with no Python, torch or transformers install
3. **Hardware matters more than language.** PyTorch on MPS (Apple GPU) was the fastest Laya setup, at about 200 ms. CoreML through ONNX Runtime was slower than plain CPU for this model.
4. **INT8 is faster but wrong.** Dynamic INT8 quantization cut latency by roughly 2-3x, but it flips most decisions: only 4 of 7 matched FP32 in the export check. That held even with per-channel weights, MatMul only.
5. **Laya cannot play snake zero-shot.** The four move probabilities sit around 0.23-0.30, with a constant bias toward `right`, which is the current heading. It walked into the wall even when the state said `right = wall (deadly)`. Every game ended with score 0. The rule baseline averaged 25 points at about 0.002 ms per decision.
6. **CoreML needs static shapes.** The dynamic-shape model runs slower on CoreML than on CPU, because ONNX Runtime splits the graph into many small CoreML and CPU pieces.
   A fixed 192-token model changes that. Inputs never exceeded 188 tokens over 200 games.
   With the MLProgram format, CoreML takes 1481 of 1574 nodes and runs at about 150-250 ms per decision, on par with PyTorch MPS. Decisions stay identical, and Rust and Python match step for step.
   The NeuralNetwork format leaves about a third of the graph on CPU and is 3-5x slower.
   Rust sometimes measured faster than Python here, but Python's own overhead is only about 0.7 ms per call. The gap is load noise plus different ONNX Runtime builds (Rust bundles 1.28, pip has 1.30), not the language.
7. **The official ONNX export script only works for one input length.** `scripts/export_onnx.py` uses the TorchScript tracer, which bakes the example sequence length into the head's attention reshape. Any other length fails with a Reshape error. `python/export_onnx.py` uses the `torch.export` (dynamo) exporter instead.

## Notes

- The game loop in both runners is sequential: one Laya call per move, timed end to end. Each timing covers building the state text, tokenizing, running the model and post-processing.
- Each CoreML cache under `models/coreml_cache/` is about 3.3 GB. Delete stale ones freely; they are rebuilt on demand.
- The Rust port supports `choice`, `score` and `noul` questions. It does not support custom `noul` labels or non-string instructions.
