"""Diff two --trace files (e.g. Python vs Rust) step by step.

Usage:
    python python/compare_traces.py traces/python.jsonl traces/rust.jsonl
"""

import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    same_moves, worst, first_diff = 0, 0.0, None
    for i, (x, y) in enumerate(zip(a, b)):
        if (x["ep"], x["step"]) != (y["ep"], y["step"]):
            first_diff = first_diff or (i, "games diverged", x, y)
            break
        if x["move"] == y["move"]:
            same_moves += 1
        elif first_diff is None:
            first_diff = (i, "different move", x, y)
        if x.get("probs") and y.get("probs"):
            worst = max(worst, max(abs(x["probs"][k] - y["probs"][k]) for k in x["probs"]))

    n = min(len(a), len(b))
    print("steps        : %d vs %d" % (len(a), len(b)))
    print("same move    : %d/%d" % (same_moves, n))
    print("max prob diff: %.4f" % worst)
    if first_diff:
        i, why, x, y = first_diff
        print("first mismatch at line %d (%s):\n  %s\n  %s" % (i + 1, why, x, y))
    ok = first_diff is None and len(a) == len(b)
    print("RESULT       : %s" % ("IDENTICAL" if ok else "DIFFERENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
