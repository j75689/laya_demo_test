"""Fine-tune Laya's decision head to play snake, using the rule baseline as the teacher.

The encoder stays frozen, so its output for every training example is computed once and cached;
only the decision head is trained: 2 transformer layers, the question-type embedding and the
option scorer. That keeps a run to minutes on an Apple Silicon laptop.

Data: games played by the rule baseline with some random safe moves mixed in (so the snake also
sees bad positions). At each kept step the question is built with a random mix of --safe/--hints,
and the target spreads probability evenly over the moves that are safe and eat or approach the food.
Training and validation use different game seeds.

Output: models/finetuned/, a regular Laya checkpoint (loads with laya.Agent, exports with
export_onnx.py --model-dir models/finetuned).

Usage:
    python python/finetune.py                      # laya-multilingual base, 4000 examples (~30 min on an M3)
    python python/finetune.py --base english --epochs 6
"""

import argparse
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def padded_len(n: int) -> int:
    """Round lengths up to a multiple of 64: MPS recompiles its kernels for every new tensor shape."""
    return (n + 63) // 64 * 64


def base_checkpoint(name: str) -> str:
    from huggingface_hub import snapshot_download

    sub = "multilingual/" if name == "multilingual" else ""
    patterns = [sub + p for p in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")]
    try:
        root = snapshot_download("convaiinnovations/laya", allow_patterns=patterns, local_files_only=True)
    except Exception:
        root = snapshot_download("convaiinnovations/laya", allow_patterns=patterns)
    return os.path.join(root, sub.rstrip("/")) if sub else root


def good_moves(g, options):
    """Teacher: safe moves that eat or approach the food; else any safe move; else anything."""
    from snake import DEADLY, DIRS

    hx, hy = g.body[0]
    fx, fy = g.food
    safe = [d for d in options if g.cell_status(d) not in DEADLY]

    def closer(d):
        dx, dy = DIRS[d]
        return abs(hx + dx - fx) + abs(hy + dy - fy) < abs(hx - fx) + abs(hy - fy)

    good = [d for d in safe if g.cell_status(d) == "food" or closer(d)]
    return good or safe or list(options)


def collect(first_seed: int, count: int, base_q: dict, explore: float, keep: float, rng: random.Random):
    """Returns [(state_text, question, target_probs, good_set)]."""
    from snake import DEADLY, ORDER, Game, move_question, rule_move, state_text

    samples, seed = [], first_seed
    while len(samples) < count:
        g = Game(10, seed)
        seed += 1
        while g.alive and g.steps < 300 and len(samples) < count:
            if rng.random() < keep:
                q = move_question(g, base_q, safe=rng.random() < 0.5, hints=rng.random() < 0.5)
                options = list(q["criteria"])
                if len(options) > 1:
                    good = good_moves(g, options)
                    target = [1.0 / len(good) if d in good else 0.0 for d in options]
                    samples.append((state_text(g), q, target, set(good)))
            safe = [d for d in ORDER if g.cell_status(d) not in DEADLY]
            move = rng.choice(safe) if safe and rng.random() < explore else rule_move(g)
            g.step(move)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Laya's decision head on snake")
    parser.add_argument("--base", choices=["multilingual", "english"], default="multilingual")
    parser.add_argument("--train", type=int, default=4000, help="training examples")
    parser.add_argument("--val", type=int, default=800, help="validation examples (separate games)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--explore", type=float, default=0.3, help="share of random safe moves while collecting")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/finetuned")
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    import torch
    from laya.agent import Agent
    from laya.common import QTYPES, build_sequence
    from laya.onnx_agent import ONNXAgent
    from safetensors.torch import load_file, save_file

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    with open(os.path.join(ROOT, "shared", "question.json"), encoding="utf-8") as f:
        qid, base_q = next(iter(json.load(f).items()))

    t0 = time.time()
    train = collect(100_000, args.train, base_q, args.explore, 0.1, rng)
    val = collect(200_000, args.val, base_q, args.explore, 0.1, rng)
    print("collected %d train / %d val examples (%.0f s)" % (len(train), len(val), time.time() - t0))

    src = base_checkpoint(args.base)
    agent = Agent(src, device=args.device, compile=False)
    model = agent.model.float()
    model.eval()
    dev = agent.device

    def encode(samples):
        """Tokenize, then run the frozen encoder once; hidden states are cached in fp16 on the CPU."""
        items = []
        for state, q, target, good in samples:
            qi = ONNXAgent._to_internal(q)
            ids, markers = build_sequence(agent.tok, state, qi, agent.cfg["max_len"], agent.cfg["head_max_len"])
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": target,
                          "good": [i for i, d in enumerate(q["criteria"]) if d in good]})
        order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
        with torch.no_grad():
            for start in range(0, len(order), 32):
                chunk = [items[i] for i in order[start:start + 32]]
                n = padded_len(max(len(it["ids"]) for it in chunk))
                ids = torch.full((len(chunk), n), agent.tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(chunk), n), dtype=torch.long)
                for r, it in enumerate(chunk):
                    ids[r, :len(it["ids"])] = torch.tensor(it["ids"])
                    att[r, :len(it["ids"])] = 1
                h = model.encoder(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state
                for r, it in enumerate(chunk):
                    it["h"] = h[r, :len(it["ids"])].half().cpu()
                if start % 1600 == 0:
                    print("  encoded %d/%d" % (start + len(chunk), len(items)), flush=True)
        return items

    t0 = time.time()
    train_items, val_items = encode(train), encode(val)
    print("encoded with the frozen encoder (%.0f s)" % (time.time() - t0))

    def batch(items):
        n = padded_len(max(len(it["ids"]) for it in items))
        k = 4  # the snake question never has more than 4 options; a fixed size avoids MPS recompiles
        h = torch.zeros((len(items), n, items[0]["h"].shape[-1]))
        att = torch.zeros((len(items), n), dtype=torch.long)
        mpos = torch.zeros((len(items), k), dtype=torch.long)
        mmask = torch.zeros((len(items), k), dtype=torch.bool)
        target = torch.zeros((len(items), k))
        for r, it in enumerate(items):
            h[r, :len(it["ids"])] = it["h"].float()
            att[r, :len(it["ids"])] = 1
            mpos[r, :len(it["markers"])] = torch.tensor(it["markers"])
            mmask[r, :len(it["markers"])] = True
            target[r, :len(it["target"])] = torch.tensor(it["target"])
        qtype = torch.tensor([it["qtype"] for it in items])
        return [x.to(dev) for x in (h, att, mpos, mmask, qtype, target)]

    def head_logits(h, att, mpos, mmask, qtype):
        """DecisionModel.forward after the encoder (laya/common.py)."""
        h = h + model.type_emb(qtype)[:, None, :]
        pad = ~att.bool()
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        m = torch.gather(h, 1, mpos[:, :, None].expand(-1, -1, h.size(-1)))
        return model.scorer(m).squeeze(-1).float().masked_fill(~mmask, -1e4)

    def evaluate(items):
        model.eval()
        hits = 0
        with torch.no_grad():
            for start in range(0, len(items), 64):
                chunk = items[start:start + 64]
                h, att, mpos, mmask, qtype, _ = batch(chunk)
                pred = head_logits(h, att, mpos, mmask, qtype).argmax(-1).tolist()
                hits += sum(p in it["good"] for p, it in zip(pred, chunk))
        return hits / len(items)

    print("zero-shot: picks a good move on %.1f%% of validation examples" % (100 * evaluate(val_items)))

    for p in model.parameters():
        p.requires_grad_(False)
    trained = [model.head, model.type_emb, model.scorer]
    params = [p for mod in trained for p in mod.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train_items)
        total, t0 = 0.0, time.time()
        for start in range(0, len(train_items), args.batch):
            h, att, mpos, mmask, qtype, target = batch(train_items[start:start + args.batch])
            logits = head_logits(h, att, mpos, mmask, qtype)
            # Log score against the teacher distribution: a strictly proper scoring rule, like RLCD's
            loss = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(h)
        print("epoch %d: loss %.4f, validation %.1f%% good moves (%.0f s)" % (
            epoch + 1, total / len(train_items), 100 * evaluate(val_items), time.time() - t0))

    # Save as a regular Laya checkpoint: original files, with the trained tensors swapped in
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    for name in ("tokenizer", "encoder"):
        if os.path.exists(os.path.join(out, name)):
            shutil.rmtree(os.path.join(out, name))
        shutil.copytree(os.path.join(src, name), os.path.join(out, name))
    weights = load_file(os.path.join(src, "model.safetensors"))
    for key, value in model.state_dict().items():
        if key.startswith(("head.", "type_emb.", "scorer.")):
            weights[key] = value.detach().float().cpu().contiguous()
    save_file(weights, os.path.join(out, "model.safetensors"))
    cfg = dict(agent.cfg, finetuned={"task": "snake", "base": args.base, "train": len(train_items),
                                     "epochs": args.epochs, "lr": args.lr, "trained": "head,type_emb,scorer"})
    with open(os.path.join(out, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("wrote %s" % args.out)

    # Round trip: load the saved checkpoint through laya's own API and re-check validation
    check = Agent(out, device=args.device, compile=False)
    subset = val[:200]
    hits = sum(max(ans, key=ans.get) in good for ans, good in (
        (check.predict(s, {qid: q})["answers"][qid]["probabilities"], good) for s, q, _, good in subset))
    print("reloaded with laya.Agent: %.1f%% good moves on %d validation examples" % (
        100 * hits / len(subset), len(subset)))


if __name__ == "__main__":
    main()
