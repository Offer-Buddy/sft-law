"""扫一遍 checkpoint，挑出 val 上真的最高的那个。

为什么要单独扫：训练日志里只有 eval_loss（token 级交叉熵），没有「50 选 1 选对率」。
两者通常同向，但并不保证同向——`load_best_model_at_end` 按 eval_loss 挑出来的那个，
未必是选对率最高的。所以训完得自己扫一遍，看峰落在哪个 step。

做法：基座只加载一次，用 peft 反复换 adapter，每个 checkpoint 在 val 子集上跑候选打分。
比把 eval_cls.py 循环调用快得多（省掉 N 次加载基座）。

用法：
    # 扫整个输出目录，val 抽 300 条（先粗筛）
    python scripts/sweep_checkpoints.py --dir /root/autodl-tmp/saves/qwen3-1.7b/cls --limit 300

    # 只扫指定的几个
    python scripts/sweep_checkpoints.py --ckpt .../checkpoint-574 .../checkpoint-861

    # 粗筛出峰之后，把那两三个用全量 val 再比一次
    python scripts/sweep_checkpoints.py --ckpt .../checkpoint-861 .../checkpoint-1148 --limit 0

输出：终端一张 step → accuracy 的表 + ASCII 柱状图，同时写 sweep_report.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_cls import DEFAULT_LABELS, DEFAULT_MODEL, DEFAULT_PROMPT, build_prompt_ids, load_labels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = str(ROOT / "data" / "cls" / "val.jsonl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="逐个 checkpoint 在 val 上打分，找出峰值")
    p.add_argument("--dir", default=None, help="训练输出目录，里面是 checkpoint-* 子目录")
    p.add_argument("--ckpt", nargs="*", default=None, help="直接指定若干 checkpoint 目录")
    p.add_argument("--model", default=DEFAULT_MODEL, help="基座模型路径或 HF id")
    p.add_argument("--data", default=DEFAULT_DATA, help="val 集（question/label 或 query/response）")
    p.add_argument("--labels-from", default=DEFAULT_LABELS, help="标签来源（label_stats.json）")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="与训练时一致的指令前缀")
    p.add_argument("--limit", type=int, default=300, help="val 抽多少条；0 = 全量")
    p.add_argument("--out", default=str(ROOT / "data" / "cls" / "sweep_report.json"))
    return p.parse_args()


def step_of(path: Path) -> int:
    """checkpoint-1147 → 1147；不是这个命名就排到最后。"""
    tail = path.name.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else 10**9


def collect_ckpts(args: argparse.Namespace) -> list[Path]:
    if args.ckpt:
        paths = [Path(x).expanduser().resolve() for x in args.ckpt]
    elif args.dir:
        paths = [p for p in Path(args.dir).expanduser().resolve().iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    else:
        sys.exit("要么给 --dir <输出目录>，要么给 --ckpt <目录...>")
    paths = sorted(paths, key=step_of)
    if not paths:
        sys.exit("一个 checkpoint 都没找到——先确认训练真的存了快照")
    missing = [str(p) for p in paths if not (p / "adapter_config.json").exists()]
    if missing:
        sys.exit("这些目录里没有 adapter_config.json，不是 LoRA 快照：\n  " + "\n  ".join(missing))
    return paths


def load_val(path: str, limit: int) -> list[dict]:
    """两种格式都收：评测用的 question/label，和训练集的 query/response。"""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "question" in r and "label" in r:
                rows.append({"question": r["question"], "label": r["label"]})
            elif "query" in r and "response" in r:
                q = r["query"]
                # 训练样本的 query 是「指令\n\n问题：xxx」，切出问题本体
                rows.append({"question": q.split("问题：", 1)[-1].strip(), "label": r["response"]})
            else:
                sys.exit(f"既不是 question/label 也不是 query/response：{line[:80]}")

    if not limit or limit >= len(rows):
        return rows
    # 固定 seed 抽样：不同 checkpoint 必须比在同一批样本上，否则数字没有可比性
    import random
    return random.Random(0).sample(rows, limit)


def score(model, tokenizer, rows, labels, label_ids, eos, pad_id, prompt) -> tuple[float, float]:
    """和 eval_cls.py 完全同一套：50 候选 + eos + 按 token 数归一化。返回 (accuracy, 平均分差)。"""
    import torch

    hit = 0
    gaps: list[float] = []
    with torch.inference_mode():
        for row in rows:
            prefix = build_prompt_ids(tokenizer, prompt, row["question"])
            seqs = [prefix + label_ids[lab] + ([eos] if eos is not None else []) for lab in labels]
            maxlen = max(len(s) for s in seqs)
            padded = [s + [pad_id] * (maxlen - len(s)) for s in seqs]
            input_ids = torch.tensor(padded, device=model.device)
            attention_mask = (input_ids != pad_id).long()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)

            scores: dict[str, float] = {}
            offset = len(prefix) - 1
            for row_i, lab in enumerate(labels):
                ids = label_ids[lab] + ([eos] if eos is not None else [])
                total = 0.0
                for j, tid in enumerate(ids):
                    total += logprobs[row_i, offset + j, tid].item()
                scores[lab] = total / len(ids)

            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            if ranked[0][0] == row["label"]:
                hit += 1
            gaps.append(ranked[0][1] - ranked[1][1])

    n = len(rows)
    return hit / n, sum(gaps) / n


def bar(v: float, vmax: float, width: int = 24) -> str:
    if vmax <= 0:
        return ""
    return "█" * max(1, round(v / vmax * width))


def main() -> None:
    args = parse_args()
    ckpts = collect_ckpts(args)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        sys.exit(f"缺依赖（{exc}）：pip install torch transformers peft accelerate")

    labels = load_labels(args.labels_from)
    rows = load_val(args.data, args.limit)
    print(f"val {len(rows)} 条 · 标签 {len(labels)} 类 · {len(ckpts)} 个 checkpoint · 基座 {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    label_ids = {lab: tokenizer(lab, add_special_tokens=False).input_ids for lab in labels}
    eos = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # 基座只加载这一次，后面全是换 adapter 权重
    model = PeftModel.from_pretrained(base, str(ckpts[0]), adapter_name=ckpts[0].name)
    for ck in ckpts[1:]:
        model.load_adapter(str(ck), adapter_name=ck.name)
    model.eval()

    results = []
    for ck in ckpts:
        model.set_adapter(ck.name)
        acc, gap = score(model, tokenizer, rows, labels, label_ids, eos, pad_id, args.prompt)
        results.append({"checkpoint": ck.name, "step": step_of(ck), "accuracy": round(acc, 4), "gap": round(gap, 3), "path": str(ck)})
        print(f"  {ck.name:<22} acc={acc:.4f}  首选与次选差={gap:.3f}")

    best = max(results, key=lambda r: r["accuracy"])
    vmax = max(r["accuracy"] for r in results)

    print("\n" + "=" * 60)
    print(f"{'checkpoint':<20} {'acc':>7}  {'gap':>6}  曲线")
    print("-" * 60)
    for r in results:
        mark = " ←最佳" if r is best else ""
        print(f"{r['checkpoint']:<20} {r['accuracy']:>7.4f}  {r['gap']:>6.3f}  {bar(r['accuracy'], vmax)}{mark}")
    print("=" * 60)

    # 峰在最后一个：说明还没学够（或正踩在上升段），不是在过拟合
    if best["step"] == max(r["step"] for r in results):
        print("峰落在最后一个 checkpoint：还在往上走，可以加一轮看看。")
    else:
        later = [r for r in results if r["step"] > best["step"]]
        drop = best["accuracy"] - max(r["accuracy"] for r in later)
        print(f"峰落在中间（{best['checkpoint']}）：后面的 step 掉了 {drop:.4f}，这是过拟合的形状，用它。")
    print(f"注意：这里是在 val 上挑的，报成绩还得回 test 跑一次 —— 两边别混。")

    args_out = Path(args.out)
    args_out.parent.mkdir(parents=True, exist_ok=True)
    args_out.write_text(
        json.dumps({"n": len(rows), "best": best, "all": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"报告已写入 {args_out}")


if __name__ == "__main__":
    main()
