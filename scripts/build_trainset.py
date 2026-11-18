"""构造 HyDE 训练集：从 DISC-Law Triplet 抽出「问题 -> 法条语体」训练对，切 train/val。

口径（对齐 README，2026-11-18 定）：
- 只用 Triplet（23,331 条），不用 Pair（Pair 无 reference 金标准，只能训问答，不可量化验证）
- 输入 = 从 input 的「<问题>：」后剥出的真实问题（不是塞满法条的 input）
- 目标 = reference 各条法条原文拼接（纯法条语体，不含 output 答案，
  避免「因此/所以」等问答语气污染改写器，也与 prompt 里的「不要写结论」对齐）
- 产出 query/response 两字段：ms-swift 直接 --dataset 吃；LLaMA-Factory 用 --format messages 转
- 默认剔除评测集问题（evalset 400 + book40 + oral40），保证 train/val 不泄漏考卷

用法：
    python scripts/build_trainset.py                     # 全量，产出 train.jsonl / val.jsonl
    python scripts/build_trainset.py --max-samples 500   # 先跑 500 条验证
    python scripts/build_trainset.py --format messages   # 出 LLaMA-Factory 的 messages 格式
    python scripts/build_trainset.py --keep-evalset      # 复现旧版（训练集保留评测样本，会泄漏）
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TRIPLET = BASE_DIR / "data" / "raw" / "DISC-Law-SFT-Triplet-QA-released.jsonl"
DEFAULT_OUTDIR = BASE_DIR / "data" / "sft"
DEFAULT_EVAL_FILES = [
    BASE_DIR / "data" / "eval" / "evalset.jsonl",
    BASE_DIR / "data" / "eval" / "evalset_book40.jsonl",
    BASE_DIR / "data" / "eval" / "evalset_oral.jsonl",
]

SYSTEM_INSTR = (
    "请根据用户法律问题生成一段用于语义检索的假设性法律文档。"
    "要求包含核心法律概念、可能涉及的法律关系、责任类型和检索关键词；"
    "不要给用户建议，不要写结论，只输出检索用文本。"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构造 HyDE 训练集并切 train/val")
    p.add_argument("--triplet-path", default=str(DEFAULT_TRIPLET))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--max-chars", type=int, default=700,
                   help="目标文本截断长度（字），默认 700 覆盖 97.4% 样本")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=0, help="0=全量")
    p.add_argument("--with-answer", action="store_true",
                   help="目标追加 output 答案（复现旧 demo 用，默认关）")
    p.add_argument("--format", choices=["query_response", "messages"], default="query_response")
    p.add_argument("--keep-evalset", action="store_true",
                   help="训练集保留评测集样本（默认剔除，避免评测泄漏）")
    p.add_argument("--eval-files", nargs="*", default=[str(x) for x in DEFAULT_EVAL_FILES],
                   help="评测集文件，用于构建剔除黑名单")
    return p.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_exclude_questions(paths: list[str]) -> set[str]:
    """读评测集的问题文本，作为训练集剔除黑名单。"""
    qs: set[str] = set()
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            continue
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                q = str(json.loads(line).get("question", "")).strip()
            except json.JSONDecodeError:
                continue
            q = re.sub(r"\s+", " ", q).strip()
            if q:
                qs.add(q)
    return qs


def extract_question(text: str) -> str:
    """从 input 里剥出「<问题>：」之后的真实问题，去掉塞在里面的法条。"""
    m = re.search(r"<问题>\s*[：:]\s*(.+)", text, flags=re.S)
    q = m.group(1) if m else text
    return re.sub(r"\s+", " ", q).strip()


def clean(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].strip()


def make_pair(item: dict, max_chars: int, with_answer: bool) -> tuple[str, str] | None:
    question = extract_question(str(item.get("input", "")))
    refs = item.get("reference", [])
    ref_text = " ".join(str(r).strip() for r in refs if str(r).strip())
    target = ref_text
    if with_answer:
        answer = str(item.get("output", "")).strip()
        if answer:
            target = f"{ref_text} {answer}"
    target = clean(target, max_chars)
    if not question or not target:
        return None
    return question, target


def to_record(question: str, target: str, fmt: str) -> dict:
    if fmt == "query_response":
        return {"query": f"{SYSTEM_INSTR}\n\n用户问题：{question}", "response": target}
    return {"messages": [
        {"role": "system", "content": SYSTEM_INSTR},
        {"role": "user", "content": question},
        {"role": "assistant", "content": target},
    ]}


def main() -> None:
    args = parse_args()
    triplet = Path(args.triplet_path)
    if not triplet.exists():
        raise FileNotFoundError(f"Triplet 数据不存在: {triplet}")

    exclude = load_exclude_questions(args.eval_files) if not args.keep_evalset else set()

    pairs: list[tuple[str, str]] = []
    skipped = 0
    excluded = 0
    for item in read_jsonl(triplet):
        pair = make_pair(item, args.max_chars, args.with_answer)
        if pair is None:
            skipped += 1
            continue
        if pair[0] in exclude:
            excluded += 1
            continue
        pairs.append(pair)

    random.seed(args.seed)
    random.shuffle(pairs)
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    n = len(pairs)
    n_val = max(1, int(n * args.val_ratio))
    val = pairs[:n_val]
    train = pairs[n_val:]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.max_samples}" if args.max_samples > 0 else ""
    train_path = outdir / f"train{suffix}.jsonl"
    val_path = outdir / f"val{suffix}.jsonl"

    for path, subset in ((train_path, train), (val_path, val)):
        with open(path, "w", encoding="utf-8") as f:
            for q, t in subset:
                f.write(json.dumps(to_record(q, t, args.format), ensure_ascii=False) + "\n")

    total_lens = sorted(len(q) + len(t) for q, t in pairs)
    print("训练数据已生成:")
    print(f"  train: {train_path}  ({len(train)} 条)")
    print(f"  val:   {val_path}  ({len(val)} 条)")
    print(f"  剔除评测集样本: {excluded}   跳过无效样本: {skipped}")
    print(f"  query+response 总长(字): P50={total_lens[len(total_lens) // 2]}, "
          f"P99={total_lens[int(n * 0.99) - 1]}, max={total_lens[-1]}")
    print(f"  格式: {args.format}")


if __name__ == "__main__":
    main()
