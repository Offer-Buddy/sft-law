"""构造「问题 -> 部法名」50 类分类数据集，切 train/val/test。

任务：给定一个法律问题，判断它该适用哪部法律（50 部，单标签）。
标签来源：Triplet 的 reference 里《X》第Y条 的 X。
- 99.7% 单标签；0.3% 多部法取 reference 中出现次数最多的部法名（平局取先出现的）
- 默认剔除评测集问题，避免与 evalset 重叠泄漏

产出（默认 data/cls/）：
    train.jsonl / val.jsonl / test.jsonl    query/response，ms-swift/LLaMA-Factory 直接吃
    label_stats.json                        50 类 + 每类样本数 + 切分统计（课件用）

用法：
    python scripts/build_cls_dataset.py
    python scripts/build_cls_dataset.py --val-ratio 0.1 --test-ratio 0.1
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TRIPLET = BASE_DIR / "data" / "raw" / "DISC-Law-SFT-Triplet-QA-released.jsonl"
DEFAULT_OUTDIR = BASE_DIR / "data" / "cls"
DEFAULT_EVAL_FILES = [
    BASE_DIR / "data" / "eval" / "evalset.jsonl",
    BASE_DIR / "data" / "eval" / "evalset_book40.jsonl",
    BASE_DIR / "data" / "eval" / "evalset_oral.jsonl",
]

CITE = re.compile(r"《([^》]{2,30})》第")
QUESTION = re.compile(r"<问题>\s*[：:]\s*(.+)", re.S)
SYSTEM_INSTR = "请判断以下法律问题应适用哪部法律。只输出法律名称，不要解释。"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构造问题->部法 50 类分类数据集")
    p.add_argument("--triplet-path", default=str(DEFAULT_TRIPLET))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-evalset", action="store_true",
                   help="保留评测集问题（默认剔除，避免泄漏）")
    p.add_argument("--eval-files", nargs="*", default=[str(x) for x in DEFAULT_EVAL_FILES])
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
    m = QUESTION.search(text)
    q = m.group(1) if m else text
    return re.sub(r"\s+", " ", q).strip()


def majority_law(refs: list) -> tuple[str | None, int]:
    """取 reference 里出现次数最多的部法名；返回 (部法名, 涉及部法数)。"""
    names = []
    for r in refs:
        m = CITE.search(str(r))
        if m:
            names.append(m.group(1))
    if not names:
        return None, 0
    counter = Counter(names)
    law = counter.most_common(1)[0][0]
    return law, len(counter)


def to_record(question: str, label: str) -> dict:
    return {"query": f"{SYSTEM_INSTR}\n\n问题：{question}", "response": label}


def main() -> None:
    args = parse_args()
    triplet = Path(args.triplet_path)
    if not triplet.exists():
        raise FileNotFoundError(f"Triplet 数据不存在: {triplet}")

    exclude = load_exclude_questions(args.eval_files) if not args.keep_evalset else set()

    samples: list[tuple[str, str]] = []
    skipped = 0
    excluded = 0
    duplicated = 0
    multi_law = 0
    seen: set[str] = set()
    for item in read_jsonl(triplet):
        question = extract_question(str(item.get("input", "")))
        law, n_laws = majority_law(item.get("reference") or [])
        if not question or not law:
            skipped += 1
            continue
        if question in exclude:
            excluded += 1
            continue
        if question in seen:
            duplicated += 1
            continue
        seen.add(question)
        if n_laws > 1:
            multi_law += 1
        samples.append((question, law))

    random.seed(args.seed)
    random.shuffle(samples)

    n = len(samples)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    test = samples[:n_test]
    val = samples[n_test:n_test + n_val]
    train = samples[n_test + n_val:]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("val", val), ("test", test)):
        with open(outdir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for q, lab in subset:
                f.write(json.dumps(to_record(q, lab), ensure_ascii=False) + "\n")

    label_counts = Counter(lab for _, lab in samples)
    stats = {
        "total": n,
        "num_labels": len(label_counts),
        "excluded_evalset": excluded,
        "skipped": skipped,
        "duplicated_question": duplicated,
        "multi_law_samples": multi_law,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "labels": dict(label_counts.most_common()),
    }
    (outdir / "label_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print("分类数据集已生成:")
    print(f"  总样本: {n}   标签数: {len(label_counts)}")
    print(f"  train: {len(train)}   val: {len(val)}   test: {len(test)}")
    print(f"  剔除评测集: {excluded}   跳过无效: {skipped}   重复问题: {duplicated}   多部法样本: {multi_law}")
    print(f"  输出目录: {outdir}")
    print(f"  最大类: {label_counts.most_common(1)[0]}")
    print(f"  最小类: {label_counts.most_common()[-1]}")


if __name__ == "__main__":
    main()
