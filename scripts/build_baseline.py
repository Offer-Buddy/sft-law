"""从 DISC-Law Triplet 造法条语料与评测集，并跑出检索基线。

这个项目不含检索系统，但**必须能自己量出基线**，否则微调完不知道有没有变好。
全部原料来自 Triplet 的 reference 字段——它标的就是「这个问题该命中哪条法条」。

三个步骤：
    python scripts/build_baseline.py corpus    # 抽法条语料（去重）
    python scripts/build_baseline.py evalset   # 抽评测集（问题 + 标准答案）
    python scripts/build_baseline.py run       # 建索引、跑检索、出 baseline.json

第一步和第二步只需跑一次；之后每次换了模型或数据，重跑第三步即可。
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_TRIPLET = str(BASE_DIR / "data" / "raw" / "DISC-Law-SFT-Triplet-QA-released.jsonl")
DEFAULT_CORPUS = str(BASE_DIR / "data" / "corpus" / "articles.jsonl")
DEFAULT_EVALSET = str(BASE_DIR / "data" / "eval" / "evalset.jsonl")
DEFAULT_BASELINE = str(BASE_DIR / "data" / "eval" / "baseline.json")

# 《民法典》第一百八十四条：因自愿实施紧急救助行为……
CITE = re.compile(r"《([^》]{2,30})》第([一二三四五六七八九十百千零〇两\d]+)条")
QUESTION = re.compile(r"<问题>\s*[：:]\s*(.+)", re.S)


def article_key(law: str, number: str) -> str:
    """法条唯一标识，评测和对齐都用它。"""
    return f"{law}_第{number}条"


def read_triplets(path: str):
    """逐行读 Triplet，产出 (问题, [(key, 法条原文), ...])。"""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            match = QUESTION.search(row.get("input", ""))
            if not match:
                continue
            question = match.group(1).strip()
            refs = []
            for ref in row.get("reference") or []:
                cite = CITE.search(ref)
                if cite:
                    refs.append((article_key(cite.group(1), cite.group(2)), ref.strip()))
            if question and refs:
                yield question, refs


def cmd_corpus(args: argparse.Namespace) -> None:
    """把 reference 里的法条去重，得到检索要命中的目标库。"""
    articles: dict[str, dict] = {}
    for _, refs in read_triplets(args.triplet):
        for key, text in refs:
            if key not in articles:
                articles[key] = {"key": key, "text": text}

    out = Path(args.corpus)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in articles.values():
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    laws = {k.split("_第")[0] for k in articles}
    print(f"法条语料：{len(articles)} 条 / {len(laws)} 部法 -> {out}")


def cmd_evalset(args: argparse.Namespace) -> None:
    """抽评测集：一条问题 + 它该命中的法条 key 列表。

    注意：这里抽出来的是**书面语**问题，命中率会偏高。
    要得到真实基线，得先用 LLM 把问题改写成口语（标注不动），
    再以改写的文件作为 --evalset 跑 run。见 docs 里的说明。
    """
    samples = list(read_triplets(args.triplet))
    random.Random(args.seed).shuffle(samples)
    picked = samples[: args.size]

    out = Path(args.evalset)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for question, refs in picked:
            row = {
                "question": question,
                "relevant": [key for key, _ in refs],
                "style": "书面语",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    avg = sum(len(r) for _, r in picked) / len(picked)
    print(f"评测集：{len(picked)} 条，平均每条 {avg:.2f} 个标准法条 -> {out}")


def cmd_run(args: argparse.Namespace) -> None:
    """编码语料与查询，按余弦相似度取 top-k，算 hit_rate 与 MRR。"""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    corpus = [json.loads(x) for x in open(args.corpus, encoding="utf-8") if x.strip()]
    queries = [json.loads(x) for x in open(args.evalset, encoding="utf-8") if x.strip()]
    print(f"语料 {len(corpus)} 条 / 查询 {len(queries)} 条，模型 {args.model}")

    model = SentenceTransformer(args.model, device=args.device)
    corpus_vec = model.encode(
        [c["text"] for c in corpus],
        batch_size=64, normalize_embeddings=True, show_progress_bar=True,
    )
    query_vec = model.encode(
        [q["question"] for q in queries],
        batch_size=64, normalize_embeddings=True, show_progress_bar=True,
    )

    # 余弦相似度：向量已归一化，内积即余弦
    scores = query_vec @ corpus_vec.T
    top_idx = np.argsort(-scores, axis=1)[:, : args.top_k]

    hits, rr, details = 0, 0.0, []
    for i, query in enumerate(queries):
        gold = set(query["relevant"])
        ranked = [corpus[j]["key"] for j in top_idx[i]]
        rank = next((r + 1 for r, key in enumerate(ranked) if key in gold), None)
        if rank:
            hits += 1
            rr += 1.0 / rank
        details.append({
            "question": query["question"],
            "style": query.get("style", "书面语"),
            "gold": sorted(gold),
            "top": ranked,
            "rank": rank,
            "hit": rank is not None,
        })

    total = len(queries)
    result = {
        "model": args.model,
        "top_k": args.top_k,
        "corpus_size": len(corpus),
        "eval_size": total,
        "style": sorted({d["style"] for d in details}),
        "hit_rate": round(hits / total, 4),
        "mrr": round(rr / total, 4),
        "miss": total - hits,
        "details": details,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"hit_rate@{args.top_k} = {result['hit_rate']}")
    print(f"MRR            = {result['mrr']}")
    print(f"未命中          = {result['miss']} / {total}")
    print(f"明细 -> {out}")
    if result["style"] == ["书面语"]:
        print()
        print("提醒：这是书面语问题的数字，会明显偏高。")
        print("      把问题改写成口语后重跑，才是这个项目真实的基线。")


def main() -> None:
    parser = argparse.ArgumentParser(description="造法条语料 / 评测集，并跑检索基线")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("corpus", help="从 Triplet 抽法条语料")
    p1.add_argument("--triplet", default=DEFAULT_TRIPLET)
    p1.add_argument("--corpus", default=DEFAULT_CORPUS)
    p1.set_defaults(func=cmd_corpus)

    p2 = sub.add_parser("evalset", help="抽评测集")
    p2.add_argument("--triplet", default=DEFAULT_TRIPLET)
    p2.add_argument("--evalset", default=DEFAULT_EVALSET)
    p2.add_argument("--size", type=int, default=400)
    p2.add_argument("--seed", type=int, default=42)
    p2.set_defaults(func=cmd_evalset)

    p3 = sub.add_parser("run", help="跑检索基线")
    p3.add_argument("--corpus", default=DEFAULT_CORPUS)
    p3.add_argument("--evalset", default=DEFAULT_EVALSET)
    p3.add_argument("--out", default=DEFAULT_BASELINE)
    p3.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    p3.add_argument("--device", default="cpu")
    p3.add_argument("--top-k", type=int, default=5)
    p3.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
