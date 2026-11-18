"""把 data/cls/ 的 query/response 训练对导出成两个平台各自吃的格式。

- 魔搭 ms-swift：原生吃 query/response，直接用 data/cls/{train,val}.jsonl，无需转换
- AutoDL LLaMA-Factory：默认 alpaca 列名是 instruction/output，需要转一份

产物（都落在 data/cls/export/）：
  ms_swift/train.jsonl         query/response，18,338 条
  ms_swift/val.jsonl           query/response，2,292 条
  llamafactory/cls_train.jsonl instruction/output
  llamafactory/cls_val.jsonl   instruction/output
  llamafactory/dataset_info.json  贴进 LLaMA-Factory/data/dataset_info.json 的片段
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLS_DIR = ROOT / "data" / "cls"
OUT = CLS_DIR / "export"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    train = read_jsonl(CLS_DIR / "train.jsonl")
    val = read_jsonl(CLS_DIR / "val.jsonl")
    test = read_jsonl(CLS_DIR / "test.jsonl")

    # ── ms-swift：原样拷贝 ──
    write_jsonl(OUT / "ms_swift" / "train.jsonl", train)
    write_jsonl(OUT / "ms_swift" / "val.jsonl", val)

    # ── LLaMA-Factory：query/response → instruction/output ──
    def to_alpaca(rows: list[dict]) -> list[dict]:
        return [
            {"instruction": r["query"], "input": "", "output": r["response"]}
            for r in rows
        ]

    lf_train = to_alpaca(train)
    lf_val = to_alpaca(val)
    write_jsonl(OUT / "llamafactory" / "cls_train.jsonl", lf_train)
    write_jsonl(OUT / "llamafactory" / "cls_val.jsonl", lf_val)

    # ── dataset_info 片段：file_name 相对 LLaMA-Factory/data/ ──
    info = {
        "cls_law": {
            "file_name": "cls_train.jsonl",
            "columns": {"prompt": "instruction", "response": "output"},
        },
        "cls_law_val": {
            "file_name": "cls_val.jsonl",
            "columns": {"prompt": "instruction", "response": "output"},
        },
    }
    (OUT / "llamafactory" / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ── 评测用：只留问题与标签，供 eval_cls.py 读 ──
    eval_rows = [
        {"question": r["query"].split("问题：", 1)[-1].strip(), "label": r["response"]}
        for r in test
    ]
    write_jsonl(OUT / "test_eval.jsonl", eval_rows)

    print("导出完成")
    print(f"  ms_swift      train={len(train)}  val={len(val)}")
    print(f"  llamafactory  train={len(lf_train)}  val={len(lf_val)}")
    print(f"  test_eval     {len(eval_rows)} 条（只含 question / label）")
    for p in sorted(OUT.rglob("*.json*")):
        print(f"    {p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
