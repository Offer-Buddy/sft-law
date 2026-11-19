"""压测 50 类部法分类服务：并发 1 / 8 / 32 各打一轮，报 P50、P95、吞吐。

为什么不报平均值：平均值会被最快的那些请求拉低，用户感受到的是慢的那一端。
对外承诺用 P95——「95% 的请求在多少毫秒内返回」。

用法：
    python scripts/bench_cls.py --n 20 --conc 1,8,32
    python scripts/bench_cls.py --url http://127.0.0.1:8100 --n 50 --conc 16

输出：一张表 + 一份 bench_report.json（同一份并发档可反复跑，比的是改动前后）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "cls" / "export" / "test_eval.jsonl"


def load_questions(path: str, limit: int = 500) -> list[str]:
    """从评测集里取问题当压测样本——真实分布，比自己编 10 句话准。"""
    qs: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qs.append(json.loads(line)["question"])
            if len(qs) >= limit:
                break
    if not qs:
        raise SystemExit(f"没读到问题：{path}")
    return qs


def pct(xs: list[float], p: float) -> float:
    """最近秩分位数，小样本下比插值稳，也不引第三方库。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, math.ceil(p / 100 * len(s)) - 1)
    return s[k]


async def one_round(url: str, questions: list[str], n: int, conc: int) -> dict:
    """打一轮：n 个请求、最多 conc 个同时在飞，返回耗时分布。"""
    sem = asyncio.Semaphore(conc)
    lat: list[float] = []
    errors = 0
    async with httpx.AsyncClient(base_url=url, timeout=60.0) as client:

        async def task(q: str) -> None:
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post("/classify", json={"question": q})
                    r.raise_for_status()
                    r.json()
                except Exception:  # noqa: BLE001 - 压测只数错误个数
                    errors += 1
                lat.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        await asyncio.gather(*(task(q) for q in questions[:n]))
        wall = time.perf_counter() - t0

    ok = [x for x in lat if x > 0]
    return {
        "并发": conc,
        "请求数": n,
        "P50_ms": round(pct(ok, 50), 1),
        "P95_ms": round(pct(ok, 95), 1),
        "最慢_ms": round(max(ok) if ok else 0, 1),
        "吞吐_qps": round(n / wall, 2) if wall else 0,
        "错误": errors,
        "墙钟_s": round(wall, 2),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description="分类服务压测")
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--n", type=int, default=20, help="每档打多少个请求")
    ap.add_argument("--conc", default="1,8,32", help="并发档，逗号分隔")
    ap.add_argument("--out", default=str(ROOT / "data" / "cls" / "bench_report.json"))
    args = ap.parse_args()

    qs = load_questions(args.data)
    random.shuffle(qs)
    concs = [int(x) for x in args.conc.split(",") if x.strip()]

    rows = []
    for c in concs:
        # 并发比请求数还大时，实际并发就是请求数，先纠正一下
        row = await one_round(args.url, qs, max(args.n, c), min(c, max(args.n, c)))
        rows.append(row)
        print(f"  conc={row['并发']:>3}  P50={row['P50_ms']:>8}ms  "
              f"P95={row['P95_ms']:>8}ms  qps={row['吞吐_qps']:>6}  错误={row['错误']}")

    print("\n| 并发 | 请求数 | P50 (ms) | P95 (ms) | 吞吐 (req/s) | 错误 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['并发']} | {r['请求数']} | {r['P50_ms']} | {r['P95_ms']} | "
              f"{r['吞吐_qps']} | {r['错误']} |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"url": args.url, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n已写入 {args.out}")

    if len(rows) >= 2:
        grow = rows[-1]["P95_ms"] / max(rows[0]["P95_ms"], 1e-6)
        print(f"\n最高并发档的 P95 是最低档的 {grow:.1f} 倍。"
              f"跟并发近似线性 → 后端在排队（看 --max-num-seqs）；几乎不变 → 还有余量。")


if __name__ == "__main__":
    asyncio.run(main())
