# -*- coding: utf-8 -*-
"""临时脚本：经 GitHub Contents API 拉取 DISC-Law-Eval 评测数据与代码。

raw.githubusercontent / jsdelivr 在本机均不通，API 可达（返回 base64）。
"""
import base64
import json
import os
import time
import urllib.request

API = "https://api.github.com/repos/FudanDISC/DISC-LawLLM/contents/eval/{}"
OUT = r"E:/workspace/Legal/data/external/DISC-Law-Eval"

FILES = [
    "README.md",
    "README-en.md",
    "datasets/objective/mcq_sing_nje.csv",
    "datasets/objective/mcq_sing_pae.csv",
    "datasets/objective/mcq_sing_cpa.csv",
    "datasets/objective/mcq_sing_ungee.csv",
    "datasets/objective/mcq_sing_lbk.csv",
    "datasets/objective/mcq_sing_pfe.csv",
    "datasets/objective/mcq_mult_nje.csv",
    "datasets/objective/mcq_mult_pae.csv",
    "datasets/objective/mcq_mult_cpa.csv",
    "datasets/objective/mcq_mult_ungee.csv",
    "datasets/subjective/qa_short_answer.json",
    "src/eval.py",
    "src/main.py",
    "src/models.py",
    "src/utils.py",
]


def fetch(rel: str) -> bytes:
    req = urllib.request.Request(API.format(rel), headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    if isinstance(d, dict) and d.get("encoding") == "base64":
        return base64.b64decode(d["content"])
    raise RuntimeError(f"unexpected payload for {rel}")


def main() -> None:
    ok, fail = [], []
    for rel in FILES:
        dst = os.path.join(OUT, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            data = fetch(rel)
            with open(dst, "wb") as f:
                f.write(data)
            ok.append((rel, len(data)))
            print(f"OK   {len(data):>9,} B  {rel}", flush=True)
        except Exception as e:  # noqa: BLE001
            fail.append((rel, repr(e)))
            print(f"FAIL {rel} -> {e}", flush=True)
        time.sleep(0.35)
    print(f"\n完成 {len(ok)} / 失败 {len(fail)}")
    for rel, err in fail:
        print("  fail:", rel, err)


if __name__ == "__main__":
    main()
