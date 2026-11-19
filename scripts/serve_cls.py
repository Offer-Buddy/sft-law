"""50 类部法分类 · 线上服务（FastAPI 薄壳，后端是 vLLM 的 OpenAI 兼容接口）。

一次分类 = 50 次候选打分，与 scripts/eval_cls.py 同口径：
同样的 chat 模板（关 thinking）、同样把 eos 算进候选、同样按 token 数取平均。
线上少对齐任何一处，离线报告上的 accuracy 就不再代表线上。

依赖：fastapi uvicorn httpx transformers

用法：
    # 1) 先起推理后端（见 L08 第 4 章）
    vllm serve Qwen/Qwen3-1.7B \
        --served-model-name cls-base \
        --enable-lora --lora-modules cls=./cls_adapter \
        --max-lora-rank 16 --max-loras 2 --port 8000

    # 2) 再起本服务
    export VLLM_BASE_URL=http://127.0.0.1:8000/v1
    export CLS_MODEL=cls
    uvicorn scripts.serve_cls:app --host 0.0.0.0 --port 8100

    # 3) 第一次接新版本，先探字段（别猜响应长什么样）
    python scripts/serve_cls.py --probe "公司拖欠我三个月工资怎么办"

    # 4) 正常调用
    curl -s localhost:8100/classify -H 'Content-Type: application/json' \
         -d '{"question":"公司拖欠我三个月工资怎么办"}'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent

# 三项必须与训练/评测一致，改任何一项都要重跑一遍 test
PROMPT = "请判断以下法律问题应适用哪部法律。只输出法律名称，不要解释。"
DEFAULT_LABELS = ROOT / "data" / "cls" / "label_stats.json"
DEFAULT_TOKENIZER = "Qwen/Qwen3-1.7B"

# 部署侧配置全部走环境变量：换机器只改环境，不改代码
BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.getenv("CLS_MODEL", "cls")          # vLLM 里 --lora-modules 起的名字
TOKENIZER_PATH = os.getenv("CLS_TOKENIZER", DEFAULT_TOKENIZER)
TIMEOUT = float(os.getenv("CLS_TIMEOUT", "8"))  # 秒，超时必须设


def load_labels(path: str | Path) -> list[str]:
    """按样本数从多到少读出 50 个部法名，顺序与 eval_cls.py 一致。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = data.get("labels") if isinstance(data, dict) else None
    if not labels:
        raise SystemExit(f"标签文件里没有 labels 字段：{path}")
    return sorted(labels, key=lambda x: -labels[x])


_LABELS = load_labels(DEFAULT_LABELS)
_LABEL_IDS: dict[str, list[int]] = {}
_PREFIX_CACHE: dict[str, list[int]] = {}
_EOS: int | None = None
_TOKENIZER = None
_CLIENT: httpx.AsyncClient | None = None


def build_prompt_ids(question: str) -> list[int]:
    """按训练时的 qwen3 模板拼 prompt，并显式关掉 thinking——与 eval_cls.py 同一个函数。"""
    if question in _PREFIX_CACHE:
        return _PREFIX_CACHE[question]
    messages = [{"role": "user", "content": f"{PROMPT}\n\n问题：{question}"}]
    kwargs = dict(add_generation_prompt=True, tokenize=True, return_tensors=None)
    try:
        ids = _TOKENIZER.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        # 老版本 tokenizer 不认 enable_thinking
        ids = _TOKENIZER.apply_chat_template(messages, **kwargs)
    if len(_PREFIX_CACHE) > 4096:  # 简单上限，别让缓存无限长
        _PREFIX_CACHE.clear()
    _PREFIX_CACHE[question] = ids
    return ids


async def score_one(label: str, prompt_ids: list[int]) -> tuple[str, float]:
    """给一个候选打分：把「前缀 + 候选 + eos」整段喂进去，取候选那几位的 logprob 平均。"""
    tail = _LABEL_IDS[label] + ([_EOS] if _EOS is not None else [])
    payload = {
        "model": MODEL,
        "prompt": prompt_ids + tail,   # vLLM 接受 token id 列表，避开模板二次渲染
        "max_tokens": 1,               # 只要算力，不要它生成
        "temperature": 0.0,
        "prompt_logprobs": 0,          # 返回 prompt 每个位置的 logprob
    }
    assert _CLIENT is not None
    r = await _CLIENT.post("/completions", json=payload)
    r.raise_for_status()
    body = r.json()
    plp = (body.get("choices") or [{}])[0].get("prompt_logprobs")
    if plp is None:
        raise HTTPException(
            500,
            "响应里没有 prompt_logprobs —— 先跑 `python scripts/serve_cls.py --probe \"<问题>\"` "
            "把原始 JSON 打出来看字段名，不同 vLLM 版本这里可能不一样",
        )
    start = len(prompt_ids)
    total = 0.0
    for j, tid in enumerate(tail):
        entry = plp[start + j] or {}
        total += float(entry.get(str(tid), -99.0))  # 缺就当成极低分，不让它赢
    return label, total / len(tail)                  # 长度归一化，和离线一字不差


async def classify(question: str) -> dict:
    prompt_ids = build_prompt_ids(question)
    # 50 个候选一起发：vLLM 的连续批处理 + 前缀缓存会把公共前缀只算一次
    scored = await asyncio.gather(*(score_one(lab, prompt_ids) for lab in _LABELS))
    ranked = sorted(scored, key=lambda kv: -kv[1])
    return {
        "label": ranked[0][0],
        "score": round(ranked[0][1], 4),
        "top3": [{"label": lab, "score": round(s, 4)} for lab, s in ranked[:3]],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载 tokenizer、编码 50 个候选，服务期间复用同一个连接池。"""
    global _CLIENT, _TOKENIZER, _LABEL_IDS, _EOS
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    _LABEL_IDS = {
        lab: _TOKENIZER(lab, add_special_tokens=False).input_ids for lab in _LABELS
    }
    _EOS = _TOKENIZER.eos_token_id
    _CLIENT = httpx.AsyncClient(
        base_url=BASE_URL, timeout=TIMEOUT,
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
    )
    print(f"服务就绪：{len(_LABELS)} 类 · 后端 {BASE_URL} · 模型 {MODEL}")
    yield
    await _CLIENT.aclose()


app = FastAPI(title="50 类部法分类服务", lifespan=lifespan)


class Ask(BaseModel):
    question: str


@app.get("/healthz")
async def healthz():
    """探活：能连上后端、且列得出模型，才算真活着。"""
    assert _CLIENT is not None
    try:
        r = await _CLIENT.get("/models")
        r.raise_for_status()
        names = [m.get("id") for m in r.json().get("data", [])]
    except Exception as exc:  # noqa: BLE001 - 探活只要结论，不要栈
        raise HTTPException(503, f"后端不可达：{exc}") from exc
    return {"ok": MODEL in names, "backend": BASE_URL, "models": names}


@app.post("/classify")
async def classify_api(req: Ask):
    """入参一句话，出参一个部法名 + top3。这是接入文档里唯一要写清的接口。"""
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(400, "question 不能为空")
    if len(q) > 1000:
        raise HTTPException(400, "question 超过 1000 字，先截断再传")
    return await classify(q)


def probe(question: str) -> None:
    """把一次打分的原始响应原样打出来——换 vLLM 版本后先跑这个，不要猜字段。"""
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    lab = _LABELS[0]
    tail = _TOKENIZER(lab, add_special_tokens=False).input_ids
    prefix = build_prompt_ids(question)  # 与线上同一个模板函数
    payload = {
        "model": MODEL,
        "prompt": prefix + tail,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 0,
    }
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
        r = c.post("/completions", json=payload)
    print("HTTP", r.status_code)
    print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="分类服务：探字段 / 单条试跑")
    ap.add_argument("--probe", metavar="问题", help="打印一次打分的原始响应")
    ap.add_argument("--ask", metavar="问题", help="本地跑一次完整分类（不经 HTTP 服务）")
    args = ap.parse_args()
    if args.probe:
        probe(args.probe)
    elif args.ask:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        _LABEL_IDS = {
            lab: _TOKENIZER(lab, add_special_tokens=False).input_ids for lab in _LABELS
        }
        _EOS = _TOKENIZER.eos_token_id

        async def _run():
            global _CLIENT
            _CLIENT = httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT)
            try:
                print(json.dumps(await classify(args.ask), ensure_ascii=False, indent=2))
            finally:
                await _CLIENT.aclose()

        asyncio.run(_run())
    else:
        ap.print_help()
