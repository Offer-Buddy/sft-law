# sft-law

法律场景的 HyDE 微调工程。**独立项目**，不含检索系统、不含业务系统、不依赖任何外部仓库。

## 这个项目做什么

一句话：训一个能把大白话翻成法条语体的适配器，插在检索前面，让「老板让我天天加班不给钱」这类问法也能命中《劳动合同法》第四十四条。

```
用户提问（生活语言） → [翻译器 · 本项目产出] → 语义检索 → 命中法条
```

范围只有中间那一格。不回答用户、不给建议、不写「建议您」、不推理结论。

| 契约 | 内容 |
|---|---|
| 输入 | 一句生活语言的法律问题，实测平均 68.6 字 |
| 输出 | 一段法条语体文本，上限 220 token |
| 基座 | Qwen3-1.7B-Instruct（4bit 加载 + LoRA） |
| 指标 | hit_rate@5、首位命中率、MRR |

## 目录

```
sft-law/
├── course/          # 15 课时课纲 + 课时页（离线可开、可打印）
├── data/
│   ├── raw/         # DISC-Law-SFT 原始数据（171 MB）
│   ├── corpus/      # 法条语料 6,432 条 / 50 部法（从 reference 抽出）
│   ├── eval/        # 评测集与基线结果
│   └── external/    # DISC-Law-Eval（遗忘检测用）
├── docs/            # 微调说明
├── scripts/
│   ├── build_baseline.py      # 造语料、造评测集、跑基线
│   ├── build_trainset.py      # 造训练集：问题 → 法条（切 train/val，剔除评测集）
│   ├── train_qwen_hyde_lora.py
│   ├── infer_qwen_hyde_lora.py
│   └── merge_qwen_lora.py
└── requirements-finetune.txt
```

## 数据从哪来

全部来自复旦 DISC-Law，已下载到 `data/raw/`：

| 文件 | 条数 | 用途 |
|---|---|---|
| `DISC-Law-SFT-Triplet-QA-released.jsonl` | 23,331 | **训练对来源**：问题 + `reference` 法条原文 |
| `DISC-Law-SFT-Pair-QA-released.jsonl` | 79,692 | 本项目基本不用 |
| `data/external/DISC-Law-Eval/` | 2,563 + 300 | 遗忘检测（微调前后准确率对比） |

Triplet 的 `reference` 标的就是「这个问题该命中哪条法条」，所以既是训练目标，也是评测标注，不用人工标。

## 怎么跑

```bash
pip install -r requirements-finetune.txt
pip install sentence-transformers     # 只有跑基线才需要

# 1. 抽法条语料（6,432 条）
python scripts/build_baseline.py corpus

# 2. 抽评测集（默认 400 条书面）
python scripts/build_baseline.py evalset

# 3. 造训练集（剔除评测集问题，切 train/val，默认只留 reference 当目标）
python scripts/build_trainset.py

# 4. 跑基线
python scripts/build_baseline.py run
```

国内下载 embedding 模型先设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # Windows: set HF_ENDPOINT=...
```

## 已经量出来的数

法条库 6,432 条、模型 `BAAI/bge-small-zh-v1.5`、top-5：

| 评测集 | hit_rate@5 | 首位命中 | MRR |
|---|---|---|---|
| 书面语 400 条 | 0.7975 | — | 0.6538 |
| 书面语 40 条（对照组） | 0.775 | 87% | 0.7063 |
| **口语 40 条（同批改写）** | **0.725** | **66%** | **0.5792** |

两个结论：

1. **口语化之后，命中率只掉 5 个点，但首位命中从 87% 掉到 66%。** 法条还在前 5 条里，只是被挤到后面——语义鸿沟主要伤排序，不只伤召回。
2. **40 条里有 7 条两种问法都失败**，翻标注发现是标错了（家暴题标到《反家庭暴力法》第一条即立法目的条款、怀孕拒录标到村委会义务条款）。**未命中不等于模型差**，归因必须留「标注问题」这一类。

注意：口语版只有 40 条，是人工改写的样例，噪声不小。要得到稳定基线，按 L02 的做法把 400 条全部改写成口语再跑。

## 训练路径

主线走 HyDE，不走法律问答：

| 走法 | 训出来什么 | 能不能验证 |
|---|---|---|
| 法律问答 SFT（Pair 8 万条） | 会答法律题的模型 | 基本不能，答得好不好没法量化，且下游系统本来就由大模型负责这层 |
| **HyDE（Triplet 的 reference 当目标）** | **口语 → 法条语体的翻译器** | **能，hit_rate 涨多少一跑就知道** |
