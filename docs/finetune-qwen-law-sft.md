# 微调实现说明

本文讲**脚本怎么用、参数怎么填、评测怎么算**。项目是什么、目录里有什么、整体怎么跑，看根目录 `README.md`。

主线是 50 部法分类；HyDE 改写链路降为支线，命令收在文末附录。

---

## 一、造数据

### 命令

```bash
python scripts/build_cls_dataset.py
```

### 参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--triplet-path` | `data/raw/DISC-Law-SFT-Triplet-QA-released.jsonl` | 原始数据，只读 |
| `--output-dir` | `data/cls` | 三份 jsonl + 标签统计的落点 |
| `--val-ratio` | 0.1 | 验证集比例（分层抽样） |
| `--test-ratio` | 0.1 | 测试集比例 |
| `--eval-files` | `data/eval/` 下三份评测集 | 要从训练数据里剔掉的考卷 |
| `--keep-evalset` | 关 | 打开就不剔考卷——只会用来复现「数据泄漏」的坏结果 |
| `--seed` | 42 | 切分随机种子，换了就不是同一组数 |

### 产出

```text
data/cls/
├── train.jsonl      18,338 条   query / response
├── val.jsonl         2,292 条
├── test.jsonl        2,292 条
└── label_stats.json            50 类 + 每类条数 + 切分统计
```

### 跑完自己验三句

```bash
python -c "import json;d=json.load(open('data/cls/label_stats.json',encoding='utf-8'));print(d['num_labels'],d['total'],d['splits'])"
# 期望：50 22922 {'train': 18338, 'val': 2292, 'test': 2292}

python -c "import json;d=json.load(open('data/cls/label_stats.json',encoding='utf-8'));print(min(d['labels'].values()),max(d['labels'].values()))"
# 期望：93 3727 —— 没有 0 条类

# 三份文件的问题零重叠
python -c "
import json
s=[{json.loads(l)['query'] for l in open(f'data/cls/{n}.jsonl',encoding='utf-8')} for n in ('train','val','test')]
print(len(s[0]&s[1]),len(s[0]&s[2]),len(s[1]&s[2]))"
# 期望：0 0 0
```

### 出平台格式

```bash
python scripts/export_platform_data.py
```

无参数，固定读 `data/cls/`，写 `data/cls/export/`：

| 产物 | 给谁吃 |
|---|---|
| `ms_swift/{train,val}.jsonl` | 魔搭 ms-swift，query/response 原样 |
| `llamafactory/cls_{train,val}.jsonl` | AutoDL LLaMA-Factory，instruction/output |
| `llamafactory/dataset_info.json` | 并进 LLaMA-Factory 的 `data/dataset_info.json` |
| `test_eval.jsonl` | `eval_cls.py`，字段 `question` / `label` |

---

## 二、训练

两个平台同一套超参，只是入口不同。

| 项 | 取值 | 为什么 |
|---|---|---|
| 基座 | Qwen3-1.7B | Apache 2.0，中文友好，1.7B 单卡可训 |
| 方法 | QLoRA 4bit（NF4 + 双量化） | 基座权重 3.4G → 0.9G |
| LoRA | r=16 / α=32，target all-linear | 分类任务 r=8 也够，16 留余量 |
| 学习率 | 2e-4，cosine + warmup 0.05 | QLoRA 常用档，高了发散、低了欠拟合 |
| epoch | 2 | 50 类收敛快，1 可能欠、3 易过拟合 |
| max_len | 512 | 问题最长 333 字 + 指令，512 足够 |
| batch | 8 × 梯度累积 2 | 有效 batch 16 |

### 魔搭 ms-swift

```bash
pip install ms-swift -U

swift sft \
  --model Qwen/Qwen3-1.7B \
  --tuner_type lora \
  --quant_method bnb --quant_bits 4 \
  --dataset /mnt/workspace/cls/train.jsonl \
  --val_dataset /mnt/workspace/cls/val.jsonl \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --lora_rank 16 --lora_alpha 32 \
  --target_modules all-linear \
  --max_length 512 \
  --eval_steps 100 --save_steps 100 --save_total_limit 2 \
  --logging_steps 20 \
  --warmup_ratio 0.05 \
  --output_dir /mnt/workspace/output/cls_qwen3
```

ms-swift 没有 `qlora` 这个 tuner，QLoRA 就是 `--tuner_type lora` 加 `--quant_method bnb --quant_bits 4`。

产物长这样：

```text
/mnt/workspace/output/cls_qwen3/
└─ v0-20261118-153012/
   ├─ checkpoint-600/  adapter_config.json + adapter_model.safetensors（约 40 MB）
   ├─ checkpoint-800/
   └─ logging.jsonl    完整 loss 与 eval 记录
```

### AutoDL LLaMA-Factory

```bash
# 1. 数据
cp data/cls/export/llamafactory/cls_*.jsonl  /root/autodl-tmp/LLaMA-Factory/data/
# 2. dataset_info.json 片段并进去，注册名 cls_law / cls_law_val
# 3. 起训
llamafactory-cli train cls_qlora.yaml
# 4. 产物
/root/autodl-tmp/saves/qwen3-1.7b/cls/checkpoint-*
```

yaml 里除上表超参外，dataset 填注册名（`cls_law`），不是文件路径。

### 显存账

| 项 | 占用 | 说明 |
|---|---|---|
| 基座（NF4 + 双量化） | ≈ 0.9 G | 1.7B × 0.5 byte，权重固定 |
| LoRA 权重 / 梯度 / 优化器 | ≈ 0.3 G | 可训参数约 20M |
| 激活值 | ≈ 2~4 G | 最大变量，随 batch × 长度涨；开梯度检查点能压住 |
| CUDA 上下文 + 碎片 | ≈ 1~2 G | 省不掉 |
| **合计** | **≈ 5~7 G** | 24G 卡真实占用不到三分之一 |

分类任务账特别轻，原因只有一个：max_len 只要 512。生成任务要 768~1024，光激活值就差一倍。

---

## 三、评测

### 命令

```bash
# zero-shot 基线（先跑这个）
python scripts/eval_cls.py --no-adapter --limit 200

# 微调后
python scripts/eval_cls.py --adapter output/checkpoint-800
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `--no-adapter` | 关 | 不加载适配器，量基线 |
| `--adapter` | 无 | 适配器目录（checkpoint） |
| `--model` | `Qwen/Qwen3-1.7B` | 基座 |
| `--data` | `data/cls/export/test_eval.jsonl` | 评测集，`question` / `label` |
| `--labels-from` | `data/cls/label_stats.json` | 50 个候选标签从这读，不手抄 |
| `--prompt` | 与训练一致的指令 | 必须与训练时一模一样 |
| `--limit` | 全量 | 先跑 200 条估个大概，再跑全量 |
| `--out` | `eval_report.json` | 结果落点 |

### 为什么不自由生成

自由生成会「说漏嘴」——模型吐「民法典第一编」，不在 50 类里，字符串一匹配就判错，其实它知道答案。改成 **50 个候选逐个打分取最大**：

```python
scores = {}
for law in labels:                      # 50 个部法名
    lp = sum(logprob(t) for t in tokenize(law))
    scores[law] = lp / len(tokenize(law))   # 按 token 数取平均
pick = max(scores, key=scores.get)
top3 = sorted(scores, key=scores.get, reverse=True)[:3]
```

**按 token 数取平均这一步不能漏**：「民法典」3 个 token、「国际刑事司法协助法」9 个，直接加总比，长标签天然分更低——混淆矩阵里那批「本该判长名、却判给短名」的错案，原因不在模型，在打分代码。

### 四把尺子

| 指标 | 算法 | 回答什么 |
|---|---|---|
| accuracy | 判对条数 / 总条数 | 50 选 1 整体对了多少 |
| macro-F1 | 每类 F1 取算术平均（不按样本数加权） | 每一类平均好不好，小类错了立刻掉 |
| top-3 | 正确答案落在前 3 个候选的比例 | 差一点就对的有多少 |
| 混淆矩阵 | 50×50，行=真实、列=预测 | 具体混在哪两对 |

### 混淆矩阵只读三处

- **对角线**——每类判对的条数；
- **行和**——这一类总共多少条，是这一类的分母；
- **非对角线最亮的格子**——病灶。重点盯刑法 → 刑事诉讼法、民法典 → 民事诉讼法。

报结论别说「准确率 88%」就完事，要说「民法典被判成民诉法吃掉了多少条」。

### 小类的数要带分母

公益事业捐赠法 test 只有 8 条，错 1 条就是 12.5 个百分点。报小类准确率必须同时报 test 条数，只给百分比等于没报。

---

## 附录：HyDE 支线（前身链路）

把口语翻成法条语体、靠检索命中率验证。**不是当前主线**，仅保留可跑的命令。

```bash
# 1. 抽法条语料 / 抽评测集 / 跑检索基线
python scripts/build_baseline.py corpus     # 6,432 条 → data/corpus/articles.jsonl
python scripts/build_baseline.py evalset    # 400 条 → data/eval/evalset.jsonl
python scripts/build_baseline.py run        # → data/eval/baseline.json

# 2. 造训练集：问题 → 法条语体
python scripts/build_trainset.py                 # 全量 21,785 / 1,146
python scripts/build_trainset.py --max-samples 500   # 小样先验

# 3. 训练（手写 transformers+peft，非主线；主线用 LLaMA-Factory / ms-swift）
python scripts/train_qwen_hyde_lora.py --use-4bit
python scripts/infer_qwen_hyde_lora.py --question "加班不给加班费怎么办？"

# 4. 需要完整 HF 目录时合并
python scripts/merge_qwen_lora.py --adapter-path <适配器> --output-path <输出目录>
```

已知数（法条库 6,432 条、`BAAI/bge-small-zh-v1.5`、top-5）：书面语 400 条 hit_rate 0.7975 / MRR 0.6538；书面语 40 条 0.775 / 首位 87%；口语 40 条 0.725 / 首位 66%。

`scripts/prepare_hyde_sft_data.py` 是更早的一版造数脚本（Pair + Triplet 混用），口径已被 `build_trainset.py` 取代，留档不删。

---

## 常见错法

| 现象 | 真实原因 |
|---|---|
| 准确率虚高，换真实数据就掉 | 评测集问题混进了训练集 |
| 模型只会抄法条 | 输入里没剥掉法条原文 |
| 长名部法系统性判错 | 候选打分没按 token 数取平均 |
| 报「提升 X 个点」没人信 | 没跑 zero-shot 基线 |
| 小类准确率忽高忽低 | test 条数太少，波动本来就大 |
| 训练报 dataset not found | LLaMA-Factory 的 dataset 要填注册名，不是文件路径 |
