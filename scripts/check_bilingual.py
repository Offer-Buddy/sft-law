# -*- coding: utf-8 -*-
"""扫课时页正文（从第一个 <h2> 起）：每个专有名词首次出现的句子，是否带英文/缩写。
"""
import re, sys, io

path = sys.argv[1] if len(sys.argv) > 1 else "course/L00_微调全景与前导_课时页.html"
html = io.open(path, encoding="utf-8").read()
html = re.sub(r"<style[\s\S]*?</style>", "", html)
html = re.sub(r"<script[\s\S]*?</script>", "", html)
html = html[html.find("<h2"):]

# 块级切分 → 句子级
chunks = re.split(r"</p>|</li>|</td>|</div>|<br>|<h3>|</h3>|<figcaption>|</figcaption>", html)
text = "\n".join(re.sub(r"<[^>]+>", "", c).strip() for c in chunks)
sentences = re.split(r"(?<=[。！？])", text)

TERMS = ["预训练", "继续预训练", "微调", "全参", "冻结", "Adapter", "LoRA", "QLoRA", "PEFT",
    "SFT", "CPT", "RAG", "蒸馏", "偏好对齐", "RLHF", "DPO", "量化", "推理",
    "训练集", "验证集", "测试集", "过拟合", "欠拟合", "泛化", "早停", "dropout",
    "epoch", "学习率", "梯度", "反向传播", "前向传播", "loss", "交叉熵",
    "显存", "基座", "checkpoint", "token", "prompt", "上下文", "embedding",
    "幻觉", "拒答", "意图识别", "问题改写", "HyDE", "hit_rate", "评测集",
    "灾难性遗忘", "batch", "梯度累积", "混合精度", "fp16", "bf16", "fp32",
    "超参数", "偏置", "权重", "自回归", "标注", "样本"]

def first_line(term):
    for s in sentences:
        if term in s:
            return s.strip()
    return None

print("=== 正文首次出现（从第 1 章起）===")
for t in TERMS:
    s = first_line(t)
    if s is None:
        print("  --  %-10s | （正文未出现）" % t)
        continue
    i = s.find(t)
    seg = s[i:i+len(t)+55]
    ok = bool(re.search(r"[（(][^）)]*[A-Za-z]{2,}[^）)]*[）)]", seg)) or bool(re.search(r"[A-Za-z]{2,}", s[max(0,i-25):i]))
    print("  %s %-10s | %s" % ("OK " if ok else "!! ", t, s[:88]))
