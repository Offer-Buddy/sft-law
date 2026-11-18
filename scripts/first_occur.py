# -*- coding: utf-8 -*-
"""逐词定位首现行：输出行号 + 是否在 svg 内 + 去标签后的该行文字（截断 120 字）。
"""
import re, sys, io

path = sys.argv[1]
html = io.open(path, encoding="utf-8").read()
lines = html.split("\n")

# 标记每行是否处在 <svg> 内
in_svg = []
flag = False
for ln in lines:
    if "<svg" in ln:
        flag = True
    in_svg.append(flag)
    if "</svg>" in ln:
        flag = False

TERMS = ["继续预训练", "CPT", "全参", "PEFT", "偏好对齐", "RLHF", "DPO", "量化",
    "推理", "测试集", "泛化", "欠拟合", "过拟合", "早停", "学习率", "显存",
    "token", "prompt", "意图识别", "问题改写", "拒答", "评测集", "标注",
    "幻觉", "灾难性遗忘", "batch", "基座", "超参数", "样本", "前向传播",
    "反向传播", "loss", "梯度", "epoch", "dropout", "Embedding", "向量"]

for t in TERMS:
    for i, ln in enumerate(lines):
        if t in ln:
            txt = re.sub(r"<[^>]+>", "", ln).strip()
            print("%-8s L%-5d %-5s | %s" % (t, i + 1, "SVG" if in_svg[i] else "text", txt[:110]))
            break
    else:
        print("%-8s ----  （未出现）" % t)
