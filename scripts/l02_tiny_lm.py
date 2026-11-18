"""
L02 配套实验：3 秒看一次 loss 掉下来。

做什么
    一个只“看上一个字、猜下一个字”的最小模型（bigram），纯 torch，CPU 上几秒跑完。
    不上 GPU、不下载任何预训练权重。

跑法
    python scripts/l02_tiny_lm.py

看什么
    四档学习率各训 300 步，对比最后 20 步的平均 loss。
    学习率太小 → loss 几乎不动；太大 → 直接炸成 nan。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

TEXT = (
    "中华人民共和国劳动合同法。用人单位安排劳动者延长工作时间的，"
    "支付不低于工资的百分之一百五十的工资报酬。"
) * 40

chars = sorted(set(TEXT))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in TEXT])
V = len(chars)


class Bigram(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 64)
        self.out = nn.Linear(64, vocab)

    def forward(self, x):
        return self.out(self.emb(x))


def train(lr, steps=300):
    model = Bigram(V)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    xs, ys = data[:-1], data[1:]
    curve = []
    for _ in range(steps):
        loss = F.cross_entropy(model(xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(loss.item())
    return curve, model


def generate(model, n=24):
    cur = stoi["中"]
    out = "中"
    with torch.no_grad():
        for _ in range(n):
            cur = int(model(torch.tensor([cur])).argmax(-1))
            out += itos[cur]
    return out


if __name__ == "__main__":
    print(f"语料 {len(TEXT)} 字，词表 {V} 个字，模型 {sum(p.numel() for p in Bigram(V).parameters()):,} 个参数")
    print(f"开局 loss（纯瞎猜）≈ ln({V}) = {torch.log(torch.tensor(float(V))).item():.2f}\n")
    print(f"{'学习率':>8} {'第1步':>8} {'最后20步均值':>14} {'结局':>10}  接龙结果")
    print("-" * 76)
    for lr in [1e-5, 1e-3, 5e-3, 2.0]:
        curve, model = train(lr)
        first = curve[0]
        tail = sum(curve[-20:]) / 20
        if tail > 3.0:
            verdict, text = "炸了 / 没学", generate(model)
        else:
            verdict, text = "学到了", generate(model)
        print(f"{lr:>8} {first:>8.2f} {tail:>14.2f} {verdict:>10}  {text}")
