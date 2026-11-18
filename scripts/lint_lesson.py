# -*- coding: utf-8 -*-
"""课时长页机检：div 配平 / 变量 / 锚点 / 表格列数 / 外链 / 占位 / 痕迹 / takeaway / 打印"""
import re, sys, io, os

FILES = sys.argv[1:] or [
    r"E:/workspace/sft-law/course/L01_项目全景与基座选型_课时页.html",
    r"E:/workspace/sft-law/course/L02_模型是怎么学会的_课时页.html",
    r"E:/workspace/sft-law/course/L03_LoRA与QLoRA_课时页.html",
]

MECH = ["课时", "作业", "签到", "考勤", "教学大纲", "本节课将", "本课时将"]
META = ["本文将", "本节将", "接下来我们", "下面让我们", "综上所述"]
PLACE = ["TBD", "TODO", "待补", "占位", "xxx", "XXX", "略"]

def visible_text(html):
    s = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", s)

def check(path):
    html = io.open(path, encoding="utf-8").read()
    name = os.path.basename(path)
    errs, warns = [], []
    body = re.sub(r"<!--[\s\S]*?-->", "", html)

    # 1 div 配平
    opens = len(re.findall(r"<div\b", body))
    closes = len(re.findall(r"</div>", body))
    if opens != closes:
        errs.append(f"div 不配平 open={opens} close={closes}")

    # 2 CSS 变量
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", body))
    used = set(re.findall(r"var\((--[a-z0-9-]+)", body))
    miss = used - defined
    if miss:
        errs.append(f"未定义变量 {sorted(miss)}")

    # 3 锚点
    ids = set(re.findall(r'id="([^"]+)"', body))
    hrefs = set(re.findall(r'href="#([^"]+)"', body))
    dead = hrefs - ids
    if dead:
        errs.append(f"死锚点 {sorted(dead)}")

    # 4 表格列数
    for ti, tbl in enumerate(re.findall(r"<table[\s\S]*?</table>", body), 1):
        counts = []
        for tr in re.findall(r"<tr[\s\S]*?</tr>", tbl):
            n = len(re.findall(r"<t[dh]\b", tr))
            if n:
                counts.append(n)
        if counts and len(set(counts)) > 1:
            errs.append(f"表{ti} 列数不一 {sorted(set(counts))}")

    # 5 外链
    ext = re.findall(r'(?:src|href)="(https?://[^"]+)"', body)
    if ext:
        errs.append(f"外链 {ext[:3]}")

    # 6 占位
    vis = visible_text(body)
    for w in PLACE:
        if w in vis:
            errs.append(f"占位词「{w}」")

    # 7 授课痕迹
    for w in MECH:
        if w in vis:
            errs.append(f"授课痕迹「{w}」")

    # 8 元叙述
    for w in META:
        if w in vis:
            warns.append(f"元叙述「{w}」")

    # 9 每章 recap + takeaway 配对（小结章只需 takeaway）
    h2s = re.findall(r'<h2 id="([^"]+)"[\s\S]*?(?=<h2 id=|</main>)', body)
    h2ids = [m.group(1) for m in re.finditer(r'<h2 id="([^"]+)"', body)]
    segs = re.split(r'(?=<h2 id=")', body)
    n_recap = n_ta = 0
    for seg in segs:
        m = re.match(r'<h2 id="([^"]+)"', seg)
        if not m:
            continue
        sid = m.group(1)
        has_r = 'class="recap"' in seg
        has_t = 'class="takeaway"' in seg
        n_recap += has_r
        n_ta += has_t
        if sid == "s-wrap":
            if not has_t:
                errs.append("小结章缺 takeaway")
        elif not (has_r and has_t):
            miss = ("recap " if not has_r else "") + ("takeaway" if not has_t else "")
            errs.append(f"章 {sid} 缺 {miss.strip()}")
    if n_ta == 0:
        errs.append("全页无 takeaway")

    # 10 代码块裸尖括号
    for cb in re.findall(r"<pre[\s\S]*?</pre>|<code[\s\S]*?</code>", body):
        inner = re.sub(r"<[^>]+>", "", cb)
        if "<" in inner or ">" in inner:
            errs.append("代码块内有裸尖括号")

    # 11 打印规则
    if "@media print" not in body:
        errs.append("缺 @media print")

    # 12 svg viewBox
    svgs = re.findall(r"<svg\b[^>]*>", body)
    for s in svgs:
        if "viewBox" not in s:
            errs.append("svg 缺 viewBox")
            break

    print(f"\n=== {name} ===")
    print(f"  svg={len(svgs)} table={body.count('<table')} 字数={len(vis)}")
    if errs:
        for e in errs:
            print("  [ERR]", e)
    if warns:
        for w in warns:
            print("  [WARN]", w)
    if not errs:
        print("  PASS")
    return len(errs)

total = sum(check(f) for f in FILES)
print(f"\n合计错误 {total}")
sys.exit(1 if total else 0)
