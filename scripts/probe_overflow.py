# -*- coding: utf-8 -*-
"""窄屏溢出探针：在页面末尾注入脚本，把 scrollWidth 与越界元素写入 title"""
import io, os, re, subprocess, sys, tempfile

CHROME = r"C:/Program Files/Google/Chrome/Application/chrome.exe"
FILES = [
    r"E:/workspace/sft-law/course/L01_项目全景与基座选型_课时页.html",
    r"E:/workspace/sft-law/course/L02_模型是怎么学会的_课时页.html",
    r"E:/workspace/sft-law/course/L03_LoRA与QLoRA_课时页.html",
]

PROBE = """
<script>
(function(){
  var W = document.documentElement.clientWidth;
  var bad = [];
  document.querySelectorAll('body *').forEach(function(el){
    var r = el.getBoundingClientRect();
    if (r.width > 0 && (r.right > W + 1.5 || r.left < -1.5)) {
      var t = (el.tagName || '').toLowerCase();
      if (t === 'script' || t === 'style') return;
      bad.push(t + '.' + (el.className || '-') + '@' + Math.round(r.left) + '..' + Math.round(r.right));
    }
  });
  document.title = 'PROBE|win=' + W + '|scroll=' + document.documentElement.scrollWidth
    + '|body=' + document.body.scrollWidth
    + '|bad=' + bad.slice(0, 6).join(' ; ');
})();
</script>
"""

def run(path, width):
    html = io.open(path, encoding="utf-8").read()
    html = html.replace("</body>", PROBE + "</body>")
    tmp = os.path.join(tempfile.gettempdir(), "probe_%d_%s" % (width, os.path.basename(path)))
    io.open(tmp, "w", encoding="utf-8").write(html)
    out = subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--window-size=%d,2000" % width,
         "--dump-dom", "file:///" + tmp.replace("\\", "/")],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    m = re.search(r"<title>([^<]*)</title>", out.stdout or "")
    return m.group(1) if m else "(no title)"

for f in FILES:
    for w in (390, 900):
        title = run(f, w)
        flag = "溢出" if "bad=" in title and title.split("bad=")[1].strip() else "OK"
        print("%-6s %-4s %s" % (os.path.basename(f)[:5], w, title.replace("PROBE|", "")))
