# -*- coding: utf-8 -*-
"""窄屏溢出探针 v2：用 iframe 强制出真实窄视口（绕开 headless 最小窗口宽 500 的限制）"""
import io, os, re, subprocess, sys, tempfile

CHROME = r"C:/Program Files/Google/Chrome/Application/chrome.exe"
FILES = [
    r"E:/workspace/sft-law/course/L00_微调全景与前导_课时页.html",
    r"E:/workspace/sft-law/course/L01_项目全景与基座选型_课时页.html",
    r"E:/workspace/sft-law/course/L02_模型是怎么学会的_课时页.html",
    r"E:/workspace/sft-law/course/L03_LoRA与QLoRA_课时页.html",
]

PROBE = """
<script>
(function(){
  if (location.hash === '#inner') { return; }
  var widths = [390, 768];
  var i = 0;
  function next(){
    if (i >= widths.length) return;
    var w = widths[i++];
    var f = document.createElement('iframe');
    f.setAttribute('style','width:'+w+'px;height:1000px;border:0;position:absolute;left:-99999px;top:0');
    document.body.appendChild(f);
    f.onload = function(){
      try{
        var doc = f.contentDocument;
        var cw = f.contentWindow.innerWidth;
        var sw = doc.documentElement.scrollWidth;
        var bad = [];
        doc.querySelectorAll('body *').forEach(function(el){
          var t = (el.tagName||'').toLowerCase();
          if (t==='script'||t==='style'||t==='iframe') return;
          var r = el.getBoundingClientRect();
          if (r.width > 0 && (r.right > cw + 1.5 || r.left < -1.5)) {
            bad.push(t+'.'+(el.className||'-')+'@'+Math.round(r.left)+'..'+Math.round(r.right));
          }
        });
        var res = 'vw'+w+':cw='+cw+',sw='+sw+',bad='+(bad.length ? bad.slice(0,4).join(' ; ') : 'none');
        document.title = (document.title||'') + ' || ' + res;
      }catch(e){ document.title = (document.title||'')+' || vw'+w+':ERR '+e.message; }
      f.remove();
      next();
    };
    f.src = location.pathname + '#inner' + (location.search||'');
  }
  next();
})();
</script>
"""

def run(path):
    html = io.open(path, encoding="utf-8").read()
    html = html.replace("</body>", PROBE + "</body>")
    tmp = os.path.join(tempfile.gettempdir(), "probe2_" + os.path.basename(path))
    io.open(tmp, "w", encoding="utf-8").write(html)
    out = subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--allow-file-access-from-files",
         "--window-size=1400,1200", "--virtual-time-budget=6000",
         "--dump-dom", "file:///" + tmp.replace("\\", "/")],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    m = re.search(r"<title>([^<]*)</title>", out.stdout or "")
    return m.group(1) if m else "(no title)"

for f in FILES:
    print(os.path.basename(f)[:6], "→", run(f).strip())
