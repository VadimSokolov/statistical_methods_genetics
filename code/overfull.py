import re,sys
log=open(sys.argv[1],errors='replace').read()
cur=None
for m in re.finditer(r'\(\./chapters/(\w+)\.tex|Overfull \\hbox \(([\d.]+)pt too wide\) (?:in paragraph at lines (\d+)--(\d+)|detected at line (\d+)|in alignment at lines (\d+)--(\d+))',log):
    if m.group(1): cur=m.group(1); continue
    w=float(m.group(2))
    if w<float(sys.argv[2] if len(sys.argv)>2 else 3): continue
    a=m.group(3) or m.group(5) or m.group(6); b=m.group(4) or m.group(5) or m.group(7)
    print(f"{cur:6s} {w:7.1f}pt lines {a}-{b}")
