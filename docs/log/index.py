#!/usr/bin/env python3
"""Print the work log as a table. Usage: index.py [--stream bc|frame|...] [--branch NAME]"""
import argparse, pathlib, re
ap = argparse.ArgumentParser(); ap.add_argument("--stream"); ap.add_argument("--branch"); a = ap.parse_args()
for p in sorted(pathlib.Path(__file__).parent.glob("2*.md")):
    t = p.read_text(); m = re.match(r"---\n(.*?)\n---\n# ([^\n]*)", t, re.S)
    if not m: print(f"{p.name}: BAD FRONTMATTER"); continue
    f = dict(l.split(":", 1) for l in m[1].splitlines() if ":" in l and not l.startswith(" "))
    f = {k.strip(): v.split("#")[0].strip() for k, v in f.items()}
    if a.stream and f.get("stream") != a.stream or a.branch and f.get("branch") != a.branch: continue
    print(f"{p.stem[:10]}  {f.get('stream',''):6} {f.get('branch',''):18} {f.get('status',''):12} {m[2]}")
