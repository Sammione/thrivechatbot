"""Help-centre articles (markdown files in knowledge/) for policy questions the APIs can't answer."""
from __future__ import annotations

import math
import re
from pathlib import Path

STOP = set("a an the is are was were be to of and or in on for with my me i you your it this that how what "
           "why when can do does did we our us at by from as".split())


def _tok(s):
    out = []
    for w in re.findall(r"[a-z0-9]+", s.lower()):
        if w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


class KnowledgeBase:
    def __init__(self, folder):
        self.chunks = []    # dicts: source, heading, text, tokens
        for p in sorted(Path(folder).glob("**/*")) if Path(folder).exists() else []:
            if p.suffix.lower() not in (".md", ".txt"):
                continue
            heading, buf = p.stem, []
            for line in p.read_text(encoding="utf-8").splitlines() + ["# END"]:
                if line.startswith("#"):
                    if "".join(buf).strip():
                        text = "\n".join(buf).strip()
                        self.chunks.append({"source": p.name, "heading": heading, "text": text,
                                            "tokens": set(_tok(heading + " " + text)),
                                            "head": set(_tok(heading))})
                    heading, buf = line.lstrip("# ").strip(), []
                else:
                    buf.append(line)
        n = max(1, len(self.chunks))
        df = {}
        for c in self.chunks:
            for t in c["tokens"]:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + n / d) for t, d in df.items()}

    def search(self, query: str, k: int = 3):
        q = set(_tok(query))
        # heading matches count double
        scored = [(sum(self.idf.get(t, 0) for t in q & c["tokens"]) + sum(self.idf.get(t, 0) for t in q & c["head"]), c)
                  for c in self.chunks]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: -x[0])
        return [{"source": c["source"], "heading": c["heading"], "text": c["text"][:1500]} for _, c in scored[:k]]
