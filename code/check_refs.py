"""Check labels, references and citations across the chapter sources (read only).

Usage: python3 -I code/check_refs.py [chapters_dir]

Errors (exit status 1): a duplicate label; a \\ref, \\eqref or \\pageref to a label that no chapter
defines; a label, cite key or \\bibitem key not prefixed with its chapter tag (the only exception is
ch:<tag>); a \\cite key with no \\bibitem in the same chapter. Warnings: a \\bibitem that the chapter
never cites, and literal numbers after Chapter, Section, Table, Figure, Theorem and the like in
prose (they may be legitimate citations of the literature; the ones that point inside the book
should be \\ref).
"""
import re
import sys
from collections import Counter
from pathlib import Path

ORDER = ["preface", "intro", "toolkit", "ftns", "hs", "mp", "esf", "py", "pig", "rc", "suff", "coal", "future"]
MATH_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*",
             "eqnarray", "eqnarray*", "displaymath", "flalign", "flalign*", "alignat", "alignat*")
LITERAL = re.compile(r"\b(Chapter|Section|Table|Figure|Theorem|Lemma|Proposition|Corollary|Example|"
                     r"Definition|Remark|Appendix)s?[~\s]+\(?\d")


def root():
    here = Path(__file__).resolve().parent
    return next(p for p in (here, *here.parents) if (p / "book.tex").exists())


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def prose(s):
    s = strip_comments(s)
    for env in MATH_ENVS:
        e = re.escape(env)
        s = re.sub(rf"\\begin\{{{e}\}}.*?\\end\{{{e}\}}", " ", s, flags=re.S)
    s = re.sub(r"\$\$.*?\$\$", " ", s, flags=re.S)
    s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.S)
    return re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " ", s, flags=re.S)


def keys(pattern, s):
    out = []
    for m in re.finditer(pattern, s):
        out += [k.strip() for k in m.group(1).split(",")]
    return out


def prefix(k):
    return k.split(":", 1)[0] if ":" in k else "(none)"


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else root() / "chapters"
    text = {t: strip_comments((d / f"{t}.tex").read_text(encoding="utf8")) for t in ORDER}
    errors, warnings = [], []
    labels = {t: re.findall(r"\\label\{([^}]*)\}", text[t]) for t in ORDER}
    seen = Counter(k for t in ORDER for k in labels[t])
    for k, n in seen.items():
        if n > 1:
            errors.append(f"duplicate label {k} ({n} times)")
    for t in ORDER:
        for k in labels[t]:
            if k != f"ch:{t}" and prefix(k) != t:
                errors.append(f"{t}: label {k} is not prefixed {t}:")
        cites = keys(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", text[t])
        bibs = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", text[t])
        for k in sorted(set(cites)):
            if prefix(k) != t:
                errors.append(f"{t}: cite key {k} is not prefixed {t}:")
            if k not in bibs:
                errors.append(f"{t}: cite key {k} has no \\bibitem in the chapter")
        for k in bibs:
            if prefix(k) != t:
                errors.append(f"{t}: bibitem key {k} is not prefixed {t}:")
        unused = sorted(set(bibs) - set(cites))
        if unused:
            warnings.append(f"{t}: {len(unused)} bibitems never cited, e.g. {', '.join(unused[:4])}")
        for k in keys(r"\\(?:ref|eqref|pageref)\{([^}]*)\}", text[t]):
            if k not in seen:
                errors.append(f"{t}: reference to undefined label {k}")
        hits = [m.group(0) for m in LITERAL.finditer(prose(text[t]))]
        if hits:
            warnings.append(f"{t}: {len(hits)} literal numbers after Chapter, Section, Table and the like, "
                            f"e.g. {', '.join(repr(h) for h in hits[:4])}")
    for w in warnings:
        print("WARN  ", w)
    for e in errors:
        print("ERROR ", e)
    print(f"{len(ORDER)} chapters, {sum(seen.values())} labels, {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
