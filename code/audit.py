r"""House style audit for the chapter sources: hyphenated words and dashes in prose, bold and boxed text.

Usage: python3 -I code/audit.py [chapter.tex ...]

With no arguments it checks chapters/*.tex under the repository root, the nearest directory above this script that contains book.tex. With arguments it checks the given files. It only reads files.

Prose means everything outside math, comments and the arguments of labels, refs, cites, file names and similar commands. Text inside \text{...} and its relatives within math counts as prose.

Output: one line "<file> <count>" per file, then up to 60 indented hits (a hyphen or dash with 25 characters of context on each side), then one BOLD/BOX line for every \textbf, \bfseries, \bf, \boxed or \fbox anywhere in the source, math and comments included.

Exit status: 1 if any hit is not allowlisted, else 2 if a file cannot be read or parsed, else 0. The allowlist has one entry, the DOI fragment 10.1214/24\nobreakdash-BA1463 in chapter rc, which is printed with an ALLOWED marker.
"""
import glob
import os
import re
import sys

USAGE = "usage: python3 -I code/audit.py [chapter.tex ...]"

# The only allowlisted hit, by chapter file name without extension. A hit is allowed when it is flagged only because of this fragment.
ALLOW = {"rc": [r"10.1214/24\nobreakdash-BA1463"]}

HIT = re.compile(r".{0,25}(?<![-\\])(?:-{1,3})(?!-).{0,25}")
BAD = re.compile(r"[A-Za-z}$]-[A-Za-z$\\]|--|^-[A-Za-z]|[A-Za-z]-$")
STYLE = re.compile(r"\\textbf|\\bfseries|\\bf\b|\\boxed|\\fbox")

# Commands whose arguments are not prose: labels, refs, cites, file names, lengths, colours, macro definitions.
PROTECT = {"label", "ref", "eqref", "pageref", "autoref", "cite", "citet", "citep", "citealt",
           "citealp", "citeauthor", "citeyear", "includegraphics", "url", "href", "begin", "end",
           "input", "include", "hypersetup", "definecolor", "color", "textcolor", "graphicspath",
           "cmidrule", "cline", "hspace", "vspace", "setlength", "addtolength", "rowcolor",
           "cellcolor", "columncolor", "bibitem", "DeclareRobustCommand", "newcommand",
           "renewcommand", "numberwithin", "setcounter", "addcontentsline", "kern", "hskip",
           "vskip", "raisebox", "rule", "multicolumn", "multirow", "item", "resizebox",
           "linewidth", "textwidth", "enlargethispage", "phantom", "hphantom", "vphantom"}
# Brace arguments skipped after a protected command, when that is not exactly one.
SKIP_COUNT = {"item": 0, "multicolumn": 2, "rule": 2, "resizebox": 2, "definecolor": 3,
              "setlength": 2, "addtolength": 2, "setcounter": 2, "addcontentsline": 3,
              "DeclareRobustCommand": 2, "newcommand": 2, "renewcommand": 2,
              "numberwithin": 2, "multirow": 2}
COMMAND = re.compile(r"\\([A-Za-z]+)\*?")
MATH_ENV = re.compile(r"\\begin\{((?:equation|align|gather|multline|eqnarray|alignat|flalign"
                      r"|displaymath|math)\*?)\}")
TABULAR = re.compile(r"\\begin\{(?:tabular|array)\*?\}")
# Text mode arguments inside math, which are prose again.
TEXT_ARG = re.compile(r"\\(?:text|textrm|mbox|textit|emph|textup|textnormal)\s*\{")


def match_brace(s, i):
    """s[i] is '{'; return the index of the matching '}'."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise ValueError("unbalanced brace")


def prose_chunks(s):
    """Return the prose pieces of a LaTeX source: text outside math, comments and protected arguments, plus the text mode arguments found inside math."""
    chunks, buf = [], []
    n, i = len(s), 0

    def flush():
        if buf:
            chunks.append("".join(buf))
            buf.clear()

    def find_end(closer, start):
        k = s.find(closer, start)
        while closer == "$" and k > 0 and s[k - 1] == "\\":
            k = s.find(closer, k + 1)
        if k < 0:
            raise ValueError("missing " + closer)
        return k

    def math(seg):
        p = 0
        for m in TEXT_ARG.finditer(seg):
            if m.start() < p:
                continue
            b = m.end() - 1
            e = match_brace(seg, b)
            chunks.append(seg[b + 1:e])
            p = e

    def skip_args(k, name):
        """Return the index after the optional and brace arguments of a protected command."""
        while True:
            while k < n and s[k] == " ":
                k += 1
            if k < n and s[k] == "[":
                depth = 0
                while k < n:
                    if s[k] == "[":
                        depth += 1
                    elif s[k] == "]":
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                k += 1
                continue
            if k < n and s[k] == "(" and name == "cmidrule":
                k = s.index(")", k) + 1
                continue
            break
        for _ in range(SKIP_COUNT.get(name, 1)):
            while k < n and s[k] in " \n":
                k += 1
            if k < n and s[k] == "{":
                k = match_brace(s, k) + 1
        return k

    try:
        while i < n:
            c = s[i]
            # An escaped percent is consumed with its backslash below, so a percent seen here starts a comment.
            if c == "%":
                flush()
                k = s.find("\n", i)
                i = n if k < 0 else k
                continue
            if c == "$":
                flush()
                if s.startswith("$$", i):
                    k = find_end("$$", i + 2)
                    math(s[i:k + 2])
                    i = k + 2
                else:
                    k = find_end("$", i + 1)
                    math(s[i:k + 1])
                    i = k + 1
                continue
            if c == "\\":
                if s.startswith("\\[", i):
                    flush()
                    k = find_end("\\]", i + 2)
                    math(s[i:k + 2])
                    i = k + 2
                    continue
                if s.startswith("\\(", i):
                    flush()
                    k = find_end("\\)", i + 2)
                    math(s[i:k + 2])
                    i = k + 2
                    continue
                m = MATH_ENV.match(s, i)
                if m:
                    flush()
                    endtag = "\\end{%s}" % m.group(1)
                    k = find_end(endtag, i) + len(endtag)
                    math(s[i:k])
                    i = k
                    continue
                m = COMMAND.match(s, i)
                if m:
                    if m.group(1) in PROTECT:
                        flush()
                        k = skip_args(m.end(), m.group(1))
                        if m.group(1) == "begin" and TABULAR.match(s, i):
                            while k < n and s[k] in " \n":
                                k += 1
                            if k < n and s[k] == "[":
                                k = s.index("]", k) + 1
                            if k < n and s[k] == "{":
                                k = match_brace(s, k) + 1
                        i = k
                    else:
                        buf.append(s[i:m.end()])
                        i = m.end()
                    continue
                buf.append(s[i:i + 2])
                i += 2
                continue
            buf.append(c)
            i += 1
    except ValueError as e:
        raise ValueError("%s near line %d" % (e, s.count("\n", 0, i) + 1)) from None
    flush()
    return chunks


def find_root():
    """Return the nearest directory above this script that contains book.tex, or None."""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(d, "book.tex")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def allowed(chapter, hit):
    """True if the hit is flagged only because of an allowlisted fragment of this chapter."""
    return any(f in hit and not BAD.search(hit.replace(f, "")) for f in ALLOW.get(chapter, ()))


def main(argv):
    if "-h" in argv or "--help" in argv:
        print(__doc__.strip())
        return 0
    if any(a.startswith("-") for a in argv):
        print(USAGE, file=sys.stderr)
        return 2
    if argv:
        targets = [(a, a) for a in argv]
    else:
        root = find_root()
        if root is None:
            print("audit.py: no book.tex above %s, so name the files to check" % os.path.dirname(os.path.abspath(__file__)), file=sys.stderr)
            return 2
        paths = sorted(glob.glob(os.path.join(root, "chapters", "*.tex")))
        if not paths:
            print("audit.py: no chapters/*.tex under %s" % root, file=sys.stderr)
            return 2
        targets = [(os.path.relpath(p, root), p) for p in paths]
    failed = errors = False
    for label, path in targets:
        chapter = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                s = fh.read()
            hits = [m.group(0) for t in prose_chunks(s) for m in HIT.finditer(t)]
        except (OSError, ValueError) as e:
            print("%s: cannot audit: %s" % (label, e), file=sys.stderr)
            errors = True
            continue
        bad = [h for h in hits if BAD.search(h)]
        real = [h for h in bad if not allowed(chapter, h)]
        print(label, len(bad))
        for h in real[:60]:
            print("   ", h)
        for h in bad:
            if allowed(chapter, h):
                print("    ALLOWED", h)
        styles = list(STYLE.finditer(s))
        for m in styles:
            print("   BOLD/BOX", s[max(0, m.start() - 20):m.end() + 20].replace("\n", " "))
        if real or styles:
            failed = True
    return 1 if failed else (2 if errors else 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
