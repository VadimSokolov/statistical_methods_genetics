#!/bin/bash
# Usage: code/build_one.sh <tag>   builds chapter <tag> alone with the full book preamble
# Output: ${SMG_BUILD_DIR:-/tmp/smg_single}/<tag>/one.pdf and one.log, never inside the repo (set SMG_BUILD_DIR to build somewhere else)
[ -n "$1" ] || { echo "usage: code/build_one.sh <tag>" >&2; exit 2; }
tag=$1
unset CDPATH
src=${BASH_SOURCE[0]}
while [ -L "$src" ]; do
  d=$(cd -P "$(dirname "$src")" && pwd)
  src=$(readlink "$src")
  case $src in /*) ;; *) src=$d/$src ;; esac
done
B=$(cd -P "$(dirname "$src")/.." && pwd)
D=${SMG_BUILD_DIR:-/tmp/smg_single}/$tag
mkdir -p "$D"
cd "$D" || exit 1
ln -sfn "$B/chapters" chapters
ln -sfn "$B/figures" figures
sed -n '1,/\\begin{document}/p' "$B/book.tex" > one.tex
printf '\\mainmatter\n\\chapterfile{%s}\n\\end{document}\n' "$tag" >> one.tex
for i in 1 2; do pdflatex -interaction=nonstopmode one.tex > /dev/null 2>&1; done
echo "errors: $(grep -c '^!' one.log)"
grep -A2 '^!' one.log | head -20
python3 -I "$B/code/overfull.py" one.log 3
echo "undefined refs (ch: refs to other chapters are expected): $(grep -o "Reference \`[^']*' on" one.log | grep -v "\`ch:" | sort -u | head -20)"
echo "pages: $(pdfinfo one.pdf | awk '/Pages/{print $2}')"
