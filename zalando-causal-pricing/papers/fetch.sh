#!/bin/bash
# Fetches both papers from arXiv (PDF + LaTeX source) and converts the source to markdown here.
# Outputs are git-ignored: arXiv papers are not redistributed with this repo.
# Needs curl, tar, pandoc (2.9 works), python3.
set -e
cd "$(dirname "$0")"
TMP=$(mktemp -d)

fetch() {  # id
  curl -sL -o "$1.pdf" "https://arxiv.org/pdf/$1"
  mkdir -p "$TMP/$1" && curl -sL -o "$TMP/$1.tar" "https://arxiv.org/e-print/$1"
  tar xf "$TMP/$1.tar" -C "$TMP/$1" 2>/dev/null || gunzip -c "$TMP/$1.tar" > "$TMP/$1/main.tex"
}

convert() {  # id  main.tex  [appendix.tex]
  local id=$1 main=$2 app=$3 src="$TMP/$1"
  # pandoc 2.9 hangs on custom \newcommand macros: expand the simple ones and drop the definitions.
  cat "$src/$main" ${app:+"$src/$app"} \
    | sed -e '/^\\input{/d' -e '/^\\newcommand/d' -e '/^\\bibliography/d' -e '/^\\pdfinfo/,/^}/d' \
          -e 's/\\expec{\([^}]*\)}/\\mathbb{E}[\1]/g' -e 's/\\bsdem/q_{it}^{(b)}/g' \
          -e 's/\\DMLForecaster\\\?/DML Forecaster/g' -e 's/\\DML\b\\\?/DML/g' -e 's/\\Baseline\\\?/TF/g' \
    > "$TMP/$id.tex"
  timeout 180 pandoc "$TMP/$id.tex" -f latex-latex_macros -t gfm --wrap=none -o "$TMP/$id.body.md"
  local title; title=$(grep -o '\\title{[^}]*}' "$src/$main" | head -1 | sed 's/\\title{//;s/}$//')
  { echo "# $title"; echo; echo "arXiv: https://arxiv.org/abs/$id"; echo; echo "## Abstract"; echo
    sed -n '/begin{abstract}/,/end{abstract}/p' "$src/$main" | sed '1d;$d'; echo; cat "$TMP/$id.body.md"; } > "$id.md"
  echo "$id.md: $(grep -c '^#' "$id.md") headings"
}

fetch 2312.15282v2 && convert 2312.15282v2 dml.tex appendix.tex
fetch 2305.14406   && convert 2305.14406 main.tex
rm -rf "$TMP"
