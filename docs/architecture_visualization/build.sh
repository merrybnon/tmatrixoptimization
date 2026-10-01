#!/usr/bin/env bash
# Build the architecture figures to PDF, SVG and PNG.
#
#   ./build.sh          # every figure
#   ./build.sh tmvae    # just one
#
# SVG is the format to insert into PowerPoint (2016 and later): it stays vector
# on a projector, and right-click -> Convert to Shape makes the labels editable
# in the deck.  Text is converted to paths (--no-fonts) so nothing depends on
# Helvetica being installed wherever the deck is opened.  The PNG is the
# fallback for older Office.
set -euo pipefail
cd "$(dirname "$0")"

figs=("$@")
if [ ${#figs[@]} -eq 0 ]; then figs=(tmvae generic); fi

for f in "${figs[@]}"; do
  f="${f%.tex}"
  echo "== $f"
  pdflatex -halt-on-error -interaction=nonstopmode "$f.tex" > "$f.build.log"
  dvisvgm --pdf --no-fonts --output="$f.svg" "$f.pdf" >> "$f.build.log" 2>&1
  gs -dNOPAUSE -dBATCH -dQUIET -sDEVICE=png16m -r600 \
     -sOutputFile="$f.png" "$f.pdf"
  rm -f "$f.aux" "$f.log" "$f.build.log"
  ls -l "$f.pdf" "$f.svg" "$f.png"
done
