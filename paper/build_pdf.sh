#!/bin/sh
# Build the paper PDF locally with the TinyTeX install set up on 2026-06-20.
# Usage: sh paper/build_pdf.sh   (run from anywhere)
export HOME=/home/anatbr/students/noamshakedc/env/tinytex_root
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
cd "$(dirname "$0")"
mkdir -p build
pdflatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null
pdflatex -interaction=nonstopmode -output-directory=build main.tex | tail -3
cp build/main.pdf main.pdf
echo "wrote $(pwd)/main.pdf"
