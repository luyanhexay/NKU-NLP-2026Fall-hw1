#!/usr/bin/env bash
# Build report/main.pdf.
#
# The LaTeX source refers to tables/ and figs/ by relative path, so the
# compiler has to run with report/ as the working directory - compiling
# report/main.tex from the project root fails to find them.  This script
# exists so that is not something anyone has to remember.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Regenerate the generated inputs first, so the PDF cannot be stale.
#
# The exit status matters: if either step fails, the generated files keep
# their previous contents and the PDF builds happily from stale tables and
# macros.  Swallowing the error here would hide exactly that.
for step in scripts.collect_results scripts.make_figs; do
    if ! uv run --project "$root" python -m "$step" >"/tmp/build_report_${step##*.}.log" 2>&1; then
        echo "FAILED: $step" >&2
        tail -20 "/tmp/build_report_${step##*.}.log" >&2
        exit 1
    fi
done

cd "$root/report"

# Markdown emphasis and heading markers are not LaTeX.  One leaked into the
# source once and rendered as literal "**" in the PDF, which no error message
# would have caught - so check for it explicitly.
if grep -nE '\*\*|(^|[^\\])#{1,6}[ ]' main.tex tables/*.tex; then
    echo "Markdown markup found in LaTeX source (see above)" >&2
    exit 1
fi

# A table row that ends in a single backslash makes LaTeX read "\ " (a space
# command) instead of "\\", so the tabular never ends and the compiler spins
# forever at full CPU without printing an error.  Catch it instead.
if grep -nP '(?<!\\)\\$' tables/*.tex; then
    echo "Table row ends with a single backslash - it needs \"\\\\\" (see above)" >&2
    exit 1
fi

rm -rf build
mkdir -p build

# Pass 1 writes the .aux; bibtex then builds the bibliography (the ACL style
# uses natbib with acl_natbib.bst); the remaining passes resolve the
# cross-references against it.
xelatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null 2>&1 || true
bibtex build/main >/dev/null 2>&1 || true
for _ in 1 2 3; do
    xelatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null 2>&1 || true
done

if grep -qE '^!' build/main.log; then
    echo "LaTeX errors:" >&2
    grep -nE '^!' -A3 build/main.log >&2
    exit 1
fi

# Keep a copy at a stable, committed path.  report/build/ is gitignored
# (it holds .aux/.log and a rebuild regenerates everything), but the PDF is
# the thing the assignment asks for, so it cannot live only there.
cp build/main.pdf main.pdf
echo "wrote $root/report/main.pdf"
