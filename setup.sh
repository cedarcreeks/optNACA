#!/usr/bin/env bash
# Create a virtual environment and install all dependencies for the project.
# Works on Python 3.10-3.14. Requires cmake and gfortran on the PATH
# (macOS: `brew install cmake gcc`).
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

echo ">> Using interpreter: $($PYTHON --version)"

# Check the Fortran toolchain XFOIL needs to compile.
for tool in cmake gfortran; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not found. Install it first (macOS: brew install cmake gcc)." >&2
        exit 1
    fi
done

echo ">> Creating virtual environment in $VENV"
"$PYTHON" -m venv "$VENV"

echo ">> Upgrading pip"
"$VENV/bin/pip" install --quiet --upgrade pip

echo ">> Installing requirements (this builds XFOIL from source, may take a minute)"
"$VENV/bin/pip" install -r requirements.txt

echo ">> Verifying XFOIL import"
"$VENV/bin/python" -c "from xfoil import XFoil; print('XFOIL import OK')"

echo
echo "Done. Activate the environment with:  source $VENV/bin/activate"
echo "Then run the pipeline:  python 01_generate_dataset.py  (etc.)"
