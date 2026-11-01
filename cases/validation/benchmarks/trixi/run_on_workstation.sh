#!/usr/bin/env bash
# Everything for the Trixi.jl comparison on a Linux / WSL workstation, from nothing: clone, build, Python environment,
# Julia + Trixi.jl, run. This one file is enough (download it and run it); arguments go to compare_trixi.py.
#
#   bash run_on_workstation.sh                       # full comparison, threads 1 and 44
#   bash run_on_workstation.sh --threads 44 --tgv-n 32 --tgv-levels 3
#   FENICE_DIR=~/src/fenice bash run_on_workstation.sh   # where the repository lives (default ~/fenice)
#   SKIP_TRIXI=1 bash run_on_workstation.sh          # only the Fenice side (no Julia download)
#
# Needs: git, curl, a C++17 compiler, cmake >= 3.25, python3 with venv; the missing ones are installed with apt-get
# (sudo) on Debian / Ubuntu.
set -euo pipefail

REPO="${FENICE_REPO:-https://github.com/edotaglialatela/fenice.git}"
DIR="${FENICE_DIR:-$HOME/fenice}"

need=()
for c in git curl g++ cmake python3; do command -v "$c" >/dev/null || need+=("$c"); done
tmpv=$(mktemp -d); python3 -m venv "$tmpv/v" >/dev/null 2>&1 || need+=("python3-venv"); rm -rf "$tmpv"
command -v ninja >/dev/null || need+=("ninja-build")
if [ ${#need[@]} -gt 0 ]; then
  echo "[setup] installing: ${need[*]}"
  sudo apt-get update && sudo apt-get install -y build-essential git curl cmake ninja-build python3 python3-venv
fi
cmake_ver=$(cmake --version | head -1 | awk '{print $3}')
if [ "$(printf '%s\n3.25\n' "$cmake_ver" | sort -V | head -1)" != "3.25" ]; then
  echo "cmake $cmake_ver is older than 3.25: install a newer one (pip install cmake, or the Kitware apt repository)"; exit 2
fi

if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone --depth 1 "$REPO" "$DIR"
fi
cd "$DIR"
git log -1 --format='[repo] %h %s'

# OpenMP, no MPI, no tests: the comparison needs the solver only
cmake --preset release -DFV_USE_MPI=OFF -DFV_BUILD_TESTS=OFF
cmake --build --preset release -j "$(nproc)"
test -x build/release/fenice

VENV="$DIR/.venv"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet numpy matplotlib -e python
export FENICE_EXE="$DIR/build/release/fenice"
export PYTHONPATH="$DIR/python"

if [ -z "${SKIP_TRIXI:-}" ]; then
  "$VENV/bin/python" cases/validation/benchmarks/compare_trixi.py --setup
  exec "$VENV/bin/python" cases/validation/benchmarks/compare_trixi.py "$@"
fi
exec "$VENV/bin/python" cases/validation/benchmarks/compare_trixi.py --fenice-only "$@"
