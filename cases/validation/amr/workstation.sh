#!/usr/bin/env bash
# AMR studies on a many-core Linux / WSL workstation (gcc, Open MPI), packed for the docs.
#
#   bash workstation.sh [WORKDIR]       (default ~/fenice-amr-studies; BRANCH=..., THREADS="1 2 4 8 16 24 48", RANKS="2 4 8 16 24")
#   SMOKE=1 bash workstation.sh /tmp/x   (tiny runs: checks the script itself in a few minutes)
#
# Clones the repository (or updates the clone), builds it (Release, MPI if found), checks it (unit tests, the 38 references,
# amr_mpi.py --check and ibm_ghost.py --check), then runs the AMR studies whose numbers depend on the machine:
#   1. amr_mpi.py on the 720 x 360 airfoil (the docs' case) and on 1440 x 720 (4x the leaves): OpenMP threads and MPI ranks, the
#      fastest of 3 runs per point; every final state must be bit for bit the serial one;
#   2. airfoil_amr.py --table on the 192 x 96 base (the docs' accuracy-per-cost table) and on 384 x 192 (against uniform 1536 x 768).
# Everything else of the AMR docs (errors, flow figures) is the same bits on any machine and is not re-run.
# Runs are cached: if the script stops, run it again and it continues. Every rank stores the whole mesh (0.43 GB on 720 x 360,
# 1.6 GB on 1440 x 720, measured), so rank counts that do not fit in the available memory are skipped.
# At the end: WORKDIR/fenice-amr-results-<date>.tar.gz (logs, JSON summaries, figures, machine info; no large state files),
# also copied to the Windows Downloads folder when found. Hand that file over for the docs update.
# Needs: git, cmake, g++, python3 (numpy, matplotlib: a venv is made if they are missing), Open MPI for the MPI rows.
# Run it on an idle machine; it takes a few hours (the uniform 1536 x 768 reference is the longest run).
set -euo pipefail
WORK=${1:-$HOME/fenice-amr-studies}
BRANCH=${BRANCH:-amr-studies}   # the study options are on this branch until it is merged
THREADS=${THREADS:-"1 2 4 8 16 24 48"}
RANKS=${RANKS:-"2 4 8 16 24"}
OMP_TABLE=${OMP_TABLE:-24}        # threads of the airfoil_amr runs (physical cores)
REPO=https://github.com/edotaglialatela/fenice.git
mkdir -p "$WORK/logs"
cd "$WORK"
exec > >(tee -a "$WORK/logs/workstation.log") 2>&1
stamp() { echo "=== $(date '+%F %T') $*"; }
stamp "start: $(nproc) logical CPUs, $(lscpu | grep 'Model name' | sed 's/.*: *//')"

# --- clone, build -------------------------------------------------------------------------------------------------------------
[ -d repo ] || git clone -q "$REPO" repo
git -C repo fetch -q origin "$BRANCH"
git -C repo checkout -q --detach FETCH_HEAD
R="$WORK/repo"
stamp "build $(git -C "$R" log --oneline -1)"
cmake -S "$R" -B "$R/build/release" -DCMAKE_BUILD_TYPE=Release > logs/cmake.log
cmake --build "$R/build/release" -j "$(nproc)" > logs/build.log 2>&1 || { tail -30 logs/build.log; exit 1; }
MPI=0
if grep -q "MPI enabled" logs/cmake.log && command -v mpiexec > /dev/null; then MPI=1; fi
[ $MPI = 1 ] || echo "!!! no MPI build or mpiexec: the MPI rows and checks are skipped (sudo apt install libopenmpi-dev openmpi-bin)"

# --- python ---------------------------------------------------------------------------------------------------------------------
PY=python3
if ! $PY -c "import numpy, matplotlib" 2> /dev/null; then
    stamp "making a venv for numpy / matplotlib"
    [ -d venv ] || python3 -m venv venv
    venv/bin/pip install -q numpy matplotlib
    PY="$WORK/venv/bin/python"
fi
export PYTHONPATH="$R/python" FENICE_EXE="$R/build/release/fenice"
cd "$R"

# --- checks ---------------------------------------------------------------------------------------------------------------------
stamp "unit tests";  build/release/fenice_tests | tail -1
stamp "references";  $PY cpp/tests/regression.py | tail -1
if [ $MPI = 1 ]; then
    stamp "amr_mpi.py --check";   $PY cases/validation/amr/amr_mpi.py --check | tail -6
    stamp "ibm_ghost.py --check"; $PY cases/validation/features/ibm_ghost.py --check | tail -7
fi

# --- rank counts that fit in memory (every rank holds the whole mesh) --------------------------------------------------------------
avail_gb=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
fit() {   # fit GB_PER_RANK RANKS... -> the rank counts that fit in 85 % of the available memory
    local per=$1; shift
    for r in "$@"; do awk -v r="$r" -v p="$per" -v a="$avail_gb" 'BEGIN { exit !(r * p <= 0.85 * a) }' && printf "%s " "$r"; done
}
RANKS_720=$([ $MPI = 1 ] && fit 0.45 $RANKS || echo 0)
RANKS_1440=$([ $MPI = 1 ] && fit 1.7 $RANKS || echo 0)
stamp "memory available ${avail_gb} GB: ranks on 720x360: ${RANKS_720:-none}; on 1440x720: ${RANKS_1440:-none}"

# --- studies --------------------------------------------------------------------------------------------------------------------
if [ "${SMOKE:-0}" = 1 ]; then
    stamp "SMOKE: tiny studies only"
    $PY cases/validation/amr/amr_mpi.py --mesh 96x48 --steps 10 --threads 1 2 --ranks ${RANKS_720:-0} 2>&1 | tee "$WORK/logs/amr_mpi_smoke.log" | tail -4
else
stamp "amr_mpi.py 720x360";  $PY cases/validation/amr/amr_mpi.py --threads $THREADS --ranks ${RANKS_720:-0} --repeat 3 2>&1 | tee "$WORK/logs/amr_mpi_720x360.log" | grep -v "^\[RUN\]"
stamp "amr_mpi.py 1440x720"; $PY cases/validation/amr/amr_mpi.py --mesh 1440x720 --threads $THREADS --ranks ${RANKS_1440:-0} --repeat 3 2>&1 | tee "$WORK/logs/amr_mpi_1440x720.log" | grep -v "^\[RUN\]"
stamp "airfoil_amr.py 192x96 --table"; $PY cases/validation/amr/airfoil_amr.py --table --threads "$OMP_TABLE" 2>&1 | tee "$WORK/logs/airfoil_amr_192x96.log"
stamp "airfoil_amr.py 384x192 --table"; $PY cases/validation/amr/airfoil_amr.py --base 384x192 --table --threads "$OMP_TABLE" 2>&1 | tee "$WORK/logs/airfoil_amr_384x192.log"
fi

# --- export -----------------------------------------------------------------------------------------------------------------------
stamp "export"
OUT="$WORK/export"
rm -rf "$OUT" && mkdir -p "$OUT"
{
    echo "commit: $(git -C "$R" log --oneline -1)"; echo "date: $(date)"; lscpu; echo; free -g; echo; g++ --version | head -1
    [ $MPI = 1 ] && mpiexec --version 2>&1 | head -1; uname -a
} > "$OUT/machine.txt"
cp "$WORK"/logs/*.log "$OUT/"
for src in "$R"/results/validation/amr_mpi* "$R"/results/validation/airfoil_amr*; do
    [ -d "$src" ] || continue
    d=$(basename "$src")
    # the summaries, figures and per-run timings; the airfoil monitors (load histories); no states, snapshots or checkpoints
    (cd "$R/results/validation" && find "$d" \( -name "*.json" -o -name "*.png" -o -name "wall.txt" -o -name "*_monitor.csv" \) -print0 \
        | xargs -0 -r tar cf - | tar xf - -C "$OUT")
done
TAR="$WORK/fenice-amr-results-$(date +%Y%m%d-%H%M).tar.gz"
tar czf "$TAR" -C "$WORK" export
echo "packed: $TAR ($(du -h "$TAR" | cut -f1))"
if command -v cmd.exe > /dev/null 2>&1; then
    win=$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2> /dev/null | tr -d '\r')" 2> /dev/null || true)
    if [ -n "$win" ] && [ -d "$win/Downloads" ]; then cp "$TAR" "$win/Downloads/" && echo "copied to $win/Downloads/"; fi
fi
stamp "done"
