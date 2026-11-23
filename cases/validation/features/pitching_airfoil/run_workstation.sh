#!/usr/bin/env bash
# Pitching NACA 0012 at Re = 1000: a moving immersed body on an adaptive mesh, validated against the static airfoil, one script from the clone to the figures.
#
#   bash run_workstation.sh                 # the full job (about 2-4 h on 24 cores at the default resolution; the benchmark prints the estimate first)
#   QUICK=1 bash run_workstation.sh         # a few-minute smoke test of the whole chain (coarse mesh, short times): run it first
#
# Physics. NACA 0012, Re = 1000 (chord and inlet speed 1), Ma 0.2, a 12 x 6 box, quarter chord at (3, 3), slip walls at +-3 chords.
# The airfoil rests at 0 degrees until T_PRE (the wake develops), turns nose-up about its quarter chord at a constant rate, then holds the angle.
#   A  slow  0 -> 6 degrees in 6 time units (reduced rate k = alpha_dot c / 2U = 0.009): stays below the critical angle (about 8 degrees at this Re,
#            Kurtulus 2015, 2019), so the flow ends steady; it must end on the STATIC 6-degree airfoil (validation of the moving wall).
#   B  fast  0 -> 15 degrees in 1.5 time units (k = 0.087): above the critical angle, a lift overshoot, separation and vortex shedding (showcase).
#   static   0, 3, 6, 10, 15 degrees on the same mesh: the polar, and the steady / unsteady transition between 6 and 10 degrees.
# Mesh: base BASE_NX x BASE_NY with LEVEL refinement levels, the body always at the finest level (default 720 x 360, level 2: 240 cells per chord).
# A is also run one level coarser to measure mesh convergence of the pitching lift.
#
# Environment (all optional):
#   REPO_URL REF            what to clone (default: the GitHub repository, branch main)
#   WORK                    working directory (default ./fenice_pitching): clone in $WORK/fenice, results in $WORK/results
#   REPO_DIR                use this existing checkout instead of cloning; SKIP_SETUP=1 also skips the build and the Python environment
#   THREADS                 cores to use (default: the physical cores); JOBS: runs side by side (default 3), each with THREADS/JOBS cores, pinned
#   BASE_NX BASE_NY LEVEL   the mesh; FRAME_DT time between the snapshots of run B (default 0.5); KEEP_VTU=1 keeps the snapshots after the frames
#   REF_CSV                 optional reference polar (columns alpha_deg,Cl,Cd) to compare the static runs with (not gated)
set -euo pipefail

QUICK=${QUICK:-0}
REPO_URL=${REPO_URL:-https://github.com/edotaglialatela/fenice.git}
REF=${REF:-main}
WORK=${WORK:-$PWD/fenice_pitching}
SKIP_SETUP=${SKIP_SETUP:-0}
PYTHON=${PYTHON:-python3}

if [[ $QUICK == 1 ]]; then   # the chain end to end, in minutes: coarse mesh, short times
    BASE_NX=${BASE_NX:-96}; BASE_NY=${BASE_NY:-48}; LEVEL=${LEVEL:-2}
    T_PRE=${T_PRE:-1.0}; T_RAMP_A=${T_RAMP_A:-1.5}; T_RAMP_B=${T_RAMP_B:-0.5}; T_END=${T_END:-5.0}; T_END_STATIC=${T_END_STATIC:-5.0}
    FRAME_DT=${FRAME_DT:-0.5}; BENCH_STEPS=${BENCH_STEPS:-20}
else
    BASE_NX=${BASE_NX:-720}; BASE_NY=${BASE_NY:-360}; LEVEL=${LEVEL:-2}
    T_PRE=${T_PRE:-15}; T_RAMP_A=${T_RAMP_A:-6}; T_RAMP_B=${T_RAMP_B:-1.5}; T_END=${T_END:-45}; T_END_STATIC=${T_END_STATIC:-40}
    FRAME_DT=${FRAME_DT:-0.5}; BENCH_STEPS=${BENCH_STEPS:-150}
fi
DALPHA_A=${DALPHA_A:-6}; DALPHA_B=${DALPHA_B:-15}
STATICS=${STATICS:-"0 3 6 10 15"}
JOBS=${JOBS:-3}
KEEP_VTU=${KEEP_VTU:-0}

say() { printf '\n==== %s ====\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }; }

# ---------------------------------------------------------------- 1. clone, build, Python
mkdir -p "$WORK"
if [[ -n ${REPO_DIR:-} ]]; then
    REPO=$(cd "$REPO_DIR" && pwd)
else
    REPO=$WORK/fenice
    if [[ $SKIP_SETUP != 1 ]]; then
        need git
        say "clone $REPO_URL ($REF)"
        if [[ -d $REPO/.git ]]; then git -C "$REPO" fetch --quiet origin && git -C "$REPO" checkout --quiet "$REF" && git -C "$REPO" pull --quiet --ff-only origin "$REF" || true
        else git clone --quiet --branch "$REF" "$REPO_URL" "$REPO"; fi
    fi
fi
echo "repository: $REPO ($(git -C "$REPO" log --oneline -1 2>/dev/null || echo 'no git'))"

if [[ $SKIP_SETUP != 1 ]]; then
    need cmake; need "$PYTHON"
    say "build (OpenMP, no MPI: one adaptive run per process is the fast way on one node)"
    ( cd "$REPO" && cmake --preset release -DFV_USE_MPI=OFF >/dev/null && cmake --build build/release -j "$(nproc)" 2>&1 | tail -3 )
    say "Python environment"
    "$PYTHON" -m venv "$WORK/venv"
    # shellcheck disable=SC1091
    source "$WORK/venv/bin/activate"
    PYTHON=python
    pip install --quiet --upgrade pip
    pip install --quiet numpy scipy matplotlib pillow
    pip install --quiet -e "$REPO/python"
fi
export FENICE_CASES=$REPO/cases
export PYTHONPATH=$REPO/python${PYTHONPATH:+:$PYTHONPATH}
FENICE=$("$PYTHON" -c "import fenice; print(fenice.exe())")
STUDY=$REPO/cases/validation/features/pitching_airfoil
RES=$WORK/results/pitching_airfoil$([[ $QUICK == 1 ]] && echo _quick)
mkdir -p "$RES"
echo "solver: $FENICE"

say "unit test of the rotation schedule"
if [[ -x $(dirname "$FENICE")/fenice_tests ]]; then "$(dirname "$FENICE")/fenice_tests" prescribed_rotation | tail -1; fi

# ---------------------------------------------------------------- 2. cores: one logical CPU per physical core, split into lanes
NPROC=$(nproc)
CORES=()
if command -v lscpu >/dev/null 2>&1; then
    mapfile -t CORES < <(lscpu -p=CPU,CORE 2>/dev/null | grep -v '^#' | sort -t, -k2n -k1n | awk -F, '!seen[$2]++{print $1}')
fi
if (( ${#CORES[@]} == 0 )); then for ((i = 0; i < NPROC; i++)); do CORES+=("$i"); done; fi
THREADS=${THREADS:-${#CORES[@]}}
(( THREADS > ${#CORES[@]} )) && THREADS=${#CORES[@]}
(( JOBS > THREADS )) && JOBS=$THREADS
PER=$(( THREADS / JOBS ))
LANE_CPUS=()
for ((j = 0; j < JOBS; j++)); do
    LANE_CPUS+=("$(IFS=,; echo "${CORES[*]:$((j * PER)):$PER}")")
done
echo "physical cores: ${#CORES[@]}; using $THREADS in $JOBS lanes of $PER threads: ${LANE_CPUS[*]}"

# ---------------------------------------------------------------- 3. the runs
pi_over_180() { awk 'BEGIN{printf "%.15f", atan2(0,-1)/180}'; }
omega_for() { awk -v d="$1" -v T="$2" 'BEGIN{printf "%.15f", -d*atan2(0,-1)/180/T}'; }   # nose-up is clockwise: negative
sum() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.10g", a+b}'; }
lvl_a=$(( LEVEL - 1 ))   # the coarse twin of A (mesh convergence), if there is a level below

# tag kind alpha_final_deg omega from until level t_end
RUNS_TSV=$RES/runs.tsv
: > "$RUNS_TSV"
declare -A SETS COST
add_run() {   # tag kind alpha omega from until level t_end cost
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >> "$RUNS_TSV"
    local s="nx=$BASE_NX ny=$BASE_NY amr_max_level=$7 t_end=$8"
    if [[ $2 == static ]]; then s+=" ibm_aoa=$3"
    else s+=" ibm_omega=$4 ibm_omega_from=$5 ibm_omega_until=$6"; fi
    SETS[$1]=$s
    COST[$1]=$9
}
WA=$(omega_for "$DALPHA_A" "$T_RAMP_A"); WB=$(omega_for "$DALPHA_B" "$T_RAMP_B")
add_run "B_L$LEVEL"   pitch  "$DALPHA_B" "$WB" "$T_PRE" "$(sum "$T_PRE" "$T_RAMP_B")" "$LEVEL" "$T_END" 11
add_run "A_L$LEVEL"   pitch  "$DALPHA_A" "$WA" "$T_PRE" "$(sum "$T_PRE" "$T_RAMP_A")" "$LEVEL" "$T_END" 11
for a in $STATICS; do add_run "S${a}_L$LEVEL" static "$a" 0 0 0 "$LEVEL" "$T_END_STATIC" 10; done
(( lvl_a >= 0 )) && add_run "A_L$lvl_a"   pitch  "$DALPHA_A" "$WA" "$T_PRE" "$(sum "$T_PRE" "$T_RAMP_A")" "$lvl_a" "$T_END" 2 || true

run_tag() {   # tag threads cpus [extra sets...]
    local tag=$1 thr=$2 cpus=$3; shift 3
    local out=$RES/$tag
    mkdir -p "$out"
    local pin=()
    if [[ -n $cpus ]] && command -v taskset >/dev/null 2>&1; then pin=(taskset -c "$cpus"); fi
    local sets=()
    # shellcheck disable=SC2206
    sets=(${SETS[$tag]})
    local args=()
    for s in "${sets[@]}" "$@"; do args+=(--set "$s"); done
    OMP_NUM_THREADS=$thr OMP_PROC_BIND=close OMP_PLACES=cores \
        ${pin[@]+"${pin[@]}"} "$FENICE" --case "IBM Airfoil Pitch" --no-logo --threads "$thr" --set "output_root=$out" --set use_mpi=false \
        "${args[@]}" > "$out/log.txt" 2>&1
}

say "benchmark: $JOBS runs side by side, $BENCH_STEPS steps each"
BENCH_PIDS=()
for ((j = 0; j < JOBS; j++)); do
    SETS["bench$j"]="nx=$BASE_NX ny=$BASE_NY amr_max_level=$LEVEL t_end=1000 ibm_aoa=10 max_steps=$BENCH_STEPS monitor_interval=1"
    ( run_tag "bench$j" "$PER" "${LANE_CPUS[$j]}" ) &
    BENCH_PIDS+=($!)
done
for p in "${BENCH_PIDS[@]}"; do wait "$p"; done
LOOP=$(grep -o 'time loop [0-9.]*' "$RES/bench0/log.txt" | tail -1 | awk '{print $3}')
DT=$("$PYTHON" "$STUDY/pitching_airfoil.py" dt "$RES/bench0")
LEAVES=$(grep -o '[0-9]* leaves at the end' "$RES/bench0/log.txt" | tail -1 | awk '{print $1}')
echo "time step dt = $DT, leaves after $BENCH_STEPS steps: ${LEAVES:-?}, $LOOP s for $BENCH_STEPS steps per lane (the start-up included: the estimate is an upper bound)"
"$PYTHON" - "$DT" "$LOOP" "$BENCH_STEPS" "$RUNS_TSV" "$JOBS" <<'PYEOF'
import sys
dt, loop, n, tsv, jobs = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
per_step = loop / n
rows = [l.split("\t") for l in open(tsv)]
cost = 0.0
for r in rows:
    steps = float(r[7]) / dt * (0.5 if int(r[6]) < max(int(x[6]) for x in rows) else 1.0)   # one level coarser: dt twice as long
    cost += steps * per_step
print(f"estimate: {cost / jobs / 3600:.1f} h wall-clock for {len(rows)} runs in {jobs} lanes (the fine runs take about {float(rows[0][7]) / dt * per_step / 3600:.1f} h each; dt shrinks a little as the flow develops)")
PYEOF
SI=$(awk -v f="$FRAME_DT" -v dt="$DT" 'BEGIN{n=int(f/dt+0.5); if(n<1)n=1; print n}')
rm -rf "$RES"/bench*

say "runs (logs in $RES/<tag>/log.txt)"
# greedy balance of the runs over the lanes by their estimated cost
declare -a LANE_TAGS LANE_LOAD
for ((j = 0; j < JOBS; j++)); do LANE_TAGS[j]=""; LANE_LOAD[j]=0; done
for tag in $(for t in "${!COST[@]}"; do [[ $t == bench* ]] || echo "${COST[$t]} $t"; done | sort -rn | awk '{print $2}'); do
    best=0
    for ((j = 1; j < JOBS; j++)); do (( LANE_LOAD[j] < LANE_LOAD[best] )) && best=$j; done
    LANE_TAGS[best]+=" $tag"
    LANE_LOAD[best]=$(( LANE_LOAD[best] + COST[$tag] ))
done
LANE_PIDS=()
for ((j = 0; j < JOBS; j++)); do
    (
        for tag in ${LANE_TAGS[$j]}; do
            extra=()
            if [[ $tag == B_L$LEVEL ]]; then extra=(save_interval="$SI" vtk_format=binary); fi
            echo "[lane $j] $tag start $(date +%H:%M:%S)"
            run_tag "$tag" "$PER" "${LANE_CPUS[$j]}" ${extra[@]+"${extra[@]}"} || { echo "[lane $j] $tag FAILED (see $RES/$tag/log.txt)"; exit 1; }
            echo "[lane $j] $tag done  $(date +%H:%M:%S)"
        done
    ) &
    LANE_PIDS+=($!)
done
FAIL=0
for p in "${LANE_PIDS[@]}"; do wait "$p" || FAIL=1; done
(( FAIL == 0 )) || { echo "a run failed: see the logs above"; exit 1; }

# ---------------------------------------------------------------- 4. post-processing
say "post-processing"
set +e
"$PYTHON" "$STUDY/pitching_airfoil.py" post "$RES" ${REF_CSV:+--reference "$REF_CSV"}
POST=$?
"$PYTHON" "$STUDY/pitching_airfoil.py" frames "$RES/B_L$LEVEL" --runs "$RUNS_TSV" ${KEEP_VTU:+--keep-vtu "$KEEP_VTU"}
set -e
if command -v ffmpeg >/dev/null 2>&1 && ls "$RES/frames"/frame_*.png >/dev/null 2>&1; then
    ffmpeg -y -loglevel error -framerate 12 -i "$RES/frames/frame_%04d.png" -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "$RES/pitch_B_vorticity.mp4" || true
fi
( cd "$WORK/results" && tar czf "$(basename "$RES").tar.gz" --exclude='*.vtu' --exclude='*.pvtu' "$(basename "$RES")" ) || true
say "finished"
echo "results:  $RES"
echo "report:   $RES/report.md   figures: $RES/*.png   movie: $RES/pitch_B_vorticity.{gif,mp4}"
echo "archive:  $WORK/results/$(basename "$RES").tar.gz   (copy this back)"
exit "$POST"
