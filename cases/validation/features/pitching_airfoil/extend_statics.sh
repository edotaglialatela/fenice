#!/usr/bin/env bash
# Continue the static runs of the pitching-airfoil study from their checkpoints (t = 40) to EXT, and test whether the
# dominant Cl frequency (about 2.75 at 3, 6 and 10 degrees) follows the remesh cadence. Run it from the work folder of
# run_workstation.sh (the one that holds fenice/, venv/ and results/):
#
#     cd ~/pitch_work && git -C fenice pull && bash fenice/cases/validation/features/pitching_airfoil/extend_statics.sh
#
# Four jobs side by side, one lane of about six physical cores each (all their hardware threads):
#     S3, S6, S10   resumed to t = EXT (default 100), same settings as before
#     FREQ          S6 resumed to t = 50 with amr_refine_every = 10 (the original runs use 5)
# Nothing in results/pitching_airfoil is touched: the checkpoints are copied. Output: results/pitching_airfoil_ext/
# (report.txt, the logs, the monitor files) and results/pitching_airfoil_ext.tar.gz to send back.
# Smoke test first (a few minutes):  EXT=42 bash .../extend_statics.sh
set -uo pipefail

WORK=${WORK:-$PWD}
EXT=${EXT:-100}
SRC=$WORK/results/pitching_airfoil
OUT=$WORK/results/pitching_airfoil_ext
REPO=$WORK/fenice
CASE_DIR=ibm_airfoil_pitch_amr
CKPT=ibm_airfoil_pitch_restart.npz
FREQ_END=$(( EXT < 50 ? EXT : 50 ))

for d in "$REPO" "$WORK/venv" "$SRC"; do [[ -d $d ]] || { echo "missing folder: $d (run this from the work folder)"; exit 1; }; done
for t in S3_L3 S6_L3 S10_L3; do [[ -f $SRC/$t/$CASE_DIR/$CKPT ]] || { echo "missing checkpoint: $SRC/$t/$CASE_DIR/$CKPT"; exit 1; }; done
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
export FENICE_CASES=$REPO/cases
export PYTHONPATH=$REPO/python${PYTHONPATH:+:$PYTHONPATH}
FENICE=$(python -c "import fenice; print(fenice.exe())")
mkdir -p "$OUT"

# four lanes of whole physical cores; each lane takes all the hardware threads of its cores
NL=4
mapfile -t LANE < <(lscpu -p=CPU,CORE | grep -v '^#' | awk -F, -v nl=$NL '
    { if ($2 in c) c[$2] = c[$2] "," $1; else { c[$2] = $1; o[$2] = n++ } }
    END { per = int(n / nl); if (per < 1) per = 1
          for (k in c) { l = int(o[k] / per); if (l >= nl) l = nl - 1; if (l in L) L[l] = L[l] "," c[k]; else L[l] = c[k] }
          for (i = 0; i < nl; i++) print L[i] }')
echo "lanes: ${LANE[*]}"

run_job() {   # name source_run aoa t_end lane [extra --set args]
    local name=$1 src=$2 aoa=$3 tend=$4 lane=$5; shift 5
    local out=$OUT/$name cpus=${LANE[$lane]}
    mkdir -p "$out/$CASE_DIR"
    cp "$SRC/$src/$CASE_DIR/$CKPT" "$out/$CASE_DIR/$CKPT"
    local thr; thr=$(awk -F, '{print NF}' <<<"$cpus")
    local pin=(); if command -v taskset >/dev/null 2>&1 && [[ -n $cpus ]]; then pin=(taskset -c "$cpus"); fi
    OMP_NUM_THREADS=$thr OMP_PROC_BIND=close OMP_PLACES=threads \
        ${pin[@]+"${pin[@]}"} "$FENICE" --case "IBM Airfoil Pitch" --no-logo --threads "$thr" \
        --set "output_root=$out" --set use_mpi=false --set nx=720 --set ny=360 --set amr_max_level=3 \
        --set "ibm_aoa=$aoa" --set "t_end=$tend" --set resume=true "$@" > "$out/log.txt" 2>&1
}

printf '\n==== resume (EXT = %s, FREQ to %s) ====\n' "$EXT" "$FREQ_END"
date +%T
PIDS=(); NAMES=()
run_job S3   S3_L3  3  "$EXT" 0 & PIDS+=($!); NAMES+=(S3)
run_job S6   S6_L3  6  "$EXT" 1 & PIDS+=($!); NAMES+=(S6)
run_job S10  S10_L3 10 "$EXT" 2 & PIDS+=($!); NAMES+=(S10)
run_job FREQ S6_L3  6  "$FREQ_END" 3 --set amr_refine_every=10 & PIDS+=($!); NAMES+=(FREQ)
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"; echo "$(date +%T)  ${NAMES[$i]} finished (exit $?)"
done

# ---------------------------------------------------------------- report
cat > "$OUT/analyse.py" <<'PYEOF'
import sys
from pathlib import Path

import numpy as np

import fenice

src, out, ext = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])


def monitor(folder):
    try:
        m = fenice.read_monitor(next(f.parent for f in Path(folder).rglob("*_monitor.csv")))
        return np.asarray(m["t"], float), np.asarray(m["Cl"], float)
    except Exception:
        return np.array([]), np.array([])


def joined(old_tag, new_tag):
    t0, c0 = monitor(src / old_tag)
    t1, c1 = monitor(out / new_tag)
    if len(t0) and len(t1):
        k = t1 > t0[-1] + 1e-12
        return np.concatenate([t0, t1[k]]), np.concatenate([c0, c1[k]])
    return (t0, c0) if len(t0) else (t1, c1)


def freq(t, c, a, b):
    w = (t >= a) & (t <= b)
    if w.sum() < 50:
        return float("nan")
    tt = np.linspace(a, b, 8192)
    y = np.interp(tt, t[w], c[w])
    y -= y.mean()
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), tt[1] - tt[0])
    return float(f[1 + np.argmax(s[1:])])


lines = []
say = lambda s="": (print(s), lines.append(s))
say("# Static runs extended from t = 40, NACA 0012 Re = 1000, level 3")
say()
say("Cl in windows of 10 time units: mean, amplitude (half the peak-to-peak), dominant frequency.")
say()
for name, tag, ang in (("S3", "S3_L3", 3), ("S6", "S6_L3", 6), ("S10", "S10_L3", 10)):
    t, c = joined(tag, name)
    if len(t) == 0:
        say(f"{ang} deg: no data")
        continue
    say(f"## {ang} deg (data to t = {t[-1]:.1f}; extension asked for {ext:g})")
    say("| window | mean Cl | amplitude | f |")
    say("|---|---|---|---|")
    amps, mids = [], []
    a = 10.0
    while a + 10 <= t[-1] + 1e-9:
        w = (t >= a) & (t < a + 10)
        if w.sum() > 20:
            amp = 0.5 * (c[w].max() - c[w].min())
            say(f"| {a:g}-{a + 10:g} | {c[w].mean():.4f} | {amp:.4f} | {freq(t, c, a, a + 10):.3f} |")
            if a >= 20:
                amps.append(amp)
                mids.append(a + 5)
        a += 10
    if len(amps) >= 3:
        rate = np.polyfit(mids, np.log(np.maximum(amps, 1e-12)), 1)[0]
        say()
        say(f"amplitude trend from t = 20: d ln(amp)/dt = {rate:+.4f} per time unit (negative = decaying, about 0 = a settled cycle)")
    say()

say("## Does the 2.75 frequency follow the remesh cadence?")
t6, c6 = joined("S6_L3", "S6")
tf, cf = monitor(out / "FREQ")
if len(tf):
    a, b = 40.0, min(t6[-1], tf[-1])
    say(f"window t = {a:g}..{b:g}: amr_refine_every 5 (the run): f = {freq(t6, c6, a, b):.3f}; amr_refine_every 10: f = {freq(tf, cf, a, b):.3f}")
    w5, w10 = (t6 >= a) & (t6 <= b), (tf >= a) & (tf <= b)
    if w5.sum() > 5 and w10.sum() > 5:
        say(f"amplitude {0.5 * (c6[w5].max() - c6[w5].min()):.4f} vs {0.5 * (cf[w10].max() - cf[w10].min()):.4f}; mean {c6[w5].mean():.4f} vs {cf[w10].mean():.4f}")
    say("same frequency: it does not come from the remesh cadence; a different one: it is an artefact of the cadence.")
else:
    say("no FREQ data")
(out / "report.txt").write_text("\n".join(lines) + "\n")
PYEOF
python "$OUT/analyse.py" "$SRC" "$OUT" "$EXT"

# the monitor files, logs and report, small enough to send back
tar -C "$WORK/results" --exclude='*.npz' --exclude='*.vtu' --exclude='*.pvd' -czf "$WORK/results/pitching_airfoil_ext.tar.gz" pitching_airfoil_ext
printf '\nreport: %s/report.txt\nsend back: %s/results/pitching_airfoil_ext.tar.gz\n' "$OUT" "$WORK"
