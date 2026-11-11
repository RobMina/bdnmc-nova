#!/bin/bash
# One job, two stages: (1) generate BdNMC events with the JAX port in its own image (needs glibc >= 2.27, so not
# NOvA's SL7 image) and write TextFileGen input; (2) run NOvASoft's simulation on that file in NOvA's SL7 image.
# The job runs where both images are reachable (a grid worker with CVMFS, a Slurm node); a compute-endpoint
# adapter (hep-endpoints) submits it through bdnmc_nova.jobspec, which sets the inputs below and embeds the card and
# the frame configuration. The stages can also run as two jobs at two sites (STAGES=gen, then STAGES=sim with the
# first job's events.txt), e.g. generation on a GPU cluster without CVMFS and the simulation on the FNAL grid.
# The file lives in bdnmc_nova/data/ (packaged); jobs/two_stage.sh links to it.
#
#   STAGES       gen,sim (default), gen, or sim
#   stage gen:   GEN_IMAGE   the BdNMC JAX image (RobMina/BdNMC gpu/containers/bdnmc-jax.def)
#                CARD        a BdNMC parameter card;  BDNMC_NOVA_CONFIG  the detector-frame configuration (private)
#                TRIALS, CHANNEL, GPU (1: pass --nv), PRECISION (float32 | float64), BIND (extra -B paths)
#                BDNMC_NOVA_SRC  a bdnmc-nova checkout as the job sees it (default: the one holding this script)
#   stage sim:   NOVA_SETUP  the NOvASoft setup command (from the experiment's own documentation)
#                NOVA_FCL    the TextFileGen simulation FCL (experiment configuration, not in this repository); its
#                            TextFileGen module reads events.txt from the working directory, which is $OUT
#                NOVA_IMAGE  NOvA's SL7 image (e.g. the Fermilab SL7 image on /cvmfs/singularity.opensciencegrid.org);
#                            empty: run NOvASoft in the job's own environment (a grid job already in that image)
#                EVENTS      (sim only) the TextFileGen input to simulate, copied to $OUT/events.txt
#                NEVENTS     events to simulate (default: all events of events.txt)
#                DUMP_PRIMARIES  a dump_primaries.py to run on the art file in the NOvASoft environment (primaries.csv)
#                CVMFSEXEC   a prepared cvmfsexec tree (github.com/cvmfs/cvmfsexec, `makedist` for the host's OS) on a
#                            site without a /cvmfs mount: the job copies it to node-local $CVMFS_LOCAL (default /tmp),
#                            mounts CVMFS_REPOS there with a private cache (CVMFS_QUOTA_MB, default 6000), runs the
#                            simulation in NOVA_IMAGE (which may itself be on CVMFS) with `apptainer --userns`, and
#                            deletes the copy and its cache. CVMFS_SERVER_URL (default: the FNAL and BNL Stratum-1s),
#                            CVMFS_HTTP_PROXY (default DIRECT).
# Outputs in $OUT (default $BDNMC_OUT, the hep-endpoints output directory, else ./out): events.txt (TextFileGen
# input), summary.json, and the simulation's art file (sim.root), its log (sim.log) and primaries.csv.
set -eo pipefail
STAGES=${STAGES:-gen,sim}
case ",$STAGES," in *,gen,*) GEN=1 ;; esac
case ",$STAGES," in *,sim,*) SIM=1 ;; esac
[ -n "$GEN$SIM" ] || { echo "STAGES=$STAGES: gen, sim or gen,sim" >&2; exit 2; }
OUT=${OUT:-${BDNMC_OUT:-$PWD/out}}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd -P)  # canonical: a symlinked scratch path is not visible inside a contained image

if [ -n "$GEN" ]; then
    : "${GEN_IMAGE:?}" "${CARD:?}" "${BDNMC_NOVA_CONFIG:?}"
    HERE=${BDNMC_NOVA_SRC:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}
    B="-B $OUT,$HERE,$(dirname "$CARD"),$(dirname "$BDNMC_NOVA_CONFIG")${BIND:+,$BIND}"
    echo "==== $(date +%T) stage 1: BdNMC (JAX port) in $GEN_IMAGE"
    apptainer exec ${GPU:+--nv} $B --env BDNMC_JAX_PRECISION="${PRECISION:-float32}" --env PYTHONPATH="$HERE" \
        "$GEN_IMAGE" python -m bdnmc_nova.generate "$CARD" --out "$OUT/events.txt" --config "$BDNMC_NOVA_CONFIG" \
        --trials "${TRIALS:-16777216}" ${CHANNEL:+--channel "$CHANNEL"} --summary "$OUT/summary.json"
elif [ -n "$EVENTS" ] && [ "$EVENTS" != "$OUT/events.txt" ]; then
    cp "$EVENTS" "$OUT/events.txt"
fi

if [ -n "$SIM" ]; then
    : "${NOVA_SETUP:?}" "${NOVA_FCL:?}"
    [ -s "$OUT/events.txt" ] || { echo "no events: $OUT/events.txt (stage gen, or EVENTS)" >&2; exit 3; }
    N=${NEVENTS:--1}
    [ "$N" -ge 0 ] || N=$(awk '!/^#/ && NF == 2' "$OUT/events.txt" | wc -l | tr -d ' ')  # event header lines
    CMD="$NOVA_SETUP && cd '$OUT' && nova -c '$NOVA_FCL' -n $N -o sim.root"
    [ -z "$DUMP_PRIMARIES" ] || CMD="$CMD && python '$DUMP_PRIMARIES' sim.root --out primaries.csv"
    echo "==== $(date +%T) stage 2: NOvASoft simulation of $N events in ${NOVA_IMAGE:-the job environment}"
    WRAP=() USERNS=() LOCAL=""
    if [ -n "$CVMFSEXEC" ]; then
        : "${NOVA_IMAGE:?CVMFSEXEC runs the simulation in NOVA_IMAGE}"
        LOCAL=$(mktemp -d "${CVMFS_LOCAL:-/tmp}/bdnmc-cvmfs.XXXXXXXX")
        cp -a "$CVMFSEXEC" "$LOCAL/cvmfsexec"
        CONF="$LOCAL/cvmfsexec/dist/etc/cvmfs"
        mkdir -p "$LOCAL/cache" "$CONF/domain.d"
        S1="http://cvmfs-s1fnal.opensciencegrid.org:8000/cvmfs/@fqrn@;http://cvmfs-s1bnl.opensciencegrid.org:8000/cvmfs/@fqrn@"
        echo "CVMFS_SERVER_URL=\"${CVMFS_SERVER_URL:-$S1}\"" > "$CONF/domain.d/opensciencegrid.org.local"
        printf 'CVMFS_HTTP_PROXY="%s"\nCVMFS_CACHE_BASE=%s\nCVMFS_QUOTA_LIMIT=%s\nCVMFS_TIMEOUT=30\nCVMFS_TIMEOUT_DIRECT=30\n' \
            "${CVMFS_HTTP_PROXY:-DIRECT}" "$LOCAL/cache" "${CVMFS_QUOTA_MB:-6000}" > "$CONF/default.local"
        REPOS=${CVMFS_REPOS:-config-osg.opensciencegrid.org fermilab.opensciencegrid.org nova.opensciencegrid.org nova-development.opensciencegrid.org singularity.opensciencegrid.org}
        WRAP=("$LOCAL/cvmfsexec/cvmfsexec" $REPOS --)
        USERNS=(--userns)
        echo "     CVMFS through cvmfsexec in $LOCAL: $REPOS"
    fi
    RC=0
    if [ -n "$NOVA_IMAGE" ]; then
        B="-B $OUT,$(dirname "$NOVA_FCL")${DUMP_PRIMARIES:+,$(dirname "$DUMP_PRIMARIES")}${BIND:+,$BIND}"
        "${WRAP[@]}" apptainer exec "${USERNS[@]}" --cleanenv -B /cvmfs $B "$NOVA_IMAGE" bash -lc "$CMD" \
            > "$OUT/sim.log" 2>&1 || RC=$?
    else
        bash -c "$CMD" > "$OUT/sim.log" 2>&1 || RC=$?
    fi
    if [ -n "$LOCAL" ]; then
        echo "     CVMFS cache used: $(du -sh "$LOCAL/cache" 2>/dev/null | cut -f1)"
        rm -rf "$LOCAL"
    fi
    [ "$RC" = 0 ] || { echo "stage 2 failed ($RC); tail of $OUT/sim.log:" >&2; tail -n 30 "$OUT/sim.log" >&2; exit "$RC"; }
fi
echo "==== $(date +%T) done: $OUT"
