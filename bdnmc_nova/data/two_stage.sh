#!/bin/bash
# One job, two stages: (1) generate BdNMC events with the JAX port in its own image (needs glibc >= 2.27, so not
# NOvA's SL7 image) and write TextFileGen input; (2) run NOvASoft's simulation on that file in NOvA's SL7 image.
# The job runs where both images are reachable (a grid worker with CVMFS, a Slurm node); a compute-endpoint
# adapter (hep-endpoints) submits it through bdnmc_nova.jobspec, which sets the inputs below and embeds the card and
# the frame configuration. The file lives in bdnmc_nova/data/ (packaged); jobs/two_stage.sh links to it.
#
#   GEN_IMAGE    the BdNMC JAX image (RobMina/BdNMC gpu/containers/bdnmc-jax.def)
#   NOVA_IMAGE   NOvA's SL7 image (e.g. the Fermilab SL7 image on /cvmfs/singularity.opensciencegrid.org)
#   NOVA_SETUP   the NOvASoft setup command to run inside NOVA_IMAGE (from the experiment's own documentation)
#   NOVA_FCL     the TextFileGen simulation FCL (experiment configuration, not in this repository)
#   CARD         a BdNMC parameter card;  BDNMC_NOVA_CONFIG  the detector-frame configuration (private)
#   TRIALS, CHANNEL, NEVENTS, BIND (extra -B paths), GPU (1: pass --nv), PRECISION (float32 | float64)
#   BDNMC_NOVA_SRC  a bdnmc-nova checkout as the job sees it (default: the one holding this script)
# Outputs in $OUT (default $BDNMC_OUT, the hep-endpoints output directory, else ./out): events.txt (TextFileGen
# input), summary.json, and the simulation's art file/log.
set -eo pipefail
: "${GEN_IMAGE:?}" "${NOVA_IMAGE:?}" "${NOVA_SETUP:?}" "${NOVA_FCL:?}" "${CARD:?}" "${BDNMC_NOVA_CONFIG:?}"
OUT=${OUT:-${BDNMC_OUT:-$PWD/out}}
mkdir -p "$OUT"
HERE=${BDNMC_NOVA_SRC:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}
B="-B $OUT,$HERE,$(dirname "$CARD"),$(dirname "$BDNMC_NOVA_CONFIG")${BIND:+,$BIND}"

echo "==== $(date +%T) stage 1: BdNMC (JAX port) in $GEN_IMAGE"
apptainer exec ${GPU:+--nv} $B --env BDNMC_JAX_PRECISION="${PRECISION:-float32}" --env PYTHONPATH="$HERE" \
    "$GEN_IMAGE" python -m bdnmc_nova.generate "$CARD" --out "$OUT/events.txt" --config "$BDNMC_NOVA_CONFIG" \
    --trials "${TRIALS:-16777216}" ${CHANNEL:+--channel "$CHANNEL"} --summary "$OUT/summary.json"

echo "==== $(date +%T) stage 2: NOvASoft simulation in $NOVA_IMAGE"
apptainer exec -B /cvmfs $B "$NOVA_IMAGE" bash -lc \
    "$NOVA_SETUP && cd '$OUT' && nova -c '$NOVA_FCL' -n ${NEVENTS:--1} -o sim.root" > "$OUT/sim.log" 2>&1
echo "==== $(date +%T) done: $OUT"
