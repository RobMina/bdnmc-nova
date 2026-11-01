"""A bdnmc-nova two-stage job as a self-contained script plus resources, submitted through the experiment-neutral
`hep-endpoints` package (RobMina/hep-endpoints, import name `hependpoints`).

hep-endpoints moves a job script to a site, runs it and brings its output directory back (Slurm over SSH, Fermilab
EAF, FNAL grid through jobsub); the site settings are its `HEPEP_*` variables or keyword arguments. This module:
- `JobSpec.script()`: data/two_stage.sh with its inputs set and the card and the frame configuration embedded, so
  the job needs only the two images, a bdnmc-nova checkout and the simulation FCL at the site; it writes to the
  endpoint's output directory (`$BDNMC_OUT`) and keeps its inputs in the job directory (`$BDNMC_JOB_DIR`);
- `JobSpec.resources()`: cores, memory, GPUs, wall time (the fields of hependpoints.Resources);
- `endpoint(name, **kw)`, `submit(spec, endpoint)`: an endpoint with bdnmc-nova's job environment (variables
  `BDNMC_*`, job names `bdnmcnova_<handle>`), and the submission. Status, fetch and cancel are the endpoint's own.

The frame configuration holds the experiment's private values: it travels inside the job script to the user's own
job directory and is never written to this repository.

    python -m bdnmc_nova.jobspec submit card.dat --handle <name> --endpoint slurm_ssh --config <private frame json> ...
    python -m bdnmc_nova.jobspec status <name> --endpoint slurm_ssh
"""
import argparse
from dataclasses import dataclass
from importlib import resources as ilr
import json
import os
from pathlib import Path
import re
import shlex

HANDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
VAR_PREFIX, JOB_PREFIX = "BDNMC", "bdnmcnova"
DELIMS = ("BDNMC_CARD_EOF", "BDNMC_CONFIG_EOF")


@dataclass
class JobSpec:
    handle: str  # the caller's name for the job (hependpoints: idempotent per handle)
    card_text: str  # the BdNMC card, embedded in the script
    config_text: str  # the detector-frame configuration (bdnmc_nova.config), embedded in the script
    src: str  # a bdnmc-nova checkout at the site (bound into the generation image)
    gen_image: str  # the BdNMC JAX image at the site
    nova_image: str  # NOvA's SL7 image at the site
    nova_setup: str  # the NOvASoft setup command inside nova_image
    nova_fcl: str  # the TextFileGen simulation FCL at the site
    trials: int = 1 << 24
    channel: str = ""
    nevents: int = -1
    precision: str = "float32"
    bind: str = ""  # extra apptainer -B paths
    gpus: int = 0
    gpu_type: str = ""
    cores: int = 1
    mem_gb: int = 8
    minutes: int = 120

    def __post_init__(self):
        if not HANDLE.match(self.handle):
            raise ValueError(f"handle {self.handle!r}: letters, digits, '_', '.', '-'")
        if self.precision not in ("float32", "float64"):
            raise ValueError("precision: float32 or float64")
        for name in ("src", "gen_image", "nova_image", "nova_setup", "nova_fcl"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required (a path or command at the site)")
        json.loads(self.config_text)  # a configuration that does not parse fails here, not in the job
        if any(d in self.card_text or d in self.config_text for d in DELIMS):
            raise ValueError("card or configuration contains a heredoc delimiter")

    def env(self):
        """The job script's inputs (data/two_stage.sh), apart from the embedded files."""
        e = {"BDNMC_NOVA_SRC": self.src, "GEN_IMAGE": self.gen_image, "NOVA_IMAGE": self.nova_image,
             "NOVA_SETUP": self.nova_setup, "NOVA_FCL": self.nova_fcl, "TRIALS": str(int(self.trials)),
             "NEVENTS": str(int(self.nevents)), "PRECISION": self.precision}
        e |= {"CHANNEL": self.channel} if self.channel else {}
        e |= {"BIND": self.bind} if self.bind else {}
        return e | ({"GPU": "1"} if self.gpus else {})  # two_stage.sh passes --nv for any nonempty GPU

    def script(self):
        body = ilr.files("bdnmc_nova").joinpath("data", "two_stage.sh").read_text()
        body = body.split("\n", 1)[1]  # drop the shebang; the prelude has its own
        prelude = ["#!/bin/bash", f"# bdnmc-nova job {self.handle}", "set -eo pipefail",
                   f'IN="${{{VAR_PREFIX}_JOB_DIR:-$PWD}}/in"', 'mkdir -p "$IN"',
                   'export CARD="$IN/card.dat" BDNMC_NOVA_CONFIG="$IN/frame.json"']
        prelude += [f"export {k}={shlex.quote(v)}" for k, v in self.env().items()]
        prelude += ["cat > \"$CARD\" <<'BDNMC_CARD_EOF'", self.card_text.rstrip("\n"), "BDNMC_CARD_EOF",
                    "cat > \"$BDNMC_NOVA_CONFIG\" <<'BDNMC_CONFIG_EOF'", self.config_text.rstrip("\n"),
                    "BDNMC_CONFIG_EOF"]
        return "\n".join(prelude) + "\n" + body

    def resources(self):
        """Keyword arguments for hependpoints.Resources."""
        return {"cores": self.cores, "mem_gb": self.mem_gb, "gpus": self.gpus, "gpu_type": self.gpu_type,
                "minutes": self.minutes}


def environment():
    """bdnmc-nova's hependpoints.JobEnvironment: jobs see $BDNMC_JOB_DIR and $BDNMC_OUT; job names bdnmcnova_<handle>."""
    from hependpoints.environment import JobEnvironment
    return JobEnvironment(var_prefix=VAR_PREFIX, job_prefix=JOB_PREFIX)


def endpoint(name, **kw):
    """A hependpoints endpoint ("slurm_ssh", "eaf", "jobsub" or one an experiment registered) with bdnmc-nova's job
    environment; its site settings come from `kw` or its HEPEP_* variables."""
    import hependpoints
    kw.setdefault("environment", environment())
    return hependpoints.get(name, **kw)


def submit(spec, ep, **endpoint_kw):
    """Submit `spec` to `ep` (an endpoint, or a name for `endpoint`). Idempotent per handle: resubmitting the same
    spec returns the same record; a changed spec under the same handle is refused. The endpoint's fetch brings
    $BDNMC_OUT back (events.txt, summary.json, the simulation's output and log)."""
    import hependpoints
    if isinstance(ep, str):
        ep = endpoint(ep, **endpoint_kw)
    return ep.submit(spec.handle, spec.script(), hependpoints.Resources(**spec.resources()))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bdnmc_nova.jobspec", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("submit", "script"):
        p = sub.add_parser(cmd, help="submit the job" if cmd == "submit" else "write the job script")
        p.add_argument("card", type=Path)
        p.add_argument("--handle", required=True)
        p.add_argument("--config", type=Path, help="frame configuration (default $BDNMC_NOVA_CONFIG)")
        for name in ("src", "gen-image", "nova-image", "nova-setup", "nova-fcl"):
            p.add_argument(f"--{name}", required=True, help="at the site")
        p.add_argument("--trials", type=int, default=1 << 24)
        p.add_argument("--channel", default="")
        p.add_argument("--nevents", type=int, default=-1)
        p.add_argument("--precision", default="float32", choices=("float32", "float64"))
        p.add_argument("--bind", default="")
        p.add_argument("--gpus", type=int, default=0)
        p.add_argument("--gpu-type", default="")
        p.add_argument("--cores", type=int, default=1)
        p.add_argument("--mem-gb", type=int, default=8)
        p.add_argument("--minutes", type=int, default=120)
        if cmd == "submit":
            p.add_argument("--endpoint", required=True, help="slurm_ssh, eaf, jobsub (settings: HEPEP_* variables)")
        else:
            p.add_argument("--out", type=Path, required=True)
    for cmd in ("status", "fetch", "cancel"):
        p = sub.add_parser(cmd)
        p.add_argument("handle")
        p.add_argument("--endpoint", required=True)
        if cmd == "fetch":
            p.add_argument("--dest", type=Path, default=Path("."), help="copies the output under DEST/<handle>/")
    a = ap.parse_args(argv)
    if a.cmd in ("submit", "script"):
        config = a.config or os.environ.get("BDNMC_NOVA_CONFIG")
        if not config:
            ap.error("no configuration: pass --config or set BDNMC_NOVA_CONFIG")
        spec = JobSpec(a.handle, a.card.read_text(), Path(config).read_text(), a.src, a.gen_image, a.nova_image,
                       a.nova_setup, a.nova_fcl, trials=a.trials, channel=a.channel, nevents=a.nevents,
                       precision=a.precision, bind=a.bind, gpus=a.gpus, gpu_type=a.gpu_type, cores=a.cores,
                       mem_gb=a.mem_gb, minutes=a.minutes)
        if a.cmd == "script":
            a.out.write_text(spec.script())
            a.out.chmod(0o700)  # it holds the private frame configuration
            print(f"wrote {a.out}")
        else:
            rec = submit(spec, a.endpoint)
            print(f"submitted {spec.handle} to {a.endpoint}: job {rec.job_id}")
        return
    ep = endpoint(a.endpoint)
    if a.cmd == "status":
        st = ep.status(a.handle)
        print(f"{a.handle}: {st.state} {st.message}".rstrip())
    elif a.cmd == "fetch":
        for f in ep.fetch(a.handle, a.dest):
            print(f)
    else:
        print(ep.cancel(a.handle))


if __name__ == "__main__":
    main()
