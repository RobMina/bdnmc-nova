"""A bdnmc-nova two-stage job as a self-contained script plus resources, submitted through the experiment-neutral
`hep-endpoints` package (RobMina/hep-endpoints, import name `hependpoints`).

hep-endpoints moves a job script to a site, runs it and brings its output directory back (Slurm over SSH, Fermilab
EAF, FNAL grid through jobsub); the site settings are its `HEPEP_*` variables or keyword arguments. This module:
- `JobSpec.script()`: data/two_stage.sh with its inputs set and the card and the frame configuration embedded, so
  the job needs only the two images, a bdnmc-nova checkout and the simulation FCL at the site; it writes to the
  endpoint's output directory (`$BDNMC_OUT`) and keeps its inputs in the job directory (`$BDNMC_JOB_DIR`). The
  stages can run as separate jobs at separate sites (`stages`): generation where the GPU is, the simulation where
  NOvASoft is, with the first job's TextFileGen file (`events_text`), the simulation FCL (`nova_fcl_text`) and the
  primaries dump (data/dump_primaries.py) embedded, so a grid worker needs nothing but CVMFS;
- `JobSpec.resources()`: cores, memory, GPUs, wall time (the fields of hependpoints.Resources);
- `endpoint(name, preamble="", **kw)`, `submit(spec, endpoint)`: an endpoint with bdnmc-nova's job environment
  (variables `BDNMC_*`, job names `bdnmcnova_<handle>`, the site's shell `preamble`, e.g. "module load apptainer"),
  and the submission. Status, fetch and cancel are the endpoint's own.

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
DELIMS = ("BDNMC_CARD_EOF", "BDNMC_CONFIG_EOF", "BDNMC_EVENTS_EOF", "BDNMC_FCL_EOF", "BDNMC_DUMP_EOF")
STAGES = ("gen,sim", "gen", "sim")


@dataclass
class JobSpec:
    handle: str  # the caller's name for the job (hependpoints: idempotent per handle)
    card_text: str  # the BdNMC card, embedded in the script (stage gen)
    config_text: str  # the detector-frame configuration (bdnmc_nova.config), embedded in the script (stage gen)
    src: str  # a bdnmc-nova checkout at the site (bound into the generation image; stage gen)
    gen_image: str  # the BdNMC JAX image at the site (stage gen)
    nova_image: str  # NOvA's SL7 image at the site; "": NOvASoft runs in the job's own environment (stage sim)
    nova_setup: str  # the NOvASoft setup command (stage sim)
    nova_fcl: str  # the TextFileGen simulation FCL at the site (stage sim; or nova_fcl_text)
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
    stages: str = "gen,sim"  # "gen,sim" (one job), "gen" or "sim" (two jobs, possibly at two sites)
    events_text: str = ""  # stage sim alone: the TextFileGen input (a stage gen job's events.txt), embedded
    events: str = ""  # stage sim alone: or that file's path at the site (e.g. a stage gen job's output, same disk)
    nova_fcl_text: str = ""  # the simulation FCL, embedded (in place of nova_fcl, e.g. for a grid worker)
    dump: bool = True  # stage sim: dump the art file's primaries to primaries.csv (data/dump_primaries.py)
    cvmfsexec: str = ""  # stage sim on a site without /cvmfs: a prepared cvmfsexec tree at the site
    cvmfs_repos: str = ""  # the repositories cvmfsexec mounts (space-separated; default: data/two_stage.sh's)

    @property
    def gen(self):
        return "gen" in self.stages.split(",")

    @property
    def sim(self):
        return "sim" in self.stages.split(",")

    def __post_init__(self):
        if not HANDLE.match(self.handle):
            raise ValueError(f"handle {self.handle!r}: letters, digits, '_', '.', '-'")
        if self.precision not in ("float32", "float64"):
            raise ValueError("precision: float32 or float64")
        if self.stages not in STAGES:
            raise ValueError(f"stages: one of {STAGES}")
        required = ("src", "gen_image") if self.gen else ()
        required += ("nova_setup",) if self.sim else ()
        for name in required:
            if not getattr(self, name):
                raise ValueError(f"{name} is required (a path or command at the site)")
        if self.sim and not (self.nova_fcl or self.nova_fcl_text):
            raise ValueError("nova_fcl (at the site) or nova_fcl_text is required")
        if self.sim and self.cvmfsexec and not self.nova_image:
            raise ValueError("cvmfsexec runs the simulation in nova_image: set it (it may be on CVMFS)")
        if self.gen:
            json.loads(self.config_text)  # a configuration that does not parse fails here, not in the job
        elif bool(self.events_text.strip()) == bool(self.events):
            raise ValueError("a simulation-only job needs events_text or events (a generation job's events.txt)")
        if any(d in t for d in DELIMS for t in (self.card_text, self.config_text, self.events_text,
                                                self.nova_fcl_text)):
            raise ValueError("card, configuration, events or FCL contains a heredoc delimiter")

    def env(self):
        """The job script's inputs (data/two_stage.sh), apart from the embedded files."""
        e = {"STAGES": self.stages}
        if self.gen:
            e |= {"BDNMC_NOVA_SRC": self.src, "GEN_IMAGE": self.gen_image, "TRIALS": str(int(self.trials)),
                  "PRECISION": self.precision}
            e |= {"CHANNEL": self.channel} if self.channel else {}
        if self.sim:
            e |= {"NOVA_SETUP": self.nova_setup, "NEVENTS": str(int(self.nevents))}
            e |= {"NOVA_IMAGE": self.nova_image} if self.nova_image else {}
            e |= {"NOVA_FCL": self.nova_fcl} if self.nova_fcl and not self.nova_fcl_text else {}
            e |= {"CVMFSEXEC": self.cvmfsexec} if self.cvmfsexec else {}
            e |= {"EVENTS": self.events} if self.events and not self.gen else {}
            e |= {"CVMFS_REPOS": self.cvmfs_repos} if self.cvmfs_repos else {}
        e |= {"BIND": self.bind} if self.bind else {}
        return e | ({"GPU": "1"} if self.gpus and self.gen else {})  # two_stage.sh passes --nv for any nonempty GPU

    def script(self):
        data = ilr.files("bdnmc_nova").joinpath("data")
        body = data.joinpath("two_stage.sh").read_text()
        body = body.split("\n", 1)[1]  # drop the shebang; the prelude has its own
        prelude = ["#!/bin/bash", f"# bdnmc-nova job {self.handle}", "set -eo pipefail",
                   f'IN="${{{VAR_PREFIX}_JOB_DIR:-$PWD}}/in"', 'mkdir -p "$IN"']
        files = []  # (variable, path in $IN, text, delimiter)
        if self.gen:
            files += [("CARD", "card.dat", self.card_text, "BDNMC_CARD_EOF"),
                      ("BDNMC_NOVA_CONFIG", "frame.json", self.config_text, "BDNMC_CONFIG_EOF")]
        if self.sim and not self.gen and self.events_text:
            files.append(("EVENTS", "events.txt", self.events_text, "BDNMC_EVENTS_EOF"))
        if self.sim and self.nova_fcl_text:
            files.append(("NOVA_FCL", "sim.fcl", self.nova_fcl_text, "BDNMC_FCL_EOF"))
        if self.sim and self.dump:
            files.append(("DUMP_PRIMARIES", "dump_primaries.py", data.joinpath("dump_primaries.py").read_text(),
                          "BDNMC_DUMP_EOF"))
        if files:
            prelude.append("export " + " ".join(f'{var}="$IN/{name}"' for var, name, _, _ in files))
        prelude += [f"export {k}={shlex.quote(v)}" for k, v in self.env().items()]
        for var, _, text, delim in files:
            prelude += [f"cat > \"${var}\" <<'{delim}'", text.rstrip("\n"), delim]
        return "\n".join(prelude) + "\n" + body

    def resources(self):
        """Keyword arguments for hependpoints.Resources."""
        return {"cores": self.cores, "mem_gb": self.mem_gb, "gpus": self.gpus, "gpu_type": self.gpu_type,
                "minutes": self.minutes}


def environment(preamble=""):
    """bdnmc-nova's hependpoints.JobEnvironment: jobs see $BDNMC_JOB_DIR and $BDNMC_OUT; job names bdnmcnova_<handle>.
    `preamble`: the site's shell setup before the job script (e.g. "module load apptainer")."""
    from hependpoints.environment import JobEnvironment
    return JobEnvironment(var_prefix=VAR_PREFIX, job_prefix=JOB_PREFIX, preamble=preamble)


def endpoint(name, preamble="", **kw):
    """A hependpoints endpoint ("slurm_ssh", "eaf", "jobsub" or one an experiment registered) with bdnmc-nova's job
    environment and the site's `preamble`; its site settings come from `kw` or its HEPEP_* variables."""
    import hependpoints
    kw.setdefault("environment", environment(preamble))
    return hependpoints.get(name, **kw)


def submit(spec, ep, after=(), **endpoint_kw):
    """Submit `spec` to `ep` (an endpoint, or a name for `endpoint`). Idempotent per handle: resubmitting the same
    spec returns the same record; a changed spec under the same handle is refused. `after`: handles of earlier jobs
    on the same endpoint that must complete successfully first (an endpoint with dependencies, e.g. slurm_ssh), so
    a simulation job on a CPU partition can follow the generation job on a GPU partition. The endpoint's fetch
    brings $BDNMC_OUT back (events.txt, summary.json, the simulation's output and log)."""
    import hependpoints
    if isinstance(ep, str):
        ep = endpoint(ep, **endpoint_kw)
    res = hependpoints.Resources(**spec.resources())
    if after:
        return ep.submit(spec.handle, spec.script(), res, after=list(after))
    return ep.submit(spec.handle, spec.script(), res)


def _setting(text):
    """KEY=VALUE for an endpoint keyword argument; VALUE as JSON when it parses (numbers, lists), else a string."""
    key, sep, value = text.partition("=")
    if not sep or not key.isidentifier():
        raise argparse.ArgumentTypeError(f"{text!r}: KEY=VALUE")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bdnmc_nova.jobspec", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def site(p):
        p.add_argument("--preamble", default="", help="shell text the site needs before the job, e.g. 'module load "
                       "apptainer' ('\\n' separates lines)")
        p.add_argument("--endpoint-setting", type=_setting, action="append", default=[], metavar="KEY=VALUE",
                       help="an endpoint keyword argument, e.g. worker_setup='source ...' for jobsub (repeatable)")

    for cmd in ("submit", "script"):
        p = sub.add_parser(cmd, help="submit the job" if cmd == "submit" else "write the job script")
        p.add_argument("card", type=Path, nargs="?", help="BdNMC card (stage gen)")
        p.add_argument("--handle", required=True)
        p.add_argument("--stages", default="gen,sim", choices=STAGES)
        p.add_argument("--config", type=Path, help="frame configuration (default $BDNMC_NOVA_CONFIG; stage gen)")
        for name in ("src", "gen-image", "nova-image", "nova-setup", "nova-fcl"):
            p.add_argument(f"--{name}", default="", help="at the site")
        p.add_argument("--nova-fcl-file", type=Path, help="a local simulation FCL to embed (in place of --nova-fcl)")
        p.add_argument("--events", type=Path, help="stage sim alone: a TextFileGen file to embed (events.txt)")
        p.add_argument("--events-at", default="", help="stage sim alone: the TextFileGen file's path at the site")
        p.add_argument("--max-events", type=int, default=-1, help="embed only the first N events of --events")
        p.add_argument("--no-dump", action="store_true", help="do not dump the art file's primaries")
        p.add_argument("--cvmfsexec", default="", help="a prepared cvmfsexec tree at the site (no /cvmfs there)")
        p.add_argument("--cvmfs-repos", default="", help="repositories for cvmfsexec to mount (space-separated)")
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
            p.add_argument("--after", action="append", default=[], metavar="HANDLE",
                           help="start after this earlier job (same endpoint) completes successfully (repeatable)")
            site(p)
        else:
            p.add_argument("--out", type=Path, required=True)
    for cmd in ("status", "fetch", "cancel"):
        p = sub.add_parser(cmd)
        p.add_argument("handle")
        p.add_argument("--endpoint", required=True)
        site(p)
        if cmd == "fetch":
            p.add_argument("--dest", type=Path, default=Path("."), help="copies the output under DEST/<handle>/")
    a = ap.parse_args(argv)
    if a.cmd in ("submit", "script"):
        gen = "gen" in a.stages.split(",")
        card_text = config_text = events_text = ""
        if gen:
            config = a.config or os.environ.get("BDNMC_NOVA_CONFIG")
            if not a.card or not config:
                ap.error("stage gen: pass a card and --config (or set BDNMC_NOVA_CONFIG)")
            card_text, config_text = a.card.read_text(), Path(config).read_text()
        elif bool(a.events) == bool(a.events_at):
            ap.error("--stages sim: pass --events (a local file, embedded) or --events-at (a path at the site)")
        if a.events:
            from .textfilegen import head
            events_text = head(a.events.read_text(), a.max_events)
        spec = JobSpec(a.handle, card_text, config_text, a.src, a.gen_image, a.nova_image, a.nova_setup, a.nova_fcl,
                       trials=a.trials, channel=a.channel, nevents=a.nevents, precision=a.precision, bind=a.bind,
                       gpus=a.gpus, gpu_type=a.gpu_type, cores=a.cores, mem_gb=a.mem_gb, minutes=a.minutes,
                       stages=a.stages, events_text=events_text, events=a.events_at,
                       nova_fcl_text=a.nova_fcl_file.read_text() if a.nova_fcl_file else "", dump=not a.no_dump,
                       cvmfsexec=a.cvmfsexec, cvmfs_repos=a.cvmfs_repos)
        if a.cmd == "script":
            a.out.write_text(spec.script())
            a.out.chmod(0o700)  # it holds the private frame configuration
            print(f"wrote {a.out}")
        else:
            rec = submit(spec, a.endpoint, after=a.after, preamble=a.preamble.replace("\\n", "\n"), **dict(a.endpoint_setting))
            print(f"submitted {spec.handle} to {a.endpoint}: job {rec.job_id}")
        return
    ep = endpoint(a.endpoint, preamble=a.preamble.replace("\\n", "\n"), **dict(a.endpoint_setting))
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
