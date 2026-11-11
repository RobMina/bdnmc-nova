import json
from pathlib import Path
import subprocess

import pytest

from bdnmc_nova import jobspec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "configs" / "example.json").read_text()


def spec(**kw):
    args = dict(handle="mb-pi0-m005", card_text="model Dark_Photon\nseed 3\n", config_text=CONFIG,
                src="/site/bdnmc-nova", gen_image="/site/bdnmc-jax.sif", nova_image="/site/sl7.sif",
                nova_setup="source /site/setup_nova.sh", nova_fcl="textfilegen_sim.fcl")
    return jobspec.JobSpec(**(args | kw))


def test_script_embeds_the_inputs(tmp_path):
    s = spec(gpus=1, gpu_type="a100", channel="pi0_decay", minutes=30).script()
    assert s.startswith("#!/bin/bash\n") and s.count("#!/") == 1
    assert "<<'BDNMC_CARD_EOF'\nmodel Dark_Photon\nseed 3\nBDNMC_CARD_EOF\n" in s
    assert "<<'BDNMC_CONFIG_EOF'\n" + CONFIG.rstrip("\n") + "\nBDNMC_CONFIG_EOF\n" in s
    assert "export GPU=1" in s and "export CHANNEL=pi0_decay" in s and "export BDNMC_NOVA_SRC=/site/bdnmc-nova" in s
    assert "export GPU" not in spec().script()  # two_stage.sh passes --nv for any nonempty GPU
    f = tmp_path / "job.sh"
    f.write_text(s)
    subprocess.run(["bash", "-n", str(f)], check=True)
    assert spec(gpus=1, minutes=30).resources() == {"cores": 1, "mem_gb": 8, "gpus": 1, "gpu_type": "",
                                                    "minutes": 30}


def test_spec_refuses_bad_inputs():
    with pytest.raises(ValueError):
        spec(handle="bad/handle")
    with pytest.raises(ValueError):
        spec(nova_fcl="")
    with pytest.raises(json.JSONDecodeError):
        spec(config_text="{not json")
    with pytest.raises(ValueError):
        spec(card_text="x\nBDNMC_CARD_EOF\n")


def test_slurm_job_file_has_the_nova_environment(tmp_path):
    from hependpoints import Resources
    s = spec(gpus=1, gpu_type="a100")
    ep = jobspec.endpoint("slurm_ssh", host="login.example", account="acct", workdir="/scratch/jobs",
                          state_dir=tmp_path / "state")
    body = ep.body(s.handle, s.script(), Resources(**s.resources()))
    assert "#SBATCH -J bdnmcnova_mb-pi0-m005" in body and "#SBATCH --gres=gpu:a100:1" in body
    assert "export BDNMC_JOB_DIR=/scratch/jobs/mb-pi0-m005 BDNMC_OUT=/scratch/jobs/mb-pi0-m005/out" in body
    f = tmp_path / "job.sh"
    f.write_text(body)
    subprocess.run(["bash", "-n", str(f)], check=True)


def test_submit_runs_both_stages_through_a_registered_endpoint(tmp_path):
    # a local endpoint registered with hependpoints runs the script in bash with $BDNMC_JOB_DIR and $BDNMC_OUT set;
    # a stub apptainer records both stages (no images, no NOvASoft)
    import hependpoints
    bindir, log = tmp_path / "bin", tmp_path / "apptainer.log"
    bindir.mkdir()
    # the generation stub writes a two-event TextFileGen file where --out points
    (bindir / "apptainer").write_text(f"#!/bin/sh\necho \"$*\" >> {log}\n" + STUB_OUT)
    (bindir / "apptainer").chmod(0o755)
    local_endpoint(tmp_path, bindir)
    rec = jobspec.submit(spec(gpus=1, minutes=5), "local-test")
    jd = SEEN["jd"]
    assert rec.job_id == "local-1" and SEEN["resources"] == hependpoints.Resources(1, 8, 1, "", 5)
    assert (jd / "in" / "card.dat").read_text() == "model Dark_Photon\nseed 3\n"
    assert json.loads((jd / "in" / "frame.json").read_text()) == json.loads(CONFIG)
    gen, sim = log.read_text().splitlines()
    assert gen.startswith("exec --nv -B ") and "/site/bdnmc-jax.sif python -m bdnmc_nova.generate" in gen
    assert f"--out {jd}/out/events.txt --config {jd}/in/frame.json" in gen and "PYTHONPATH=/site/bdnmc-nova" in gen
    assert sim.startswith("exec --cleanenv -B /cvmfs -B ") and "/site/sl7.sif bash -lc" in sim
    assert "nova -c 'textfilegen_sim.fcl' -n 2 -o sim.root" in sim
    assert f"python '{jd}/in/dump_primaries.py' sim.root --out primaries.csv" in sim
    assert (jd / "out" / "sim.log").exists()


EVENTS = "# provenance\n0 2\n1 11 0 -1 -1 -1 0 0 1 1 0.000511 1 2 3 4\n2 18 0 -1 -1 -1 0 0 1 1 0.005 1 2 3 4\n" \
         "1 2\n1 11 0 -1 -1 -1 0 0 2 2 0.000511 5 6 7 8\n2 18 0 -1 -1 -1 0 0 2 2 0.005 5 6 7 8\n"
STUB_OUT = 'while [ $# -gt 0 ]; do [ "$1" = --out ] && printf \'0 1\\nx\\n1 1\\nx\\n\' > "$2"; shift; done\n'
SEEN = {}


def local_endpoint(tmp_path, bindir):
    """Register "local-test": runs the script in bash with $BDNMC_JOB_DIR and $BDNMC_OUT set, `bindir` first on PATH."""
    import hependpoints

    class Local:
        def __init__(self, environment):
            self.env = environment

        def submit(self, handle, script, resources):
            jd = tmp_path / handle
            env = {"PATH": f"{bindir}:/usr/bin:/bin", self.env.job_dir_var: str(jd), self.env.out_var: str(jd / "out")}
            subprocess.run(["bash", "-c", self.env.preamble + script], env=env, check=True, cwd=tmp_path)
            SEEN.update(resources=resources, jd=jd)
            return hependpoints.JobRecord(handle=handle, spec_sha=hependpoints.spec_sha(script, resources),
                                          job_id="local-1", remote_dir=str(jd), submitted_at=0.0)

    hependpoints.register("local-test", Local)


def test_simulation_alone_embeds_events_fcl_and_dump(tmp_path):
    # stage sim as its own job (e.g. a grid worker already in NOvA's image): no generation inputs needed, NOvASoft in
    # the job's environment (no image), the events, FCL and primaries dump embedded; stub nova and python record calls
    bindir, log = tmp_path / "bin", tmp_path / "calls.log"
    bindir.mkdir()
    for tool in ("nova", "python", "apptainer"):
        (bindir / tool).write_text(f"#!/bin/sh\necho \"{tool} $*\" >> {log}\n")
        (bindir / tool).chmod(0o755)
    local_endpoint(tmp_path, bindir)
    s = jobspec.JobSpec("sim-only", "", "", "", "", "", "export SETUP_DONE=1", "", stages="sim", events_text=EVENTS,
                        nova_fcl_text='#include "prodtextgen.fcl"\n')
    assert "export GEN_IMAGE" not in s.script() and "BDNMC_CARD_EOF" not in s.script()
    jobspec.submit(s, "local-test", preamble="export SITE_READY=1")
    jd = SEEN["jd"]
    assert (jd / "out" / "events.txt").read_text() == EVENTS
    assert (jd / "in" / "sim.fcl").read_text() == '#include "prodtextgen.fcl"\n'
    nova, python = log.read_text().splitlines()
    assert nova == f"nova -c {jd}/in/sim.fcl -n 2 -o sim.root"
    assert python == f"python {jd}/in/dump_primaries.py sim.root --out primaries.csv"
    assert (jd / "in" / "dump_primaries.py").read_text().startswith('"""The generator\'s primaries')


def test_stage_requirements():
    gen_only = dict(nova_image="", nova_setup="", nova_fcl="", stages="gen")
    assert "export NOVA_SETUP" not in spec(**gen_only).script()
    with pytest.raises(ValueError):
        spec(**gen_only, gen_image="")
    with pytest.raises(ValueError):  # a simulation alone needs the events
        spec(stages="sim")
    with pytest.raises(ValueError):
        spec(stages="sim", events_text=EVENTS, nova_fcl="")
    with pytest.raises(ValueError):
        spec(stages="sim,gen")


def test_head_keeps_the_first_events():
    from bdnmc_nova.textfilegen import head
    assert head(EVENTS, -1) == EVENTS
    assert head(EVENTS, 1) == "".join(EVENTS.splitlines(keepends=True)[:4])
    assert head(EVENTS, 5) == EVENTS
    with pytest.raises(ValueError):
        head("1 11 0 -1 -1 -1 0 0 1 1 0 1 2 3 4\n", 1)


def test_preamble_reaches_the_slurm_job_file(tmp_path):
    from hependpoints import Resources
    s = spec()
    ep = jobspec.endpoint("slurm_ssh", preamble="module load apptainer", host="login.example", account="acct",
                          workdir="/scratch/jobs", state_dir=tmp_path / "state")
    body = ep.body(s.handle, s.script(), Resources(**s.resources()))
    assert body.index("module load apptainer\n") < body.index("# bdnmc-nova job mb-pi0-m005")


def test_cvmfsexec_mounts_around_the_simulation(tmp_path):
    # a site without /cvmfs: the job copies the cvmfsexec tree to node-local space, configures a private cache, runs
    # the simulation through it with apptainer --userns, and removes the copy afterwards
    bindir, log, local = tmp_path / "bin", tmp_path / "calls.log", tmp_path / "local"
    bindir.mkdir()
    local.mkdir()
    (bindir / "apptainer").write_text(f"#!/bin/sh\necho \"apptainer $*\" >> {log}\n")
    (bindir / "apptainer").chmod(0o755)
    ce = tmp_path / "cvmfsexec"
    (ce / "dist" / "etc" / "cvmfs").mkdir(parents=True)
    (ce / "cvmfsexec").write_text(f"#!/bin/sh\n"
                                  f"cp \"$(dirname \"$0\")/dist/etc/cvmfs/default.local\" {tmp_path}/default.local\n"
                                  f"echo \"cvmfsexec $*\" >> {log}\n"
                                  "while [ \"$1\" != -- ]; do shift; done; shift; exec \"$@\"\n")
    (ce / "cvmfsexec").chmod(0o755)
    local_endpoint(tmp_path, bindir)
    s = jobspec.JobSpec("cvmfs", "", "", "", "", "/cvmfs/images/sl7", "source setup", "", stages="sim",
                        events_text=EVENTS, nova_fcl_text="x\n", cvmfsexec=str(ce), cvmfs_repos="a.org b.org")
    with pytest.raises(ValueError):
        jobspec.JobSpec("cvmfs", "", "", "", "", "", "source setup", "", stages="sim", events_text=EVENTS,
                        nova_fcl_text="x\n", cvmfsexec=str(ce))
    jobspec.submit(s, "local-test", preamble=f"export CVMFS_LOCAL={local}")
    ce_call, app_call = log.read_text().splitlines()
    assert ce_call == "cvmfsexec a.org b.org -- apptainer exec --userns --cleanenv -B /cvmfs -B " + \
        f"{SEEN['jd'].resolve()}/out,{SEEN['jd'].resolve()}/in,{SEEN['jd'].resolve()}/in /cvmfs/images/sl7 bash -lc " + \
        f"source setup && cd '{SEEN['jd'].resolve()}/out' && nova -c '{SEEN['jd'].resolve()}/in/sim.fcl' -n 2 -o sim.root " + \
        f"&& python '{SEEN['jd'].resolve()}/in/dump_primaries.py' sim.root --out primaries.csv"
    assert app_call.startswith("apptainer exec --userns --cleanenv -B /cvmfs ")
    conf = (tmp_path / "default.local").read_text()
    assert 'CVMFS_HTTP_PROXY="DIRECT"' in conf and f"CVMFS_CACHE_BASE={local}/bdnmc-cvmfs." in conf
    assert list(local.iterdir()) == []  # the node-local copy and its cache are gone


def test_generation_on_gpu_then_simulation_on_cpu_after_it(tmp_path):
    # two Slurm jobs: the simulation (CPU partition) waits for the generation (GPU partition) with afterok and reads
    # its events.txt in place on the site's disk (nothing embedded, nothing moved)
    scripts = {}

    def runner(cmd, stdin, timeout):
        jid = str(100 + len(scripts))
        scripts[jid] = stdin.decode()
        return subprocess.CompletedProcess(cmd, 0, f"{jid}\n".encode(), b"")

    ep = jobspec.endpoint("slurm_ssh", preamble="module load apptainer", host="login.example", account="acct",
                          workdir="/scratch/jobs", cpu_partition="standard", state_dir=tmp_path / "state",
                          runner=runner)
    gen = jobspec.submit(spec(handle="gen", stages="gen", gpus=1, gpu_type="v100"), ep)
    sim = spec(handle="sim", stages="sim", events="/scratch/jobs/gen/out/events.txt", cores=1, mem_gb=16,
               nova_image="/cvmfs/images/sl7", cvmfsexec="/site/cvmfsexec")
    rec = jobspec.submit(sim, ep, after=["gen"])
    body = scripts[rec.job_id]
    assert f"#SBATCH --dependency=afterok:{gen.job_id}\n" in body and "#SBATCH -p standard" in body
    assert "--gres" not in body and "#SBATCH --gres=gpu:v100:1" in scripts[gen.job_id]
    assert "export EVENTS=/scratch/jobs/gen/out/events.txt" in body and "BDNMC_EVENTS_EOF" not in body
    assert "export CVMFSEXEC=/site/cvmfsexec" in body and "export STAGES=sim" in body
    with pytest.raises(ValueError):  # embedded and at-the-site events together
        spec(stages="sim", events="/x/events.txt", events_text=EVENTS)
