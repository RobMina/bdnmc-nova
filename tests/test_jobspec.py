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
    (bindir / "apptainer").write_text(f"#!/bin/sh\necho \"$*\" >> {log}\n")
    (bindir / "apptainer").chmod(0o755)
    seen = {}

    class Local:
        def __init__(self, environment):
            self.env = environment

        def submit(self, handle, script, resources):
            jd = tmp_path / handle
            env = {"PATH": f"{bindir}:/usr/bin:/bin", self.env.job_dir_var: str(jd), self.env.out_var: str(jd / "out")}
            subprocess.run(["bash", "-c", script], env=env, check=True, cwd=tmp_path)
            seen.update(resources=resources, jd=jd)
            return hependpoints.JobRecord(handle=handle, spec_sha=hependpoints.spec_sha(script, resources),
                                          job_id="local-1", remote_dir=str(jd), submitted_at=0.0)

    hependpoints.register("local-test", Local)
    rec = jobspec.submit(spec(gpus=1, minutes=5), "local-test")
    jd = seen["jd"]
    assert rec.job_id == "local-1" and seen["resources"] == hependpoints.Resources(1, 8, 1, "", 5)
    assert (jd / "in" / "card.dat").read_text() == "model Dark_Photon\nseed 3\n"
    assert json.loads((jd / "in" / "frame.json").read_text()) == json.loads(CONFIG)
    gen, sim = log.read_text().splitlines()
    assert gen.startswith("exec --nv -B ") and "/site/bdnmc-jax.sif python -m bdnmc_nova.generate" in gen
    assert f"--out {jd}/out/events.txt --config {jd}/in/frame.json" in gen and "PYTHONPATH=/site/bdnmc-nova" in gen
    assert sim.startswith("exec -B /cvmfs -B ") and "/site/sl7.sif bash -lc" in sim
    assert (jd / "out" / "sim.log").exists()
