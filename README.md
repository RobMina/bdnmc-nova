# bdnmc-nova

Light-dark-matter event generation with [BdNMC](https://github.com/RobMina/BdNMC) for the NOvA Near Detector:
the generic parts of the integration. BdNMC itself (the stock C++ generator and its validated JAX GPU port) stays
experiment-independent in its own repository; this package adds what NOvA needs around it.

**In this repository** (public):
- `bdnmc_nova.frame`: BdNMC's beam frame to the detector frame (detector = R^T (x - X0), cm), with per-channel
  production offsets;
- `bdnmc_nova.textfilegen`: BdNMC events as NOvASoft TextFileGen input (HEPEVT-style; electron tracked, recoil DM
  PDG 18 not tracked), through the fork's `bdnmc_jax.output.write_hepevt`;
- `bdnmc_nova.generate`: stage 1 of a job: card (+ overrides) -> JAX port -> TextFileGen file (+ HDF5, summary);
- `containers/`: how to combine BdNMC with NOvASoft in a job or a container; `build_libbdnmc_sl7.sh` builds stock
  libBdNMC with NOvASoft's compiler for the in-process mode;
- `jobs/two_stage.sh` (packaged as `bdnmc_nova/data/two_stage.sh`): generation and simulation in one job;
- `bdnmc_nova.jobspec`: that job as a self-contained script (card and frame configuration embedded), submitted to a
  compute endpoint through the experiment-neutral [hep-endpoints](https://github.com/RobMina/hep-endpoints) package
  (Slurm over SSH, Fermilab EAF, the FNAL grid through jobsub).

**Not in this repository** (NOvA collaboration material, kept in the experiment's private configuration): the
detector frame's values (the ND's position and orientation in the beam frame), the NuMI meson lists from g4numi, the
surveyed ND geometry and material, the NOvASoft generator module that calls BdNMC in process, and the simulation
FCLs. `configs/example.json` is an identity frame for tests, not NOvA's.

## Two ways to run

NOvASoft runs in an SL7 image. The JAX port needs glibc >= 2.27, which SL7 (2.17) does not have, so:

| Mode | Generation | Simulation | Endpoints |
|---|---|---|---|
| **two-stage job** (`jobs/two_stage.sh`) | JAX port in the BdNMC image (`RobMina/BdNMC` `gpu/containers/bdnmc-jax.def`), GPU or CPU | NOvASoft in its SL7 image, reading the TextFileGen file | any with Apptainer and CVMFS: the FNAL grid, a Slurm cluster, EAF |
| **single container, in process** | stock libBdNMC (branch `shared_library`, C API `BdNMC_capi.h`) built with NOvASoft's compiler (`containers/build_libbdnmc_sl7.sh`), called by a NOvASoft generator module | the same art job | CPU endpoints (FNAL grid) |

The two-stage mode uses the GPU port's speed; the in-process mode needs no intermediate file and no second image.
On a cluster with GPUs but no `/cvmfs` mount, the job mounts CVMFS itself with
[cvmfsexec](https://github.com/cvmfs/cvmfsexec) (unprivileged FUSE and user namespaces on the compute node) and runs
NOvA's SL7 image from CVMFS with `apptainer --userns`. The stages then run as two jobs on the same cluster: generation
on a GPU partition, and the simulation, which does not use the GPU, on a CPU partition with a Slurm dependency on it
(`--after`), reading the generated file in place on the cluster's disk. They can also run as one job
(`--stages gen,sim`, the default) or at two sites (the simulation on a grid worker already in NOvA's image, with the
events embedded; no `--nova-image`: NOvASoft runs in the job's own environment).

## Use

```bash
pip install -e .            # pulls bdnmc-jax from RobMina/BdNMC (branch gpu-port)
```

```bash
BDNMC_NOVA_CONFIG=/path/to/private/nova_frame.json python -m bdnmc_nova.generate card.dat --out events.txt --summary summary.json
```

```bash
GEN_IMAGE=bdnmc-jax.sif NOVA_IMAGE=<sl7 image> NOVA_SETUP='<NOvASoft setup>' NOVA_FCL=<textfilegen sim fcl> CARD=card.dat BDNMC_NOVA_CONFIG=/path/to/private/nova_frame.json GPU=1 bash jobs/two_stage.sh
```

Through hep-endpoints (site settings in its `HEPEP_*` variables; every path is as the job sees it at the site):

```bash
python -m bdnmc_nova.jobspec submit card.dat --handle <name> --endpoint slurm_ssh --config /path/to/private/nova_frame.json --src <bdnmc-nova checkout> --gen-image <bdnmc-jax.sif> --nova-image <sl7 image> --nova-setup '<NOvASoft setup>' --nova-fcl <textfilegen sim fcl> --gpus 1
```

```bash
python -m bdnmc_nova.jobspec status <name> --endpoint slurm_ssh
```

```bash
python -m bdnmc_nova.jobspec fetch <name> --endpoint slurm_ssh --dest results
```

The job keeps its inputs in the endpoint's job directory and writes to its output directory (`$BDNMC_OUT`), which
`fetch` copies to `results/<name>/`. Resubmitting the same job under the same handle returns the same record; a
changed one is refused. `jobspec script ... --out job.sh` writes the job script instead (mode 0700: it holds the
private frame configuration).

Generation on a GPU node, then the simulation on a CPU node once it has succeeded, on a Slurm cluster without
`/cvmfs` (the image, the release and the setup come from CVMFS through cvmfsexec; the simulation FCL and the
primaries dump are embedded; `HEPEP_SLURM_CPU_PARTITION` names the CPU partition):

```bash
python -m bdnmc_nova.jobspec submit card.dat --stages gen --handle <gen> --endpoint slurm_ssh --config /path/to/private/nova_frame.json --src <bdnmc-nova checkout> --gen-image <bdnmc-jax.sif> --gpus 1 --cores 4 --mem-gb 32 --minutes 30 --preamble 'module load apptainer'
```

```bash
python -m bdnmc_nova.jobspec submit --stages sim --handle <sim> --endpoint slurm_ssh --after <gen> --events-at <job dir of gen>/out/events.txt --nevents 1000 --nova-image <SL7 image directory on CVMFS> --nova-setup '<NOvASoft setup>' --nova-fcl-file /path/to/private/sim.fcl --cvmfsexec <cvmfsexec tree at the site> --cores 1 --mem-gb 16 --minutes 150 --preamble 'module load apptainer'
```

The simulation alone, as its own job (here with the events embedded, for a site that cannot see the first job's
disk):

```bash
python -m bdnmc_nova.jobspec submit --stages sim --handle <sim> --endpoint jobsub --events results/<gen>/out/events.txt --max-events 1000 --nova-fcl-file /path/to/private/sim.fcl --nova-setup 'command -v nova >/dev/null || source <NOvASoft setup>' --mem-gb 6 --minutes 240 --endpoint-setting 'worker_setup=source <NOvASoft setup>'
```

- `--after <handle>`: Slurm `afterok` on an earlier job of the same endpoint (hep-endpoints `slurm_ssh`); if the
  generation fails, Slurm cancels the simulation.
- `--cvmfsexec`: a cvmfsexec tree prepared for the compute node's OS (`makedist`); the job copies it to node-local
  `/tmp`, mounts the repositories (`--cvmfs-repos`; default config-osg, fermilab, nova, nova-development,
  singularity) with a private cache from the FNAL and BNL Stratum-1s, and deletes the copy and the cache at the end.
- `--preamble`: the site's shell setup before the job (e.g. loading Apptainer as a module); `--endpoint-setting
  KEY=VALUE`: any keyword argument of the hep-endpoints endpoint (for `jobsub`: `worker_setup`, the setup that makes
  `ifdh` available on the worker before the job's first copy; `copy_command` when the submit host reads dCache
  differently from the worker). Pass the same settings to `status`, `fetch` and `cancel`.
- The FCL's TextFileGen module reads `events.txt` from the working directory (the job's output directory). Every
  event of the file is simulated (`--nevents` to cap it).
- Off the experiment's network the experiment's conditions web service is not reachable: the simulation FCL then
  sets the values a service would look up at the start of a run (for this chain, the detector's gain setting).
- A NOvASoft setup that refuses to run twice in one shell is guarded with `command -v nova || ...` when
  `worker_setup` has already run it.
- The simulation stage dumps the art file's generator primaries to `primaries.csv`
  (`bdnmc_nova/data/dump_primaries.py`, in the NOvASoft environment; `--no-dump` to skip), and
  `jobs/check_textfilegen.py` compares them with the TextFileGen input, outside that environment:

```bash
python jobs/check_textfilegen.py results/<sim>/out/events.txt results/<sim>/out/primaries.csv --log results/<sim>/out/sim.log
```

The configuration file is described in `bdnmc_nova/config.py`. Precision: float32 by default (the port's default),
`PRECISION=float64` in the job or `BDNMC_JAX_PRECISION=float64` for float64.

## Status

- **First version (2026-10-31):** frame, TextFileGen writer, generation CLI and tests (laptop CPU).
- **Run end to end (2026-11-11)** on one Slurm cluster through hep-endpoints `slurm_ssh`, as two jobs chained with
  `--after` (Slurm `afterok`); the TextFileGen file stayed on the cluster's disk.
  - Generation: one NVIDIA V100 with 4 CPU cores, the BdNMC JAX image (fork `0dfc3e3`), float32. The generic 120 GeV
    BMPT pi0 card at m_chi = 5 MeV, with the detector placement and frame from the private configuration: 2^24
    trials in 11.9 s, 39191 events (39246 DM scatters).
  - Simulation: one CPU core, 16 GB, on a node without `/cvmfs`. cvmfsexec mounted the repositories; the Fermilab SL7
    worker image and the NOvASoft development release came from CVMFS (`apptainer --userns`), with a 1.3 GB node-local
    cache. It started 80 s after generation ended and simulated the first 1000 events (TextFileGen, Geant4, photon
    transport, readout) in 20 min 10 s. Geant4 took 1.08 s per event on average (median 0.66 s). The art file is
    55 MB.
  - Checked with `jobs/check_textfilegen.py`: 1000 art events with 2002 primaries (electrons and recoil DM), all
    equal to the TextFileGen input. Vertex, time, momentum, energy, mass, status, PDG and mother are identical, with
    0 relative difference. TrigReport 1000/1000.
  - Found: Off the experiment's network, the ideal-MC run number does not stop the simulation from asking the
    conditions web service for the detector gain, so the FCL sets the gain.
- **Tested offline only:** the `jobsub` endpoint path (simulation alone, events embedded) and the single-job mode
  (`--stages gen,sim`, both stages on the GPU node). A simulation-only grid job submitted through `jobsub`
  received its job id and was cancelled before it ran.
- **Next:** a comparison with the existing NOvA chain's TextFileGen files for the same card; g4numi meson lists
  instead of BMPT; the in-process mode.

## Tests

```bash
python -m pytest
```

CI (`.github/workflows/ci.yml`) runs them on every push and pull request and weekly, against the head of the
port's `gpu-port` branch.
