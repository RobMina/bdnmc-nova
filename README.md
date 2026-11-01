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
- `jobs/two_stage.sh`: generation and simulation in one job.

**Not in this repository** (NOvA collaboration material, kept in the experiment's private configuration): the
detector frame's values (the ND's position and orientation in the beam frame), the NuMI meson lists from g4numi, the
surveyed ND geometry and material, the NOvASoft generator module that calls BdNMC in process, and the simulation
FCLs. `configs/example.json` is an identity frame for tests, not NOvA's.

## Two ways to run

NOvASoft runs in an SL7 image. The JAX port needs glibc >= 2.27, which SL7 (2.17) does not have, so:

| Mode | Generation | Simulation | Endpoints |
|---|---|---|---|
| **two-stage job** (`jobs/two_stage.sh`) | JAX port in the BdNMC image (`RobMina/BdNMC` `gpu/containers/bdnmc-jax.def`), GPU or CPU | NOvASoft in its SL7 image, reading the TextFileGen file | any with Apptainer and CVMFS: FNAL grid, UVA Slurm, EAF |
| **single container, in process** | stock libBdNMC (branch `shared_library`, C API `BdNMC_capi.h`) built with NOvASoft's compiler (`containers/build_libbdnmc_sl7.sh`), called by a NOvASoft generator module | the same art job | CPU endpoints (FNAL grid) |

The two-stage mode uses the GPU port's speed; the in-process mode needs no intermediate file and no second image.

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

The configuration file is described in `bdnmc_nova/config.py`. Precision: float32 by default (the port's default),
`PRECISION=float64` in the job or `BDNMC_JAX_PRECISION=float64` for float64.

## Status

First version (2026-10-31): frame, TextFileGen writer, generation CLI and tests (laptop CPU); the job and container
recipes are written but not yet run end to end. Next: submission through the experiment-neutral endpoint package
(`hep-endpoints`), a run of the two-stage job on a grid worker and on a UVA GPU node, and a comparison with the
existing NOvA chain's TextFileGen files for the same card.

## Tests

```bash
python -m pytest
```
