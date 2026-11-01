# Containers

**Two-stage mode.** Two images in one job:
- generation: the BdNMC JAX image, built from `gpu/containers/bdnmc-jax.def` in `RobMina/BdNMC` (Debian bookworm,
  glibc 2.36; JAX with CUDA 12; `bdnmc_jax`), plus this package on `PYTHONPATH` (`jobs/two_stage.sh` binds it);
- simulation: NOvA's SL7 image with NOvASoft from CVMFS, as the experiment documents it.

**Single-container mode.** NOvA's SL7 image only: build stock libBdNMC there with NOvASoft's compiler,

```bash
bash containers/build_libbdnmc_sl7.sh <source dir> <install prefix>
```

(after NOvASoft's setup, inside the SL7 container), then link the experiment's in-process generator module against
`<prefix>/lib/libBdNMC.so` and `<prefix>/include/BdNMC_capi.h`. The module itself is NOvASoft code and is kept in
the experiment's private repository. The C API configures a run from a card path (`bdnmc_configure`) or from
structured parameters and meson rows (`bdnmc_configure_meson`), runs it, and returns events particle by particle.

The user builds images and binaries; these recipes are not run by any automation here.
