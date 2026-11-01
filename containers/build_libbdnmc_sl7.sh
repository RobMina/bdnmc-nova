#!/bin/bash
# Build stock libBdNMC (RobMina/BdNMC branch shared_library: libBdNMC.so + its C API, src/BdNMC_capi.h) inside
# NOvA's SL7 environment, with NOvASoft's own compiler, so that a NOvASoft generator module can call BdNMC in
# process (the single-container mode). Run it in a shell where NOvASoft is already set up (e.g. inside the SL7
# container after sourcing the NOvASoft setup script); the user builds binaries.
# Usage: build_libbdnmc_sl7.sh <source dir> <install prefix> [git ref, default shared_library]
set -eo pipefail
SRC=${1:?source directory (cloned here if missing)}
PREFIX=${2:?install prefix}
REF=${3:-shared_library}

echo "compiler: $(command -v g++) $(g++ --version | head -1)"
[[ -r /etc/redhat-release ]] && echo "os: $(cat /etc/redhat-release)"
if [[ ! -d $SRC/.git ]]; then
    git clone https://github.com/RobMina/BdNMC.git "$SRC"
fi
git -C "$SRC" fetch origin
git -C "$SRC" checkout "$REF"
git -C "$SRC" submodule update --init  # most stock sources are in BdNMC_LANL (lanl/BdNMC_LANL)
echo "BdNMC $(git -C "$SRC" rev-parse --short HEAD)"
cd "$SRC/build"
make clean >/dev/null 2>&1 || true
make -j"${NPROC:-4}" all
make install PREFIX="$PREFIX"
git -C "$SRC" rev-parse HEAD > "$PREFIX/COMMIT"
# the C API the in-process module uses
grep -q bdnmc_configure_meson "$PREFIX/include/BdNMC_capi.h"
nm -D "$PREFIX/lib/libBdNMC.so" | grep -q bdnmc_configure_meson
echo "libBdNMC installed in $PREFIX"
