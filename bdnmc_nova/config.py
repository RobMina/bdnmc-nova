"""Site and geometry configuration, kept out of the repository.

A JSON file (path from `--config` or `$BDNMC_NOVA_CONFIG`) with the detector frame and any card overrides:

    {
      "frame": {"origin_m": [x0, y0, z0], "pitch_rad": a, "yaw_rad": b, "production_z_m": {"<channel>": z, ...}},
      "card_overrides": {"<card key>": "<value>", ...}
    }

The NOvA values come from the experiment's geometry and stay in the experiment's private configuration;
`configs/example.json` is an identity frame for tests and examples, not NOvA's.
"""
import json
import os
from pathlib import Path

from .frame import Frame


def load(path=None):
    path = path or os.environ.get("BDNMC_NOVA_CONFIG")
    if not path:
        raise FileNotFoundError("no configuration: pass --config or set BDNMC_NOVA_CONFIG (see configs/example.json)")
    cfg = json.loads(Path(path).read_text())
    f = cfg["frame"]
    frame = Frame(tuple(f["origin_m"]), float(f["pitch_rad"]), float(f["yaw_rad"]),
                  {k: float(v) for k, v in f.get("production_z_m", {}).items()})
    return frame, dict(cfg.get("card_overrides", {}))
