"""Generate BdNMC events with the JAX port for the NOvA Near Detector and write TextFileGen input (stage 1 of the
two-stage job; stage 2 runs NOvASoft's simulation on the file).

Usage: python -m bdnmc_nova.generate <card> --out events.txt [--config cfg.json] [--trials N] [--channel NAME]
       [--set key=value ...] [--hdf5 events.h5] [--batch 64] [--cap 1024] [--parent-sampler stock|envelope]
       [--summary summary.json]

The card is a BdNMC parameter card; `--set` and the configuration's `card_overrides` replace keys in it (the
configuration's first, then `--set`). The summary JSON records the prediction, the counts, the precision and the
card actually run.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time

from . import config, textfilegen


def set_keys(text, overrides):
    """Card text with each `key value` line of `overrides` replaced; a key missing from the card is an error."""
    lines, seen = [], set()
    for line in text.splitlines():
        key = line.split()[0] if line.split() else ""
        if key in overrides:
            lines.append(f"{key} {overrides[key]}")
            seen.add(key)
        else:
            lines.append(line)
    missing = set(overrides) - seen
    if missing:
        raise KeyError(f"keys not in the card: {sorted(missing)}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("card", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="TextFileGen input to write")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--trials", type=int, default=1 << 24)
    ap.add_argument("--channel", help="the configuration's production_z_m entry to apply")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--hdf5", type=Path)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--parent-sampler", default="stock")
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args(argv)
    frame, overrides = config.load(args.config)
    overrides |= dict(kv.split("=", 1) for kv in args.set)
    from bdnmc_jax import PRECISION, output, pipeline  # after argument parsing: importing sets the precision
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / args.card.name
        card.write_text(set_keys(args.card.read_text(), overrides))
        t0 = time.perf_counter()
        res = pipeline.events(card, args.trials, batch=args.batch, cap=args.cap, parent_sampler=args.parent_sampler)
        n_events = textfilegen.write(res, args.out, frame, args.channel)
        if args.hdf5:
            output.write_hdf5(res, args.hdf5)
        summary = {"card": str(args.card), "card_text": card.read_text(), "overrides": overrides,
                   "channel": args.channel, "trials": res["trials"], "accepted_dm": res["accepted"],
                   "events": n_events, "se": res["se"], "se_weighted": res["se_weighted"], "precision": PRECISION,
                   "wall_s": time.perf_counter() - t0, "textfilegen": str(args.out)}
    print(f"{n_events} events ({res['accepted']} DM) to {args.out}; se {res['se']:.6g} ({PRECISION})")
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
