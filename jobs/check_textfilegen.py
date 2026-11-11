"""Check a NOvASoft art file's generator primaries against the TextFileGen input it was made from.

The primaries come from data/dump_primaries.py (run in the NOvASoft environment at the end of the simulation stage,
`primaries.csv`); this script needs only numpy. Art event i must be TextFileGen event i, with the same particles in
the same order: status, PDG and first mother exactly, vertex (cm), time (ns), momentum, energy and mass (GeV) to
`--rtol` relative to the value's scale. With `--log` (the art job's log) it also checks the TrigReport counts.

Usage: python jobs/check_textfilegen.py events.txt primaries.csv [--log sim.log] [--rtol 1e-9]
Exit status 0 when everything agrees, 1 otherwise.
"""
import argparse
import csv
import re
import sys

import numpy as np

FLOATS = ("x", "y", "z", "t", "px", "py", "pz", "e", "mass")
TEXT_COLUMNS = {"px": 6, "py": 7, "pz": 8, "e": 9, "mass": 10, "x": 11, "y": 12, "z": 13, "t": 14}


def read_textfilegen(path):
    """[(event header index, [particle fields])] of a TextFileGen file (`#` and blank lines skipped)."""
    events, lines = [], [ln.split() for ln in open(path) if ln.strip() and not ln.lstrip().startswith("#")]
    i = 0
    while i < len(lines):
        head = lines[i]
        if len(head) != 2:
            raise ValueError(f"{path}: expected an event header, got {' '.join(head)!r}")
        k = int(head[1])
        events.append((int(head[0]), lines[i + 1:i + 1 + k]))
        i += 1 + k
    return events


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("events")
    ap.add_argument("primaries")
    ap.add_argument("--log", help="the art job's log, for the TrigReport event counts")
    ap.add_argument("--rtol", type=float, default=1e-9)
    a = ap.parse_args(argv)

    ref = read_textfilegen(a.events)
    rows = list(csv.DictReader(open(a.primaries)))
    by_entry = {}
    for r in rows:
        by_entry.setdefault(int(r["entry"]), []).append(r)
    n_art = max(by_entry) + 1 if by_entry else 0
    bad = []
    if n_art > len(ref):
        bad.append(f"{n_art} art events with primaries but only {len(ref)} TextFileGen events")
    worst = {k: 0.0 for k in FLOATS}
    n_particles, empty = 0, [i for i in range(n_art) if i not in by_entry]
    for i in range(min(n_art, len(ref))):
        got, (_, want) = by_entry.get(i, []), ref[i]
        if len(got) != len(want):
            bad.append(f"event {i}: {len(got)} primaries in the art file, {len(want)} in the TextFileGen file")
            continue
        for g, w in zip(got, want):
            n_particles += 1
            if (int(g["status"]), int(g["pdg"]), int(g["mother"])) != (int(w[0]), int(w[1]), int(w[2])):
                bad.append(f"event {i}: status/pdg/mother {g['status']} {g['pdg']} {g['mother']} vs {w[:3]}")
            for k in FLOATS:
                gv, wv = float(g[k]), float(w[TEXT_COLUMNS[k]])
                worst[k] = max(worst[k], abs(gv - wv) / max(abs(wv), 1.0))
    bad += [f"{k}: relative difference {worst[k]:.3g} > {a.rtol:g}" for k in FLOATS if worst[k] > a.rtol]
    if empty:
        bad.append(f"{len(empty)} art events without primaries (first {empty[:5]})")

    pdgs = sorted({int(r["pdg"]) for r in rows})
    print(f"{n_art} art events, {n_particles} primaries compared (PDG {pdgs}) against {len(ref)} TextFileGen events")
    print("max relative difference (|art - text| / max(|text|, 1)): "
          + ", ".join(f"{k} {worst[k]:.2g}" for k in FLOATS))
    if a.log:
        m = re.search(r"TrigReport Events total = (\d+) passed = (\d+) failed = (\d+)", open(a.log).read())
        if not m:
            bad.append(f"{a.log}: no TrigReport line")
        else:
            total, passed, failed = map(int, m.groups())
            print(f"TrigReport: total {total}, passed {passed}, failed {failed}")
            if (total, passed, failed) != (n_art, n_art, 0):
                bad.append(f"TrigReport {total}/{passed}/{failed}, expected {n_art}/{n_art}/0")
    for b in bad:
        print("MISMATCH", b)
    print("OK" if not bad else f"{len(bad)} problems")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
