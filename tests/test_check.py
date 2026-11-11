import csv
import importlib.util
from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_textfilegen", ROOT / "jobs" / "check_textfilegen.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

EVENTS = ("0 2\n1 11 0 -1 -1 -1 0.1 0.2 3 3.01 0.000511 10 20 30 3300\n2 18 0 -1 -1 -1 0 0 9 9.1 0.005 10 20 30 3300\n"
          "1 2\n1 11 0 -1 -1 -1 0.3 0.1 2 2.02 0.000511 -5 6 700 3310\n2 18 0 -1 -1 -1 0 0 4 4.2 0.005 -5 6 700 3310\n")


def write(tmp_path, events=EVENTS, scale=1.0, drop_last=False):
    ev = tmp_path / "events.txt"
    ev.write_text(events)
    rows = []
    for entry, (_, parts) in enumerate(check.read_textfilegen(ev)):
        for i, p in enumerate(parts):
            f = [float(x) for x in p[6:]]
            rows.append([entry, 1000001, 1, entry + 1, 0, i, p[0], p[1], p[2], f[5], f[6], f[7] * scale, f[8],
                         f[0], f[1], f[2], f[3], f[4]])
    rows = rows[:-2] if drop_last else rows
    pr = tmp_path / "primaries.csv"
    with open(pr, "w") as out:
        w = csv.writer(out)
        w.writerow(["entry", "run", "subrun", "event", "truth", "index", "status", "pdg", "mother",
                    "x", "y", "z", "t", "px", "py", "pz", "e", "mass"])
        w.writerows(rows)
    log = tmp_path / "sim.log"
    n = 1 if drop_last else 2
    log.write_text(f"TrigReport Events total = {n} passed = {n} failed = 0\n")
    return str(ev), str(pr), str(log)


def test_agreeing_primaries_pass(tmp_path, capsys):
    ev, pr, log = write(tmp_path)
    assert check.main([ev, pr, "--log", log]) == 0
    assert "2 art events, 4 primaries compared (PDG [11, 18])" in capsys.readouterr().out


def test_a_moved_vertex_fails(tmp_path):
    ev, pr, log = write(tmp_path, scale=1 + 1e-6)
    assert check.main([ev, pr, "--log", log]) == 1


def test_fewer_art_events_than_text_events_is_fine_but_counted(tmp_path, capsys):
    # an art job run on the first N events of a longer file
    ev, pr, log = write(tmp_path, drop_last=True)
    assert check.main([ev, pr, "--log", log]) == 0
    assert "1 art events, 2 primaries compared" in capsys.readouterr().out


def test_dump_script_compiles():
    py_compile.compile(str(ROOT / "bdnmc_nova" / "data" / "dump_primaries.py"), doraise=True)
