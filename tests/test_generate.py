"""Stage 1 end to end on a generic card (CPU, a few 2^16-trial blocks): TextFileGen lines in the conventions of the
existing NOvA chain, and the frame applied."""
import json
from pathlib import Path

import numpy as np
import pytest

from bdnmc_nova import generate

HERE = Path(__file__).parent
CARD = HERE / "data" / "generic_mb_pi0_m005.dat"


def test_set_keys():
    text = "a 1\nb 2\n# c 3\n"
    assert generate.set_keys(text, {"b": "5"}) == "a 1\nb 5\n# c 3\n"
    with pytest.raises(KeyError):
        generate.set_keys(text, {"c": "1"})


def test_generate_writes_textfilegen(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"frame": {"origin_m": [0.0, 0.0, 490.0], "pitch_rad": 0.0, "yaw_rad": 0.0},
                               "card_overrides": {"seed": "7"}}))
    out, summ = tmp_path / "events.txt", tmp_path / "summary.json"
    s = generate.main([str(CARD), "--out", str(out), "--config", str(cfg), "--trials", str(4 << 16), "--batch", "4",
                       "--summary", str(summ)])
    assert s["events"] > 0 and s["accepted_dm"] >= s["events"] and "seed 7" in s["card_text"]
    lines = out.read_text().splitlines()
    parts = [l.split() for l in lines if len(l.split()) == 15]
    heads = [l.split() for l in lines if len(l.split()) == 2]
    assert len(heads) == s["events"] and len(parts) == 2 * s["accepted_dm"]
    el, dm = parts[0::2], parts[1::2]
    assert all(p[:6] == ["1", "11", "0", "-1", "-1", "-1"] for p in el)
    assert all(p[:6] == ["2", "18", "0", "-1", "-1", "-1"] for p in dm)
    z = np.array([float(p[13]) for p in el])  # cm, relative to the frame origin at z = 490 m (the card's detector)
    assert np.abs(z).max() < 1000.0
    assert json.loads(summ.read_text())["precision"] in ("float32", "float64")
