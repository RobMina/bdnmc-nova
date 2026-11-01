import json

import numpy as np
import pytest

from bdnmc_nova import config, frame


def test_axes_are_a_rotation():
    r = frame.axes(0.06, 0.015)
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-15)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_to_detector_is_rt_of_the_offset_in_cm():
    f = frame.Frame((11.0, -3.0, 990.0), 0.05, 0.02, {"brem_absorber": 725.0})
    pos = np.array([[11.5, -2.5, 995.0], [10.0, -4.0, 1000.0]])
    mom = np.array([[0.1, 0.0, 1.0], [0.0, 0.2, 2.0]])
    r = frame.axes(0.05, 0.02)
    got_pos, got_mom = f.to_detector(pos, mom)
    np.testing.assert_allclose(got_pos, (pos - f.origin) @ r * 100.0)  # each row: R^T (x - X0)
    np.testing.assert_allclose(got_mom, mom @ r)
    # a channel produced at z = 725 m: its vertices are relative to that point
    np.testing.assert_allclose(f.to_detector(pos - [0, 0, 725.0], channel="brem_absorber"), got_pos)


def test_config_round_trip(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"frame": {"origin_m": [1, 2, 3], "pitch_rad": 0.1, "yaw_rad": 0.2,
                                       "production_z_m": {"x": 5}}, "card_overrides": {"POT": "1e20"}}))
    f, over = config.load(p)
    assert f.origin == (1, 2, 3) and f.pitch == 0.1 and f.production_z == {"x": 5.0} and over == {"POT": "1e20"}
    monkeypatch.delenv("BDNMC_NOVA_CONFIG", raising=False)
    with pytest.raises(FileNotFoundError):
        config.load(None)
    monkeypatch.setenv("BDNMC_NOVA_CONFIG", str(p))
    assert config.load(None)[0] == f
