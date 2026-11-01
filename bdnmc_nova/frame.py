"""The NOvA Near Detector frame: BdNMC's beam frame (metres, origin at the target, z along the beam) to the
detector's local frame, as NOvASoft's simulation expects its primaries (cm, ns).

detector = R^T (beam - X0), with X0 the detector frame's origin in the beam frame and R the detector axes in the
beam frame, built from a pitch (about x) and a yaw (about y). The numbers are not in this repository: they come from
the experiment's geometry and are read from a configuration file (`config.load`).
"""
from dataclasses import dataclass, field
import math

import numpy as np


def axes(pitch, yaw):
    """R: the detector's x, y, z axes (columns) in the beam frame, for a pitch `pitch` (rad, about x) and a yaw
    `yaw` (rad, about y)."""
    ca, sa, cb, sb = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cb, -sb * sa, sb * ca],
                     [0.0, ca, sa],
                     [-sb, -cb * sa, cb * ca]])


@dataclass(frozen=True)
class Frame:
    """origin: X0 in the beam frame (m); pitch, yaw (rad); production_z: per BdNMC channel, the beam-frame z (m) of
    the production point, added to BdNMC's vertices for channels whose cards place production at z = 0 (e.g. brem
    in an absorber)."""
    origin: tuple
    pitch: float
    yaw: float
    production_z: dict = field(default_factory=dict)

    def rotation(self):
        """The matrix applied to beam-frame vectors: R^T."""
        return axes(self.pitch, self.yaw).T

    def origin_for(self, channel=None):
        """X0 shifted by the channel's production z, so that detector = R^T (vertex - origin_for(channel))."""
        x0 = np.asarray(self.origin, dtype=float).copy()
        x0[2] -= self.production_z.get(channel, 0.0)
        return x0

    def to_detector(self, pos_m, mom=None, channel=None):
        """Beam-frame positions (n, 3) in m (and momenta (n, 3)) to detector positions in cm (and momenta)."""
        rot = self.rotation()
        pos = (np.asarray(pos_m, dtype=float) - self.origin_for(channel)) @ rot.T * 100.0
        return pos if mom is None else (pos, np.asarray(mom, dtype=float) @ rot.T)
