"""BdNMC events as input to NOvASoft's TextFileGen: HEPEVT-style text in the detector frame, cm and ns, with the
conventions of the existing NOvA chain: the recoil electron tracked (status 1, PDG 11), the recoil DM not tracked
(status 2, PDG 18), mother/daughter fields `0 -1 -1 -1`."""

DM_PDG, DM_STATUS, LINKS = 18, 2, "0 -1 -1 -1"


def write(res, path, frame, channel=None):
    """Write the accepted (unweighted) events of a bdnmc_jax result in the detector frame `frame`
    (`bdnmc_nova.frame.Frame`); `channel` selects the frame's production-z shift. Returns the number of events."""
    from bdnmc_jax import output  # imported when used: the import fixes bdnmc_jax's precision for the process
    return output.write_hepevt(res, path, origin=frame.origin_for(channel), rotation=frame.rotation(),
                               length_unit="cm", time_unit="ns", dm_pdg=DM_PDG, dm_status=DM_STATUS, links=LINKS)
