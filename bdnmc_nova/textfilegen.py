"""Deprecated: the old import path of `bdnmc_nova.hepevt`. Import `bdnmc_nova.hepevt` instead; this alias will be
removed in a later release."""
import warnings

from .hepevt import DM_PDG, DM_STATUS, LINKS, head, write  # noqa: F401

__all__ = ["DM_PDG", "DM_STATUS", "LINKS", "head", "write"]

warnings.warn("bdnmc_nova.textfilegen is deprecated; import bdnmc_nova.hepevt instead", DeprecationWarning,
              stacklevel=2)
