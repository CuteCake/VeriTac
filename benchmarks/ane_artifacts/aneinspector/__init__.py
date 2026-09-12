"""Read-only compiled-ANE artifact inspector.

Independent, stdlib-only, bounded inspection of explicit files / run-dirs.
Owned solely by the ane_artifacts worker.  Reports container-level metadata
(header / load commands / segments / sections / strings) and deliberately
does NOT model instruction or task-descriptor semantics, which are
generation-specific and not assumed valid on the current target.
"""

from .version import __version__

__all__ = ["__version__"]
