"""
Base evaluation models.

Pre-configured model instances for common fabrication processes.
"""

from datetime import date

from .base import Fab, Model


Generic = Fab(
    foundry="Generic",
    process="SOI",
)

Generic_SOI_ANF1_d0 = Model(
    fab=Generic,
    version="ANF1",
    version_date=date(2025, 11, 7),
    dataset="d0",
    dataset_date=date(2025, 11, 7),
    tag="",
)

# Export models with user-facing names
__models__ = {
    "Generic_SOI": Generic_SOI_ANF1_d0,
}
