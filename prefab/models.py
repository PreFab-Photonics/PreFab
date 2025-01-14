"""Models for the PreFab library."""

import json
from datetime import date

from pydantic import BaseModel


class Fab(BaseModel):
    """
    Represents a fabrication process in the PreFab model library.

    Attributes
    ----------
    foundry : str
        The name of the foundry where the fabrication process takes place.
    process : str
        The specific process used in the fabrication.
    material : str
        The material used in the fabrication process.
    technology : str
        The technology used in the fabrication process.
    thickness : int
        The thickness of the material used, measured in nanometers.
    has_sidewall : bool
        Indicates whether the fabrication has angled sidewalls.
    """

    foundry: str
    process: str
    material: str
    technology: str
    thickness: int
    has_sidewall: bool


class Model(BaseModel):
    """
    Represents a model of a fabrication process including versioning and dataset detail.

    Attributes
    ----------
    fab : Fab
        An instance of the Fab class representing the fabrication details.
    version : str
        The version identifier of the model.
    version_date : date
        The release date of this version of the model.
    dataset : str
        The identifier for the dataset used in this model.
    dataset_date : date
        The date when the dataset was last updated or released.
    tag : str
        An optional tag for additional categorization or notes.

    Methods
    -------
    to_json()
        Serializes the model instance to a JSON formatted string.
    """

    fab: Fab
    version: str
    version_date: date
    dataset: str
    dataset_date: date
    tag: str

    def to_json(self):
        return json.dumps(self.dict(), default=str)


ANT_NanoSOI = Fab(
    foundry="ANT",
    process="NanoSOI",
    material="SOI",
    technology="E-Beam",
    thickness=220,
    has_sidewall=False,
)

ANT_SiN = Fab(
    foundry="ANT",
    process="SiN",
    material="SiN",
    technology="E-Beam",
    thickness=400,
    has_sidewall=True,
)

ANT_NanoSOI_ANF1_d10 = Model(
    fab=ANT_NanoSOI,
    version="ANF1",
    version_date=date(2024, 5, 6),
    dataset="d10",
    dataset_date=date(2024, 6, 8),
    tag="",
)

ANT_SiN_ANF1_d1 = Model(
    fab=ANT_SiN,
    version="ANF1",
    version_date=date(2024, 5, 6),
    dataset="d1",
    dataset_date=date(2024, 1, 31),
    tag="",
)

models = dict(
    ANT_NanoSOI=ANT_NanoSOI_ANF1_d10,
    ANT_NanoSOI_ANF1_d10=ANT_NanoSOI_ANF1_d10,
    ANT_SiN=ANT_SiN_ANF1_d1,
    ANT_SiN_ANF1_d1=ANT_SiN_ANF1_d1,
)
