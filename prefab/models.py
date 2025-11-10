"""
Fabrication process model definitions and configurations.

This module defines the data structures for representing nanofabrication processes
and their associated machine learning models. It includes Pydantic models for
fabrication specifications (foundry, process) and versioned model configurations
(dataset, version, release dates). Pre-configured model instances are provided
for common fabrication processes.
"""

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
    """

    foundry: str
    process: str


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
        return json.dumps(self.model_dump(), default=str)


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

models = dict(
    Generic_SOI=Generic_SOI_ANF1_d0,
)
