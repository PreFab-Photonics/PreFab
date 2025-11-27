"""
Base model definitions for fabrication processes.

This module defines the core data structures for representing nanofabrication
processes and their associated models.
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
