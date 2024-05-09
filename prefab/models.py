import json
from datetime import date

from pydantic import BaseModel


class Fab(BaseModel):
    foundry: str
    process: str
    material: str
    technology: str
    thickness: int
    has_sidewall: bool


class Model(BaseModel):
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

ANT_NanoSOI_ANF1_d9 = Model(
    fab=ANT_NanoSOI,
    version="ANF1",
    version_date=date(2024, 5, 6),
    dataset="d9",
    dataset_date=date(2024, 2, 6),
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
    ANT_NanoSOI=ANT_NanoSOI_ANF1_d9,
    ANT_NanoSOI_ANF1_d9=ANT_NanoSOI_ANF1_d9,
    ANT_SiN=ANT_SiN_ANF1_d1,
    ANT_SiN_ANF1_d1=ANT_SiN_ANF1_d1,
)
