"""Compile pinned pymatgen/CRC van der Waals radii for CrIStMa."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


SOURCE_COMMIT = "8495e941504cd5123701635b6572942c78d9589c"
SOURCE_DATA_COMMIT = "bc7b4d571bbc797d3e5b7edc8efdd4834676cc4c"
SOURCE_SHA256 = "b11669f8ccb0a9fe7647d9026ecbd30ee15ded7c464df828820a15768556d0aa"
DATASET_ID = "cristma.vdw_radii.pymatgen_crc_2024"
ELEMENTS = tuple(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
    "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
    "Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf "
    "Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm "
    "Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
PROPERTY_NAMES = ("Van der Waals radius", "Van der waals radius")


def _radius_value(properties: dict[str, object]) -> float | None:
    present = [name for name in PROPERTY_NAMES if name in properties]
    if not present:
        return None
    if len(present) != 1:
        raise ValueError("pymatgen contains duplicate van der Waals properties")
    raw = properties[present[0]]
    if isinstance(raw, dict):
        raw = raw["value"]
    if isinstance(raw, str):
        raw = raw.split()[0]
    if isinstance(raw, bool):
        raise ValueError(f"invalid van der Waals radius: {raw!r}")
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid van der Waals radius: {raw!r}")
    return value


def compile_source(source: Path, destination: Path) -> None:
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(f"Unexpected pymatgen-core source SHA-256: {source_sha256}")
    payload = json.loads(gzip.decompress(source_bytes))
    records = [
        {"symbol": symbol, "value_angstrom": radius}
        for symbol, properties in payload.items()
        if symbol != "_unit"
        and (radius := _radius_value(properties)) is not None
    ]
    order = {symbol: index for index, symbol in enumerate(ELEMENTS)}
    records.sort(key=lambda item: order[item["symbol"]])
    output = {
        "metadata": {
            "dataset_id": DATASET_ID,
            "version": "1",
            "source_repository": "https://github.com/materialsproject/pymatgen",
            "source_commit": SOURCE_COMMIT,
            "source_data_repository": "https://github.com/materialsproject/pymatgen-core",
            "source_data_commit": SOURCE_DATA_COMMIT,
            "source_path": "src/pymatgen/core/periodic_table.json.gz",
            "source_sha256": source_sha256,
            "source_license": "MIT",
            "scientific_source": (
                "CRC Handbook of Chemistry and Physics, 105th edition (2024)"
            ),
            "unit": "angstrom",
            "record_count": len(records),
        },
        "records": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    compile_source(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
