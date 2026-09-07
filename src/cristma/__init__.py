"""Stable public API for the Qt-free CrIStMa scientific library."""

from __future__ import annotations

from pathlib import Path

from .structure import CrystalStructure
from .io.cif.document import CifDocument
from .io.cif.writer import write_cif_document, write_crystal_cif
from .io.shelx.document import ShelxDocument
from .io.shelx.writer import (
    ShelxWriteOptions,
    write_crystal_shelx,
    write_shelx_document,
)
from .io.formats import FormatDescriptor, builtin_format_descriptors
from .io.registry import FormatRegistry
from .io.result import ReadResult

__version__ = "0.1.0b6"

_formats = FormatRegistry(builtin_format_descriptors())


def structure_formats() -> tuple[FormatDescriptor, ...]:
    """Return immutable descriptions of built-in structural formats."""

    return builtin_format_descriptors()


def read(path: str | Path, *, format: str | None = None) -> ReadResult:
    """Read a structure file through the native format registry."""

    result = _formats.read(path, format=format)
    if not isinstance(result, ReadResult):
        raise TypeError("structure format handler returned an invalid read result")
    return result


def read_text(
    source: str,
    *,
    format: str | None = None,
    source_name: str | None = None,
) -> ReadResult:
    """Read already decoded structure text through content probing."""

    result = _formats.read_text(
        source,
        format=format,
        source_name=source_name,
    )
    if not isinstance(result, ReadResult):
        raise TypeError("structure format handler returned an invalid read result")
    return result


def write(
    value: CifDocument | ShelxDocument | CrystalStructure,
    path: str | Path,
    *,
    mode: str | None = None,
    format: str | None = None,
    options: ShelxWriteOptions | None = None,
) -> None:
    """Write a source document or canonical structure in the selected format."""

    if isinstance(value, CifDocument):
        if format not in {None, "cif"} or options is not None:
            raise ValueError("CIF document writing does not accept format options")
        selected_mode = "preserve" if mode is None else mode
        if selected_mode != "preserve":
            raise ValueError("CifDocument write mode must be 'preserve'")
        rendered = write_cif_document(value, mode=selected_mode)
    elif isinstance(value, ShelxDocument):
        if format not in {None, "shelx", "res", "ins"} or options is not None:
            raise ValueError("SHELX document writing does not accept format options")
        selected_mode = "preserve" if mode is None else mode
        rendered = write_shelx_document(value, mode=selected_mode)
    elif isinstance(value, CrystalStructure):
        selected_mode = "canonical" if mode is None else mode
        if selected_mode != "canonical":
            raise ValueError("Crystal write mode must be 'canonical'")
        selected_format = "cif" if format is None else format.casefold()
        if selected_format == "cif":
            if options is not None:
                raise ValueError("CIF crystal writing does not accept SHELX options")
            rendered = write_crystal_cif(value)
        elif selected_format in {"shelx", "res", "ins"}:
            rendered = write_crystal_shelx(value, options=options)
        else:
            raise ValueError(f"unsupported crystal output format: {format!r}")
    else:
        raise TypeError("write accepts only source documents or CrystalStructure")
    Path(path).write_bytes(rendered.encode("utf-8"))


__all__ = [
    "ReadResult",
    "__version__",
    "read",
    "read_text",
    "structure_formats",
    "write",
]
