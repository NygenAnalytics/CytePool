"""Provenance records: how a cube was built and what it pooled.

A cube's provenance, ``Cube.provenance``, is a JSON object with ``source``, the
caller's own record of the data; ``construction``, how the cube was built; and
``operations``, the operations that derived it, oldest first, each with ``input``, the
fingerprint of the cube it was applied to where that is known without reading lazy
layers, and null otherwise. Any other key that a caller gives
(``Cube(provenance=...)``) is kept as it is. ``Cube.history()`` and
``describe_provenance`` read a record back in plain terms.
https://docs.nygen.io/CytePool/statistics/cubes#what-a-cube-records explains the
record, and https://docs.nygen.io/CytePool/reference/cube-files#provenance-keys
specifies the keys that CytePool writes.

``input`` is recorded when it is known without reading lazy layer values: for a cube
in memory, whose first operation hashes its layers once; for a cube read from a cube
file, which stores its digests; and for a lazy cube derived from one when every digest
it needs is known. It is null for a lazy cube derived from another whose groups were
not read whole, such as a cube file regrouped and then collapsed: call
``fingerprint()`` on the intermediate cube before deriving from it to record its
fingerprint. It is always null for a cube whose lazy layers are other array-likes (a
memmap or an HDF5 dataset), which can change.
"""

import json
from collections.abc import Mapping
from typing import Any

from .._messages import fitted_lines, quoted
from ._provenance_records import provenance_as_texts, provenance_from_texts

__all__ = ["describe_provenance"]

_SHOWN_IDS = 10  # ids listed before a list is shortened
_SHOWN_MEMBERS = 20  # merged groups or pooled samples listed in full
_SHOWN_CHARACTERS = 200  # of the JSON of one value of a record shown as it is
_ORDERS = {
    "sample_table": "the rows of the sample table given",
    "group_table": "the rows of the group table given",
    "sorted_labels": "the sorted distinct labels",
}
_ORIGINS = {
    "table": "the ids of a table",
    "ids": "ids given as a sequence",
    "data_frame_index": "the index of the counts data frame",
    "data_frame_columns": "the columns of the counts data frame",
}
# The functions that read cubes from scverse's data structures.
_IMPORTS = {"from_anndata": "an AnnData", "from_mudata": "a MuData"}
_LIBRARY_SIZES = {
    "cell_totals": "the total count of each cell",
    "given": "given for each cell (library_sizes)",
}
_REFERENCES = (
    "References: CytePool resolves none on its own. The source record's references "
    "need the tool that wrote them. A fingerprint names a cube without saying where "
    "it is: Cube.fingerprint() of a cube at hand says whether it is that cube."
)


def describe_provenance(provenance: Mapping[str, Any]) -> str:
    """Describe a provenance record in plain terms.

    The source record comes first, then how the cube was built, then each operation
    in order with what it kept, merged or pooled; long lists are shortened. Entries
    that the specification of the record does not describe
    (https://docs.nygen.io/CytePool/reference/cube-files#provenance-keys), such as
    those of cubes saved by earlier versions, are shown as JSON. The references that
    CytePool cannot follow on its own are reported as unresolved: those of the source
    record, and the fingerprints of the cubes that operations were applied to, which
    name those cubes without saying where they are.

    Args:
        provenance: A provenance record, such as ``Cube.provenance`` or the record of
            a cube that a result keeps.

    Returns:
        Text with one item per line, for printing, in lines of at most 80
        characters: a longer item continues on the next line, indented.

    Raises:
        TypeError: If ``provenance`` is not a mapping, holds a value JSON cannot
            hold, or is or holds a numpy masked array.
        ValueError: If it holds NaN, infinity or text that is not valid Unicode (a
            lone surrogate), or its ``operations`` is not a list.

    Example:
        >>> pooled = {"operation": "collapse_samples", "by": "donor"}
        >>> pooled["membership"] = {"D1": ["S1", "S3"], "D2": ["S2"]}
        >>> print(describe_provenance({"operations": [pooled]}))
        1. collapse_samples by 'donor'
          D1: S1 + S3
          D2: S2
    """
    record = provenance_from_texts(provenance_as_texts(provenance))
    if not record:
        return "Nothing is recorded."
    lines: list[str] = []
    references = False
    if "source" in record:
        lines += _source_lines(record["source"])
        references = isinstance(record["source"], dict)
    if "construction" in record:
        lines += _construction_lines(record["construction"])
        references |= _stacked_studies(record["construction"]) is not None
    for number, operation in enumerate(record.get("operations", []), start=1):
        lines += _operation_lines(number, operation)
        references |= isinstance(operation, dict) and operation.get("input") is not None
    others = [key for key in record if key not in ("source", "construction")]
    for key in others:
        if key != "operations":
            lines.append(f"{key}: {_json(record[key])}")
    if references:
        lines.append(_REFERENCES)
    return fitted_lines("\n".join(lines))


def _source_lines(source: Any) -> list[str]:
    if source is None:
        return ["Source: no record was given."]
    if not isinstance(source, dict):
        return [f"Source: {_json(source)}"]
    lines = [
        "Source, as the caller recorded it (unresolved: CytePool does not follow its "
        "references):"
    ]
    lines += [f"  {key}: {_json(value)}" for key, value in source.items()]
    return lines


def _construction_lines(construction: Any) -> list[str]:
    if not isinstance(construction, dict):
        return [f"Construction: {_json(construction)}"]
    settings = dict(construction)
    function = settings.pop("function", None)
    version = settings.pop("cytepool_version", None)
    header = f"Built by {function if isinstance(function, str) else _json(function)}"
    if version is not None:
        header += f" (CytePool {version})"
    if function == "from_cells" and isinstance(settings.get("n_cells"), int):
        header += f" from {settings.pop('n_cells'):,} cells"
    if function in _IMPORTS and isinstance(settings.get("rows"), int):
        header += f" from {settings.pop('rows'):,} profiles of {_IMPORTS[function]}"
        layout = settings.get("layout")
        if isinstance(layout, str):
            del settings["layout"]
            header += (
                ", with the sources given"
                if layout == "given"
                else f", in the layout of {layout}"
            )
    studies = _stacked_studies(construction)
    if studies is not None:
        header += f" from {len(studies):,} stud{'y' if len(studies) == 1 else 'ies'}"
    lines = [header + ":"]
    if studies is not None:
        del settings["studies"]
        lines += _study_lines(studies)
        by = settings.get("align_groups_by")
        if isinstance(by, str):
            del settings["align_groups_by"]
            lines.append(f"  groups aligned by the values of {by!r}")
        within = settings.get("within_study")
        if _is_text_list(within):
            del settings["within_study"]
            if within:
                lines.append(
                    f"  sample columns written study::value: {', '.join(within)}"
                )
    for axis in ("samples", "groups"):
        value = settings.get(axis)
        if (
            (function == "from_cells" or function in _IMPORTS)
            and isinstance(value, dict)
            and value.get("order") in _ORDERS
            and value.get("labels") in ("text", "integers")
            and len(value) == 2
        ):
            del settings[axis]
            text = _ORDERS[value["order"]]
            if value["labels"] == "integers":
                text += "; the labels were integers, written as decimal text"
            lines.append(f"  {axis}: {text}")
    for axis in ("samples", "features"):
        value = settings.get(axis)
        if isinstance(value, str) and (value in _ORIGINS or value == "positions"):
            del settings[axis]
            text = _ORIGINS.get(value, f"positions ({axis[:-1]}_0, {axis[:-1]}_1, ...)")
            lines.append(f"  {axis}: {text}")
    layers = settings.get("layers")
    if isinstance(layers, list) and all(isinstance(name, str) for name in layers):
        del settings["layers"]
        lines.append(f"  layers: {', '.join(layers)}")
    elif isinstance(layers, dict) and all(isinstance(s, str) for s in layers.values()):
        del settings["layers"]
        sources = (f"{name} from {source!r}" for name, source in layers.items())
        lines.append(f"  layers: {', '.join(sources)}")
    target, sizes = settings.get("target_sum"), settings.get("library_sizes")
    if (
        isinstance(target, int | float)
        and not isinstance(target, bool)
        and sizes in _LIBRARY_SIZES
    ):
        del settings["target_sum"], settings["library_sizes"]
        lines.append(
            f"  normalized values: each cell scaled to a total of {_number(target)}, "
            f"with library sizes {_LIBRARY_SIZES[sizes]}"
        )
    lines += [f"  {key}: {_json(value)}" for key, value in settings.items()]
    return lines


def _stacked_studies(construction: Any) -> dict[str, Any] | None:
    """The studies of a cube that ``CubeCollection.stack`` built, else None."""
    if not isinstance(construction, dict):
        return None
    studies = construction.get("studies")
    stacked = construction.get("function") == "CubeCollection.stack"
    return studies if stacked and isinstance(studies, dict) else None


def _study_lines(studies: dict[str, Any]) -> list[str]:
    """The studies of a stacked cube: each cube by its fingerprint, and the groups
    that were merged or renamed."""
    lines = []
    for name, study in studies.items():
        if not isinstance(study, dict):
            lines.append(f"  study {name!r}: {_json(study)}")
            continue
        found = study.get("fingerprint")
        named = found if isinstance(found, str) else _json(found)
        lines.append(f"  study {name!r}: the cube {named}, with its own record")
        groups = study.get("groups")
        if isinstance(groups, dict) and _is_membership(groups):
            changed = {
                aligned: members
                for aligned, members in groups.items()
                if members != [aligned]
            }
            for aligned, members in list(changed.items())[:_SHOWN_MEMBERS]:
                lines.append(f"    {aligned}: {_listing(members, ' + ')}")
            if len(changed) > _SHOWN_MEMBERS:
                lines.append(f"    ... ({len(changed):,} in all)")
    return lines


def _operation_lines(number: int, operation: Any) -> list[str]:
    if not isinstance(operation, dict):
        return [f"{number}. {_json(operation)}"]
    fields = dict(operation)
    name = fields.pop("operation", None)
    header = f"{number}. {name if isinstance(name, str) else _json(name)}"
    if name == "collapse_samples" and "by" in fields:
        by = fields["by"]
        if isinstance(by, str):
            header += f" by {fields.pop('by')!r}"
        elif _is_text_list(by) and by:
            header += f" by {quoted(fields.pop('by'))}"
        elif by is None:
            del fields["by"]
            header += " of every sample into one"
    if "input" in fields:
        found = fields.pop("input")
        named = found if isinstance(found, str) else _json(found)
        if name in _IMPORTS:
            header += (
                f", reading {_IMPORTS[name]} that CytePool exported from the cube "
                f"{named}"
            )
            if fields.get("source", False) is None:
                del fields["source"]
        elif found is None:
            header += ", applied to a cube whose fingerprint was not recorded"
        else:
            header += f", applied to the cube {named}"
    lines = [header]
    if name == "select":
        for axis in ("samples", "groups", "features"):
            if axis not in fields:
                continue
            kept = fields.pop(axis)
            if kept is None:
                lines.append(f"  {axis}: every {axis[:-1]}, as they were")
            elif _is_text_list(kept):
                lines.append(f"  {axis}: {_listing(kept, ', ')}")
            else:
                lines.append(f"  {axis}: {_json(kept)}")
    if name == "with_columns":
        for axis in ("samples", "groups", "features"):
            change = fields.get(axis, False)
            if change is None:
                del fields[axis]
            elif _is_column_change(change):
                del fields[axis]
                parts = [
                    f"{kind} {', '.join(change[kind])}"
                    for kind in ("added", "replaced")
                    if change[kind]
                ]
                lines.append(f"  {axis}: {'; '.join(parts) or 'no column'}")
    if name in ("regroup", "collapse_samples") and _is_membership(
        fields.get("membership")
    ):
        membership = fields.pop("membership")
        for new_id, members in list(membership.items())[:_SHOWN_MEMBERS]:
            lines.append(f"  {new_id}: {_listing(members, ' + ')}")
        if len(membership) > _SHOWN_MEMBERS:
            lines.append(f"  ... ({len(membership):,} in all)")
    lines += [f"  {key}: {_json(value)}" for key, value in fields.items()]
    return lines


def _number(value: float) -> str:
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return repr(value)


def _is_text_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_column_change(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"added", "replaced"}
        and all(map(_is_text_list, value.values()))
    )


def _is_membership(value: Any) -> bool:
    return isinstance(value, dict) and all(map(_is_text_list, value.values()))


def _listing(items: list[str], separator: str) -> str:
    shown = separator.join(items[:_SHOWN_IDS])
    if len(items) > _SHOWN_IDS:
        shown += f"{separator}... ({len(items):,} in all)"
    return shown


def _json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) > _SHOWN_CHARACTERS:
        return text[: _SHOWN_CHARACTERS - 3] + "..."
    return text
