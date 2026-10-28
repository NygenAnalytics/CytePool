"""Several studies' cubes, aligned by sample, gene and group: ``CubeCollection``.

:class:`~cytepool.CubeCollection` aligns the cubes of several studies by sample, gene
and group (cell type), says what each study measured and has, and keeps each cube as
it was given, for an analysis of each study on the collection's axes or a joint model
of the studies stacked into one cube.
https://docs.nygen.io/CytePool/statistics/aligning-studies explains why and how.

References:
    Nakatsuka N, Adler D, Jiang L, Hartman A, Cheng E, Klann E, Satija R (2025).
    Improving reproducibility of differentially expressed genes in single-cell
    transcriptomic studies of neurodegenerative diseases through meta-analysis.
    Nature Communications 16, 7436. doi:10.1038/s41467-025-62579-z

    Diehl AD, Meehan TF, Bradford YM, Brush MH, Dahdul WM, Dougall DS, He Y,
    Osumi-Sutherland D, Ruttenberg A, Sarntivijai S, Van Slyke CE, Vasilevsky NA,
    Haendel MA, Blake JA, Mungall CJ (2016). The Cell Ontology 2016: enhanced content,
    modularization, and ontology interoperability. Journal of Biomedical Semantics 7,
    44. doi:10.1186/s13326-016-0088-7
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from .. import __version__
from .._columns import text_column, values_equal
from .._masked_arrays import reject_masked
from .._messages import close_matches, fitted_lines, listing, quoted
from ..table import Table
from ._axes import INT64_MAX, selected_positions
from .cube import LAYER_NAMES, Cube

__all__ = ["CubeCollection"]

_SEPARATOR = "::"  # between a study's name and a sample id: study::sample_id
_ONTOLOGY_COLUMN = "cell_ontology_id"
_NORMALIZED_LAYERS = ("log_sum", "log_sum_sq", "linear_sum")
_FLOAT_LIMIT = 2**53  # float64 holds every whole number below it
# An Ensembl version suffix, as in ENSG00000141510.17.
_VERSION = re.compile(r"\.\d+$")
_KINDS = {
    "O": "text",
    "b": "booleans",
    "i": "integers",
    "u": "integers",
    "f": "floats",
    "c": "complex numbers",
    "M": "datetimes",
    "m": "timedeltas",
}
_NUMBERS = frozenset({"integers", "floats", "complex numbers"})


class CubeCollection:
    """Several studies' cubes, aligned by sample, gene and group.

    Studies are compared or combined only once their samples, genes and groups (cell
    types) are aligned. A collection aligns the cubes of several named studies and
    keeps each cube as it was given (``cubes``): it reads no layer, and each analysis
    of a study reads that study's cube. It is immutable: ``select`` returns a new
    collection. https://docs.nygen.io/CytePool/statistics/aligning-studies explains
    the alignment.

    Samples:
        Sample ids mean something only inside their study, so the collection names
        every sample ``study::sample_id``: sample ``S1`` of one study is not sample
        ``S1`` of another, and study names cannot hold ``::``. The same holds for
        donor ids and other columns of ids, which ``within_study`` names: their
        values are written ``study::value`` too, so that a design does not take
        donor ``D1`` of two studies for one person. ``samples`` has ``sample_id``,
        ``study`` and every other column of any study's samples, in the order in
        which the studies, taken in order, first have them; a study without a column
        has missing values in it. A column keeps its type when every study gives it
        the same; integers of different types become int64, integers and floats
        float64, datetimes take the finer unit, and integer and boolean columns that
        need a missing value become float64 with NaN, as in ``Table``. A column that
        holds different kinds of values in different studies, such as text in one
        and numbers in another, is refused.

    Genes:
        Features are aligned by ``feature_id``, which must hold ids of one kind in
        every study, such as Ensembl gene ids, compared as they are: ids are not
        converted, so Ensembl ids with versions (``ENSG00000141510.17``) must have
        them removed first. A study measured a feature when its
        cube has it and does not mark it False in the feature column ``measured``:
        tools that give every study the same genes fill those a study did not
        measure with zeros, and ``measured`` tells them apart. The collection's
        features are those that at least one study measured, in the order in which
        the studies first list them, and ``measured`` says which study measured
        which: a feature that a study did not measure is missing there, not zero.
        ``features`` has ``feature_id`` and, when a study's features have one,
        ``symbol``, the first symbol a study gives the feature, taking the studies
        in order.

    Groups:
        Groups are aligned by the values of a text column of the group tables,
        ``align_groups_by``: by default ``cell_ontology_id``, the Cell Ontology id
        (Diehl et al. 2016), when every study's group table has that column, and
        ``group_id`` otherwise. ``group_mapping`` gives the aligned id of some of a
        study's groups instead, or None to leave a group out. Each aligned group is
        named by its aligned id, or by the name ``group_names`` gives it, such as
        ``{"CL:0001054": "CD14 monocytes"}`` for groups aligned by their Cell Ontology
        ids, and groups of one study with the same aligned id are merged: their
        layers and cell numbers are summed, exactly, as ``Cube.regroup`` sums them.
        Groups are aligned by equal values only: give merges up the Cell Ontology,
        such as of CD4 T cell subsets into CD4 T cells, in ``group_mapping``.
        ``group_members`` lists the groups of each study under their aligned ids.
        ``groups`` has ``group_id``, the aligned id or its name, and every other
        column of any study's group table: a group keeps a column's value where the
        studies that have the column agree on it for every member, and has a missing
        value otherwise. With ``group_names``, the column ``align_groups_by`` holds
        each group's aligned id, unless the groups are aligned by ``group_id``.

    Analyses of each study:
        ``aligned_cube(study)`` is the study's cube on the collection's axes: its
        groups under their aligned ids and the features it measured, in the
        collection's order, and its own samples with their own ids, so that a design
        built from its sample table applies. Any analysis of one cube, such as
        ``limma_voom`` with each study's own design, then gives results keyed by
        the aligned groups; ``analyse_each_study`` runs one on every study, and the
        ``StudyResults`` it returns lines them up gene by gene, for
        ``meta_analysis`` and the other functions that combine studies.

    Stacking:
        ``stack()`` puts the studies into one cube, for a joint model with the study
        as a covariate: its samples are those of every study, as ``samples`` lists
        them, and its features and groups those that every study measured and has.
        A cube holds no missing values, so ``stack`` refuses a feature or a group
        that a study lacks rather than fill it with zeros. Each study keeps its own
        library sizes, and normalized layers are stacked only from studies
        normalized alike.

    Provenance:
        ``provenance`` records each study's cube by its fingerprint, with its own
        provenance, the settings of the alignment with each study's groups under
        their aligned ids, and each selection since. A stacked cube records the
        studies and the alignment in its provenance (``cytepool.cube.provenance``).

    Args:
        cubes: Study name to cube, at least one, in the order the collection keeps.
            Study names are non-empty and hold no ``::``.
        align_groups_by: The text column of the group tables whose values align the
            groups, such as ``"cell_ontology_id"`` or ``"group_id"``. By default
            ``"cell_ontology_id"`` when every study's group table has it, and
            ``"group_id"`` otherwise.
        group_mapping: Study name to a mapping of its group ids to aligned ids, which
            take the place of those groups' values of ``align_groups_by``; None
            leaves a group out. Groups of one study given the same aligned id are
            merged.
        group_names: Aligned id to the name that becomes the group's id, such as a
            cell type's name for its Cell Ontology id, for some or all of the
            aligned groups; the others keep their aligned ids. Each name must differ
            from every other group's id.
        within_study: Sample columns whose values mean something only inside their
            study, such as donor ids, written ``study::value`` as sample ids are.

    Raises:
        TypeError: If an argument has the wrong type, a column of the sample or
            group tables holds different kinds of values in different studies, a
            column of ``within_study`` holds neither text nor integers, or
            ``group_mapping``, ``group_names`` or ``within_study`` is or holds a
            numpy masked array.
        ValueError: If there is no study, a study name is empty or holds ``::``, a
            group has no value of ``align_groups_by`` and no aligned id, a study has
            no group or measured no feature, a study shares no feature, or no
            group, with any other study, or a name of ``group_names`` is empty or
            the id of another group.
        KeyError: If ``group_mapping`` names a study or a group that is not there,
            ``group_names`` an aligned id that no study has, or ``within_study`` or
            ``align_groups_by`` a column that no study has.

    Example:
        >>> import numpy as np
        >>> from cytepool.cube import Cube
        >>> def study(genes, donors):
        ...     return Cube(
        ...         {"counts": np.ones((1, len(donors), len(genes)), dtype=np.int64)},
        ...         samples={"sample_id": donors},
        ...         groups={"group_id": ["B"]},
        ...         features={"feature_id": genes},
        ...     )
        >>> collection = CubeCollection(
        ...     {"A": study(["g1", "g2"], ["D1", "D2"]), "B": study(["g2", "g3"], ["D1"])}
        ... )
        >>> collection.samples["sample_id"].tolist()
        ['A::D1', 'A::D2', 'B::D1']
        >>> collection.measured
        array([[ True,  True, False],
               [False,  True,  True]])
        >>> collection.stack().features["feature_id"].tolist()
        ['g2']
    """

    __slots__ = (
        "_studies",
        "_cubes",
        "_members",
        "_measured",
        "_samples",
        "_groups",
        "_features",
        "_settings",
        "_operations",
        "_study_index",
        "_group_index",
        "_feature_index",
    )

    _studies: tuple[str, ...]
    _cubes: dict[str, Cube]
    _members: dict[str, dict[str, tuple[str, ...]]]  # study: aligned id: its groups
    _measured: np.ndarray  # bool, (n_studies, n_features), read-only
    _samples: Table
    _groups: Table
    _features: Table
    _settings: str  # JSON: the alignment's settings and the groups it aligned
    _operations: tuple[str, ...]  # JSON texts of the selections, oldest first
    _study_index: dict[str, int]
    _group_index: dict[str, int]
    _feature_index: dict[str, int]

    def __init__(
        self,
        cubes: Mapping[str, Cube],
        *,
        align_groups_by: str | None = None,
        group_mapping: Mapping[str, Mapping[str, str | None]] | None = None,
        group_names: Mapping[str, str] | None = None,
        within_study: str | Sequence[str] = (),
    ) -> None:
        studies = _checked_cubes(cubes)
        given = {study: cubes[study] for study in studies}
        within = _checked_within_study(within_study, given)
        mapping = _checked_group_mapping(group_mapping, given)
        by = _alignment_column(align_groups_by, given)
        members = {
            study: _aligned_groups(study, cube, by, mapping.get(study, {}))
            for study, cube in given.items()
        }
        names = _checked_group_names(group_names, members)
        if names:
            members = {
                study: {
                    names.get(aligned, aligned): ids for aligned, ids in groups.items()
                }
                for study, groups in members.items()
            }
        group_ids = list(
            dict.fromkeys(group for study in studies for group in members[study])
        )
        feature_ids, measured = _measured_features(given)
        for position, study in enumerate(studies):
            if not members[study]:
                raise ValueError(
                    f"study {study!r} has no group to align: group_mapping leaves out "
                    "every group of its cube"
                )
            if not measured[position].any():
                raise ValueError(
                    f"study {study!r} measured no feature: its cube has no features, "
                    "or marks every one as not measured in its feature column "
                    "'measured'"
                )
        if len(studies) > 1:
            _check_shared_features(studies, feature_ids, measured)
            _check_shared_groups(studies, members, by)
        settings: dict[str, Any] = {
            "cytepool_version": __version__,
            "align_groups_by": by,
            "group_mapping": {study: dict(groups) for study, groups in mapping.items()}
            or None,
        }
        if names:
            settings["group_names"] = names
        settings |= {
            "within_study": within,
            "groups": {
                study: {aligned: list(ids) for aligned, ids in members[study].items()}
                for study in studies
            },
        }
        _assemble(
            self,
            studies,
            given,
            {
                study: {aligned: tuple(ids) for aligned, ids in members[study].items()}
                for study in studies
            },
            group_ids,
            feature_ids,
            measured,
            json.dumps(settings),
            (),
        )

    # Immutability, copying and pickling.

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("CubeCollection is immutable; select() returns a new one")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CubeCollection is immutable")

    def __copy__(self) -> "CubeCollection":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "CubeCollection":
        return self

    def __reduce__(self) -> tuple[Callable[..., "CubeCollection"], tuple[Any, ...]]:
        return (
            _restore_collection,
            (
                self._studies,
                self._cubes,
                self._members,
                self._measured,
                self._samples,
                self._groups,
                self._features,
                self._settings,
                self._operations,
            ),
        )

    # Sizes and tables.

    @property
    def studies(self) -> tuple[str, ...]:
        """The study names, in order."""
        return self._studies

    @property
    def n_studies(self) -> int:
        """The number of studies."""
        return len(self._studies)

    @property
    def n_samples(self) -> int:
        """The number of samples of every study together."""
        return len(self._samples)

    @property
    def n_groups(self) -> int:
        """The number of aligned groups."""
        return len(self._groups)

    @property
    def n_features(self) -> int:
        """The number of features that at least one study measured."""
        return len(self._features)

    @property
    def cubes(self) -> Mapping[str, Cube]:
        """The cube of every study as it was given, by study name (read-only)."""
        return MappingProxyType(self._cubes)

    @property
    def samples(self) -> Table:
        """The samples of every study, with ``sample_id`` (``study::sample_id``),
        ``study`` and every other sample column."""
        return self._samples

    @property
    def groups(self) -> Table:
        """The aligned groups, with ``group_id``, the aligned id, and the group
        columns on whose values the studies agree."""
        return self._groups

    @property
    def features(self) -> Table:
        """The features that at least one study measured, with ``feature_id`` and
        ``symbol`` when a study's features have it."""
        return self._features

    @property
    def measured(self) -> np.ndarray:
        """Whether each study measured each feature, as read-only booleans of shape
        ``(n_studies, n_features)``."""
        return self._measured.view()

    @property
    def group_members(self) -> Table:
        """The groups of each study under their aligned ids, one row per group of a
        study that the collection holds, with ``study``, ``group_id`` (the aligned
        id) and ``study_group_id`` (the group's id in the study's cube), studies and
        aligned groups in order. An aligned group with several rows for one study
        merges them; a study without a row for an aligned group does not have it."""
        rows = [
            (study, aligned, member)
            for study in self._studies
            for aligned, ids in self._members[study].items()
            for member in ids
        ]
        return Table(
            {
                name: text_column([row[position] for row in rows])
                for position, name in enumerate(("study", "group_id", "study_group_id"))
            }
        )

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of how the collection was made, as a new dict.

        It holds ``studies``: each study's cube, by study name in order, with its
        ``fingerprint`` and its own ``provenance``; ``alignment``: the version of
        CytePool that aligned them, ``align_groups_by``, ``group_mapping``,
        ``group_names`` (when given), ``within_study`` and ``groups``, each study's
        groups under their aligned ids when the collection was made; and
        ``operations``: each selection since, with the studies, groups and features
        it kept (null for an axis given as None). The fingerprints cost what
        ``Cube.fingerprint`` costs: nothing for a cube file, which stores them, and a
        pass over the values of a cube in memory the first time.
        """
        alignment = json.loads(self._settings)
        return {
            "studies": {
                study: {
                    "fingerprint": self._cubes[study].fingerprint(),
                    "provenance": self._cubes[study].provenance,
                }
                for study in self._studies
            },
            "alignment": alignment,
            "operations": [json.loads(text) for text in self._operations],
        }

    def __repr__(self) -> str:
        settings = json.loads(self._settings)
        lines = [
            f"CubeCollection of {self.n_studies} stud"
            f"{'y' if self.n_studies == 1 else 'ies'}: {self.n_samples:,} samples, "
            f"{self.n_groups:,} groups, {self.n_features:,} features",
            f"  groups aligned by {settings['align_groups_by']!r}",
        ]
        for position, study in enumerate(self._studies):
            members = self._members[study]
            merged = sum(len(ids) > 1 for ids in members.values())
            n_samples = self._cubes[study].n_samples
            text = (
                f"  {study}: {n_samples:,} samples, {len(members):,} groups"
                + (f" ({merged} merged)" if merged else "")
                + f", {int(self._measured[position].sum()):,} features measured"
            )
            lines.append(text)
        if self.n_studies > 1:
            shared_groups = sum(
                all(group in self._members[study] for study in self._studies)
                for group in self._groups["group_id"].tolist()
            )
            shared_features = int(self._measured.all(axis=0).sum())
            lines.append(
                f"  in every study: {shared_groups:,} groups, {shared_features:,} "
                "features"
            )
        return fitted_lines("\n".join(lines))

    # Each study on the collection's axes.

    def aligned_cube(self, study: str) -> Cube:
        """Return a study's cube on the collection's axes.

        The cube has the study's groups under their aligned ids, merged where several
        share one, and the features it measured, both in the collection's order, and
        the study's own samples with their own ids. It is derived from the cube as
        given with ``Cube.select`` and ``Cube.regroup``, whose records its provenance
        holds; a cube that needs neither is returned as it is. A lazy cube, such as
        one read with ``read_cube``, stays lazy; a cube in memory is copied. Integer
        layers keep their dtype, such as uint32 counts, unless groups of the study are
        merged, which sums them in int64.

        Args:
            study: A study name.

        Raises:
            KeyError: If there is no such study.
            TypeError: If ``study`` is not a string.
            OverflowError: If a merged value of a cube in memory does not fit (see
                "Sums" in ``Cube``).
        """
        position = self._study_position(study)
        members = self._members[study]
        groups = [group for ids in members.values() for group in ids]
        features = [
            feature
            for feature, measured in zip(
                self._features["feature_id"].tolist(),
                self._measured[position].tolist(),
                strict=True,
            )
            if measured
        ]
        return _on_axes(self._cubes[study], members, groups, features)

    # Deriving collections and cubes.

    def select(
        self, studies: Any = None, groups: Any = None, features: Any = None
    ) -> "CubeCollection":
        """Return a collection of some studies, groups and features.

        Names, ids and positions also set the order of the result; a single one
        selects one entry. Without ``features``, the result keeps the features that
        one of its studies measured, and without ``groups`` the groups that one of
        them has, in the order they had. The tables of samples, groups and features
        are those of the studies kept, as a collection of them would have them. The
        provenance records the selection.

        Args:
            studies: Study names, a boolean mask or positions; ``None`` keeps every
                study.
            groups: Aligned group ids, a boolean mask or positions; ``None`` keeps
                the groups of the studies kept.
            features: Feature ids, a boolean mask or positions; ``None`` keeps the
                features the studies kept measured.

        Raises:
            KeyError: If a name or id is unknown.
            IndexError: If a position is out of range.
            TypeError: If a selection is not names or ids, a boolean mask or
                integers, or is a numpy masked array.
            ValueError: If a mask has the wrong length, a selection repeats an entry
                or selects no study, or a group or feature selected is in none of the
                studies kept.
        """
        study_positions = selected_positions(
            studies,
            text_column(self._studies),
            self._study_index,
            "studies",
            "study",
            named=True,
        )
        group_positions = selected_positions(
            groups, self._groups["group_id"], self._group_index, "groups"
        )
        feature_positions = selected_positions(
            features, self._features["feature_id"], self._feature_index, "features"
        )
        selections = (study_positions, group_positions, feature_positions)
        if all(positions is None for positions in selections):
            return self
        if study_positions is None:
            study_positions = np.arange(self.n_studies)
        if study_positions.size == 0:
            raise ValueError("the selection of studies is empty; select at least one")
        kept = [self._studies[position] for position in study_positions.tolist()]
        measured = self._measured[study_positions]
        group_ids = self._groups["group_id"].tolist()
        has = self._has_groups(kept)
        group_positions = _kept_positions(group_positions, has, group_ids, "group")
        feature_ids = self._features["feature_id"].tolist()
        feature_positions = _kept_positions(
            feature_positions, measured, feature_ids, "feature"
        )
        kept_groups = [group_ids[position] for position in group_positions.tolist()]
        kept_features = [
            feature_ids[position] for position in feature_positions.tolist()
        ]
        record = {
            "operation": "select",
            "studies": None if studies is None else kept,
            "groups": None if groups is None else kept_groups,
            "features": None if features is None else kept_features,
        }
        collection = object.__new__(CubeCollection)
        _assemble(
            collection,
            tuple(kept),
            {study: self._cubes[study] for study in kept},
            {
                study: {
                    group: self._members[study][group]
                    for group in kept_groups
                    if group in self._members[study]
                }
                for study in kept
            },
            kept_groups,
            kept_features,
            measured[:, feature_positions],
            self._settings,
            (*self._operations, json.dumps(record)),
        )
        return collection

    def stack(
        self, *, groups: Any = None, features: Any = None, layers: Any = None
    ) -> Cube:
        """Return the studies stacked into one cube, for a joint model.

        The cube's samples are those of every study, named ``study::sample_id``,
        with ``samples`` as its sample table, so that a design can take the study
        as a covariate. Its groups are aligned groups that every study has, and its
        features are features that every study measured, in the collection's order:
        a cube holds no missing values, so a group or feature that a study lacks is
        refused rather than filled with zeros. Each study's part is its aligned
        cube's (``aligned_cube``): its own counts and library sizes, with merged
        groups summed. The cube is in memory; of a lazy study cube, such as one read
        with ``read_cube``, only the groups and features stacked are read.

        Layers keep their type when every study gives them the same; integer layers
        of different types become int64, and integer counts become float64 when
        another study's counts are float64. Cell numbers are stacked when every
        study records them. The normalization is that of the studies, which must be
        the same for a normalized layer to be stacked, and None for counts and
        detected cells alone.

        The provenance records, for every study, its cube's fingerprint and
        provenance and the groups stacked from it under their aligned ids, with the
        settings of the alignment and the layers (``cytepool.cube.provenance``).

        Args:
            groups: Aligned group ids, a boolean mask or positions; by default the
                groups every study has.
            features: Feature ids, a boolean mask or positions; by default the
                features every study measured.
            layers: Layer names, ``counts`` among them; by default the layers every
                study has.

        Returns:
            A cube with every study's samples, in the order of the studies and of
            each study's samples.

        Raises:
            KeyError: If an id is unknown.
            IndexError: If a position is out of range.
            TypeError: If a selection is not ids, a boolean mask or integers, or is
                or holds a numpy masked array.
            ValueError: If a group or feature is not in every study, no group or
                feature is in every study, a layer is unknown, not in every study or
                not given with ``counts``, some studies record cell numbers and
                others do not, the studies' normalizations differ for a normalized
                layer, or integer counts of 2**53 or more would become float64.
            OverflowError: If a merged value does not fit (see "Sums" in ``Cube``).
        """
        cubes = [self._cubes[study] for study in self._studies]
        group_ids = self._groups["group_id"].tolist()
        has = self._has_groups(self._studies)
        chosen_groups = _shared_positions(
            groups, self._groups["group_id"], self._group_index, has, self, "group"
        )
        chosen_features = _shared_positions(
            features,
            self._features["feature_id"],
            self._feature_index,
            self._measured,
            self,
            "feature",
        )
        names = _stacked_layers(layers, self._studies, cubes)
        with_cells = [cube.n_cells is not None for cube in cubes]
        if any(with_cells) and not all(with_cells):
            have = [
                s for s, cells in zip(self._studies, with_cells, strict=True) if cells
            ]
            lack = [s for s in self._studies if s not in have]
            raise ValueError(
                f"the cubes of {quoted(have)} record the cells of every sample and "
                f"group and those of {quoted(lack)} do not: a cube records cell "
                "numbers for all its samples or for none, so these studies are not "
                "stacked together; select studies alike (select(studies=...))"
            )
        normalization = _shared_normalization(names, self._studies, cubes)
        stacked_groups = [group_ids[position] for position in chosen_groups.tolist()]
        feature_ids = self._features["feature_id"]
        stacked_features = [feature_ids[p] for p in chosen_features.tolist()]
        parts = [
            _on_axes(
                self._cubes[study],
                {group: self._members[study][group] for group in stacked_groups},
                [
                    member
                    for group in stacked_groups
                    for member in self._members[study][group]
                ],
                stacked_features,
            )
            for study in self._studies
        ]
        stacked = {name: _stacked_layer(name, parts, self._studies) for name in names}
        n_cells = None
        if all(with_cells):
            n_cells = np.concatenate(
                [part.n_cells for part in parts if part.n_cells is not None]
            )
        settings = json.loads(self._settings)
        construction = {
            "function": "CubeCollection.stack",
            "cytepool_version": __version__,
            "studies": {
                study: {
                    "fingerprint": self._cubes[study].fingerprint(),
                    "provenance": self._cubes[study].provenance,
                    "groups": {
                        group: list(self._members[study][group])
                        for group in stacked_groups
                    },
                }
                for study in self._studies
            },
            "align_groups_by": settings["align_groups_by"],
            **(
                {"group_names": settings["group_names"]}
                if "group_names" in settings
                else {}
            ),
            "within_study": settings["within_study"],
            "layers": names,
        }
        return Cube(
            stacked,
            samples=self._samples,
            groups=self._groups.take(chosen_groups),
            features=self._features.take(chosen_features),
            n_cells=n_cells,
            normalization=normalization,
            provenance={"construction": construction},
            copy=False,
        )

    # Internals.

    def _has_groups(self, studies: Sequence[str]) -> np.ndarray:
        """Whether each of these studies has each aligned group, as booleans of
        shape ``(len(studies), n_groups)``."""
        group_ids = self._groups["group_id"].tolist()
        has = [
            [group in self._members[study] for group in group_ids] for study in studies
        ]
        return np.array(has, dtype=bool).reshape(len(studies), len(group_ids))

    def _study_position(self, study: object) -> int:
        reject_masked(study, "study")
        if not isinstance(study, str):
            raise TypeError(
                f"study must be a study name (str), got {type(study).__name__}"
            )
        position = self._study_index.get(study)
        if position is None:
            raise KeyError(
                f"no study named {study!r}; studies are: {quoted(list(self._studies))}"
                + close_matches(study, self._studies)
            )
        return position


def _assemble(
    collection: CubeCollection,
    studies: tuple[str, ...],
    cubes: dict[str, Cube],
    members: dict[str, dict[str, tuple[str, ...]]],
    group_ids: list[str],
    feature_ids: list[str],
    measured: np.ndarray,
    settings: str,
    operations: tuple[str, ...],
) -> None:
    """Build a collection's tables from its aligned parts and set its state."""
    alignment = json.loads(settings)
    within = alignment["within_study"]
    measured = np.array(measured, dtype=bool)  # a copy the collection owns
    measured.flags.writeable = False
    # Each study's aligned groups in the collection's order.
    members = {
        study: {
            group: members[study][group]
            for group in group_ids
            if group in members[study]
        }
        for study in studies
    }
    _set_state(
        collection,
        studies,
        cubes,
        members,
        measured,
        _sample_table(studies, cubes, within),
        _named_groups(
            _group_table(studies, cubes, members, group_ids),
            alignment["align_groups_by"],
            alignment.get("group_names"),
        ),
        _feature_table(cubes, feature_ids),
        settings,
        operations,
    )


def _set_state(
    collection: CubeCollection,
    studies: tuple[str, ...],
    cubes: dict[str, Cube],
    members: dict[str, dict[str, tuple[str, ...]]],
    measured: np.ndarray,
    samples: Table,
    groups: Table,
    features: Table,
    settings: str,
    operations: tuple[str, ...],
) -> None:
    state = {
        "_studies": studies,
        "_cubes": cubes,
        "_members": members,
        "_measured": measured,
        "_samples": samples,
        "_groups": groups,
        "_features": features,
        "_settings": settings,
        "_operations": operations,
    }
    for axis, ids in (
        ("study", list(studies)),
        ("group", groups["group_id"].tolist()),
        ("feature", features["feature_id"].tolist()),
    ):
        state[f"_{axis}_index"] = {key: position for position, key in enumerate(ids)}
    for name, value in state.items():
        object.__setattr__(collection, name, value)


def _restore_collection(
    studies: tuple[str, ...],
    cubes: dict[str, Cube],
    members: dict[str, dict[str, tuple[str, ...]]],
    measured: np.ndarray,
    samples: Table,
    groups: Table,
    features: Table,
    settings: str,
    operations: tuple[str, ...],
) -> CubeCollection:
    collection = object.__new__(CubeCollection)
    measured.flags.writeable = False  # unpickled arrays are new and writable
    _set_state(
        collection,
        studies,
        cubes,
        members,
        measured,
        samples,
        groups,
        features,
        settings,
        operations,
    )
    return collection


# Checking the arguments.


def _checked_cubes(cubes: Any) -> tuple[str, ...]:
    if not isinstance(cubes, Mapping):
        raise TypeError(
            "cubes must be a mapping of study names to cubes, such as "
            f"{{'cohort_a': cube_a, 'cohort_b': cube_b}}, got {type(cubes).__name__}"
        )
    if not cubes:
        raise ValueError("cubes is empty; a collection needs at least one study")
    for study, cube in cubes.items():
        if not isinstance(study, str):
            raise TypeError(
                f"study names must be strings, got {type(study).__name__} {study!r}"
            )
        if not study:
            raise ValueError("study names must be non-empty")
        if _SEPARATOR in study:
            raise ValueError(
                f"the study name {study!r} holds {_SEPARATOR!r}, which separates "
                f"a study from a sample id in 'study{_SEPARATOR}sample_id'; "
                "rename the study"
            )
        if not isinstance(cube, Cube):
            raise TypeError(
                f"cubes[{study!r}] must be a Cube, got {type(cube).__name__}"
            )
    return tuple(cubes)


def _checked_within_study(value: Any, cubes: dict[str, Cube]) -> list[str]:
    if isinstance(value, str):
        names: list[Any] = [value]
    else:
        reject_masked(value, "within_study")
        try:
            names = list(value)
        except TypeError:
            raise TypeError(
                "within_study must be a sample column name or a list of them, got "
                f"{type(value).__name__}"
            ) from None
    columns = list(
        dict.fromkeys(name for cube in cubes.values() for name in cube.samples.columns)
    )
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"within_study must name sample columns (str), got "
                f"{type(name).__name__} {name!r}"
            )
        if name in ("sample_id", "study"):
            raise ValueError(
                f"within_study names {name!r}, which the collection writes itself; "
                "name other sample columns, such as donor ids"
            )
        if names.count(name) > 1:
            raise ValueError(f"within_study names {name!r} more than once")
        if name not in columns:
            raise KeyError(
                f"within_study names {name!r}, which is not a sample column of any "
                f"study; their columns are: {listing(columns)}"
                + close_matches(name, columns)
            )
        for study, cube in cubes.items():
            if name in cube.samples:
                column = cube.samples[name]
                if column.size and column.dtype.kind not in "Oiu":
                    raise TypeError(
                        f"within_study names {name!r}, whose values in study "
                        f"{study!r} are {column.dtype}; such columns hold ids, as text "
                        "or integers"
                    )
    return names


def _checked_group_mapping(
    value: Any, cubes: dict[str, Cube]
) -> dict[str, dict[str, str | None]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "group_mapping must map study names to mappings of group ids to aligned "
            f"ids, got {type(value).__name__}"
        )
    reject_masked(value, "group_mapping")
    checked: dict[str, dict[str, str | None]] = {}
    for study, groups in value.items():
        if not isinstance(study, str):
            raise TypeError(
                f"group_mapping must be keyed by study names (str), got "
                f"{type(study).__name__} {study!r}"
            )
        if study not in cubes:
            raise KeyError(
                f"group_mapping names the study {study!r}, which is not in cubes; "
                f"studies are: {quoted(list(cubes))}"
                + close_matches(study, list(cubes))
            )
        if not isinstance(groups, Mapping):
            raise TypeError(
                f"group_mapping[{study!r}] must map the study's group ids to aligned "
                f"ids, got {type(groups).__name__}"
            )
        ids = cubes[study].groups["group_id"].tolist()
        known = set(ids)
        checked[study] = {}
        for group, aligned in groups.items():
            if not isinstance(group, str):
                raise TypeError(
                    f"group_mapping[{study!r}] must be keyed by group ids (str), got "
                    f"{type(group).__name__} {group!r}"
                )
            if group not in known:
                raise KeyError(
                    f"group_mapping[{study!r}] names the group {group!r}, which the "
                    f"cube of study {study!r} does not have; its groups are: "
                    f"{quoted(ids)}" + close_matches(group, ids)
                )
            if aligned is not None and not isinstance(aligned, str):
                raise TypeError(
                    f"group_mapping[{study!r}][{group!r}] must be an aligned id (str), "
                    f"or None to leave the group out; got {type(aligned).__name__}"
                )
            if aligned == "":
                raise ValueError(
                    f"group_mapping[{study!r}][{group!r}] is empty; aligned ids are "
                    "non-empty (None leaves the group out)"
                )
            checked[study][group] = aligned
    return checked


def _checked_group_names(
    value: Any, members: dict[str, dict[str, list[str]]]
) -> dict[str, str]:
    """The names of the aligned groups, by aligned id; empty for None."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "group_names must map aligned group ids to names, such as "
            f"{{'CL:0001054': 'CD14 monocytes'}}, got {type(value).__name__}"
        )
    reject_masked(value, "group_names")
    aligned = list(dict.fromkeys(a for groups in members.values() for a in groups))
    names: dict[str, str] = {}
    for key, name in value.items():
        if not isinstance(key, str):
            raise TypeError(
                f"group_names must be keyed by aligned group ids (str), got "
                f"{type(key).__name__} {key!r}"
            )
        if key not in aligned:
            raise KeyError(
                f"group_names names the aligned group {key!r}, which no study has; the "
                f"aligned groups are: {quoted(aligned)}" + close_matches(key, aligned)
            )
        if not isinstance(name, str):
            raise TypeError(
                f"group_names[{key!r}] must be a name (str), got {type(name).__name__}"
            )
        if not name:
            raise ValueError(f"group_names[{key!r}] is empty; names are non-empty")
        names[key] = name
    taken: dict[str, str] = {}
    for key in aligned:
        name = names.get(key, key)
        if name in taken:
            raise ValueError(
                f"the aligned groups {taken[name]!r} and {key!r} would both be named "
                f"{name!r}; give each group a name of its own, and none the id of "
                "another group"
            )
        taken[name] = key
    return names


def _named_groups(table: Table, by: str, names: dict[str, str] | None) -> Table:
    """A collection's groups table whose aligned column holds each group's aligned
    id, when its groups are named; else the table as it is."""
    if not names or by == "group_id":
        return table
    aligned = {name: key for key, name in names.items()}
    ids = [aligned.get(group, group) for group in table["group_id"].tolist()]
    return table.with_columns({by: text_column(ids)})


def _alignment_column(value: Any, cubes: dict[str, Cube]) -> str:
    if value is None:
        if all(_ONTOLOGY_COLUMN in cube.groups for cube in cubes.values()):
            return _ONTOLOGY_COLUMN
        return "group_id"
    if not isinstance(value, str):
        raise TypeError(
            f"align_groups_by must be a group column name, got {type(value).__name__}"
        )
    columns = list(
        dict.fromkeys(name for cube in cubes.values() for name in cube.groups.columns)
    )
    if value not in columns:
        raise KeyError(
            f"align_groups_by names {value!r}, which is not a group column of any "
            f"study; their columns are: {listing(columns)}"
            + close_matches(value, columns)
        )
    for study, cube in cubes.items():
        if value in cube.groups:
            column = cube.groups[value]
            if column.size and column.dtype.kind != "O":
                raise TypeError(
                    f"align_groups_by names {value!r}, which holds {column.dtype} in "
                    f"study {study!r}; groups are aligned by the values of a text "
                    "column"
                )
    return value


# Aligning.


def _aligned_groups(
    study: str, cube: Cube, by: str, mapping: dict[str, str | None]
) -> dict[str, list[str]]:
    """The study's groups by aligned id, in the order of their first member."""
    ids = cube.groups["group_id"].tolist()
    values = cube.groups[by].tolist() if by in cube.groups else None
    aligned: dict[str, list[str]] = {}
    for position, group in enumerate(ids):
        if group in mapping:
            key = mapping[group]
            if key is None:
                continue
        elif values is None:
            raise ValueError(
                f"the groups of study {study!r} are aligned by {by!r}, which its group "
                f"table does not have; give the aligned ids of its groups in "
                f"group_mapping[{study!r}], or align the groups by a column every "
                "study has (align_groups_by)"
            )
        else:
            key = values[position]
            if key is None or key == "":
                raise ValueError(
                    f"group {group!r} of study {study!r} has no {by}, by which the "
                    f"groups are aligned; give its aligned id in "
                    f"group_mapping[{study!r}], or None to leave it out"
                )
        aligned.setdefault(key, []).append(group)
    return aligned


def _measured_features(cubes: dict[str, Cube]) -> tuple[list[str], np.ndarray]:
    """The features that a study measured, in order of first appearance, and
    whether each study measured each."""
    index: dict[str, int] = {}
    measured_ids = []
    for cube in cubes.values():
        ids = cube.features["feature_id"].tolist()
        if "measured" in cube.features:
            flags = cube.features["measured"].tolist()
            ids = [feature for feature, flag in zip(ids, flags, strict=True) if flag]
        measured_ids.append(ids)
        for feature in ids:
            index.setdefault(feature, len(index))
    measured = np.zeros((len(cubes), len(index)), dtype=bool)
    for row, ids in enumerate(measured_ids):
        measured[row, [index[feature] for feature in ids]] = True
    return list(index), measured


def _check_shared_features(
    studies: tuple[str, ...], feature_ids: list[str], measured: np.ndarray
) -> None:
    counts = measured.sum(axis=0)
    for row, study in enumerate(studies):
        if (measured[row] & (counts > 1)).any():
            continue
        own = [f for f, flag in zip(feature_ids, measured[row], strict=True) if flag]
        elsewhere = np.delete(measured, row, axis=0).any(axis=0)
        others = [f for f, flag in zip(feature_ids, elsewhere, strict=True) if flag]
        hint = ""
        if {_VERSION.sub("", f) for f in own} & {_VERSION.sub("", f) for f in others}:
            hint = (
                " Some ids differ only in a version, such as '.17' in "
                "'ENSG00000141510.17': remove the versions from the feature ids first."
            )
        raise ValueError(
            f"study {study!r} shares no feature with the other studies: none of the "
            f"{len(own):,} features it measured, such as {quoted(own[:3])}, was "
            f"measured by another study, which measured features such as "
            f"{quoted(others[:3])}. Features are aligned by feature_id, which must "
            "hold ids of one kind in every study, such as Ensembl gene ids." + hint
        )


def _check_shared_groups(
    studies: tuple[str, ...], members: dict[str, dict[str, list[str]]], by: str
) -> None:
    for study in studies:
        others = {
            group for other in studies if other != study for group in members[other]
        }
        if set(members[study]) & others:
            continue
        raise ValueError(
            f"study {study!r} shares no group with the other studies: aligned by "
            f"{by!r}, its groups are {quoted(list(members[study]))}, and theirs "
            f"{quoted(sorted(others))}. Align the groups by another column "
            "(align_groups_by), or give their aligned ids in group_mapping"
        )


def _on_axes(
    cube: Cube,
    members: Mapping[str, Sequence[str]],
    groups: list[str],
    features: list[str],
) -> Cube:
    """A study's cube with the groups ``groups``, merged and renamed by ``members``
    (aligned id to the groups it sums, in the order of ``groups``), and the features
    ``features``; selecting an axis only where it changes."""
    selected = cube.select(
        groups=None if groups == cube.groups["group_id"].tolist() else groups,
        features=None if features == cube.features["feature_id"].tolist() else features,
    )
    renamed = {
        aligned: list(ids) for aligned, ids in members.items() if list(ids) != [aligned]
    }
    return selected.regroup(renamed) if renamed else selected


# The tables of a collection.


def _sample_table(
    studies: tuple[str, ...], cubes: dict[str, Cube], within: list[str]
) -> Table:
    ids: list[str] = []
    labels: list[str] = []
    for study in studies:
        sample_ids = cubes[study].samples["sample_id"].tolist()
        ids += [f"{study}{_SEPARATOR}{sample}" for sample in sample_ids]
        labels += [study] * len(sample_ids)
        if "study" in cubes[study].samples:
            found = cubes[study].samples["study"].tolist()
            if any(value != study for value in found):
                other = next(value for value in found if value != study)
                raise ValueError(
                    f"the cube of study {study!r} has a sample column 'study' with the "
                    f"value {other!r}, where the collection writes the study's name; "
                    "rename that column"
                )
    names = list(
        dict.fromkeys(
            name
            for study in studies
            for name in cubes[study].samples.columns
            if name not in ("sample_id", "study")
        )
    )
    columns: dict[str, np.ndarray] = {
        "sample_id": text_column(ids),
        "study": text_column(labels),
    }
    for name in names:
        parts = []
        for study in studies:
            table = cubes[study].samples
            column = table[name] if name in table else None
            if column is not None and name in within:
                column = text_column(
                    [
                        None if value is None else f"{study}{_SEPARATOR}{value}"
                        for value in column.tolist()
                    ]
                )
            parts.append((study, column, len(table)))
        columns[name] = _stacked_column(name, "sample", parts)
    return Table(columns)


def _group_table(
    studies: tuple[str, ...],
    cubes: dict[str, Cube],
    members: dict[str, dict[str, tuple[str, ...]]],
    group_ids: list[str],
) -> Table:
    names = list(
        dict.fromkeys(
            name
            for study in studies
            for name in cubes[study].groups.columns
            if name != "group_id"
        )
    )
    columns: dict[str, np.ndarray] = {"group_id": text_column(group_ids)}
    # The rows of each study's group table that each aligned group sums.
    rows = {
        study: {
            group: _positions_of(cubes[study].groups["group_id"], ids)
            for group, ids in members[study].items()
        }
        for study in studies
    }
    for name in names:
        present = [
            (study, cubes[study].groups[name])
            for study in studies
            if name in cubes[study].groups
        ]
        dtype = _common_dtype(name, "group", present)
        values: list[Any] = []
        for group in group_ids:
            found = [
                column[rows[study][group]].astype(dtype)
                for study, column in present
                if group in rows[study]
            ]
            if not found:
                values.append(_MISSING)
                continue
            agreed = np.concatenate(found)
            same = values_equal(agreed, np.repeat(agreed[:1], agreed.size)).all()
            values.append(agreed[0] if same else _MISSING)
        columns[name] = _column_from_values(values, dtype)
    return Table(columns)


def _feature_table(cubes: dict[str, Cube], feature_ids: list[str]) -> Table:
    columns = {"feature_id": text_column(feature_ids)}
    if any("symbol" in cube.features for cube in cubes.values()):
        index = {feature: position for position, feature in enumerate(feature_ids)}
        symbols: list[str | None] = [None] * len(feature_ids)
        for cube in cubes.values():
            if "symbol" not in cube.features:
                continue
            for feature, symbol in zip(
                cube.features["feature_id"].tolist(),
                cube.features["symbol"].tolist(),
                strict=True,
            ):
                position = index.get(feature)
                if position is not None and symbols[position] is None:
                    symbols[position] = symbol
        columns["symbol"] = text_column(symbols)
    return Table(columns)


def _positions_of(ids: np.ndarray, chosen: Sequence[str]) -> np.ndarray:
    where = {identifier: position for position, identifier in enumerate(ids.tolist())}
    return np.array([where[identifier] for identifier in chosen], dtype=np.intp)


class _Missing:
    """A missing value of a column that is built value by value."""


_MISSING = _Missing()


def _common_dtype(
    name: str, axis: str, present: list[tuple[str, np.ndarray]]
) -> np.dtype:
    """The dtype that holds a table column of every study that has it.

    Columns without rows do not count. Raises TypeError for columns whose kinds of
    values differ, such as text in one study and numbers in another.
    """
    filled = [(study, column) for study, column in present if column.size]
    if not filled:
        return present[0][1].dtype if present else np.dtype(object)
    dtypes = [column.dtype for _, column in filled]
    if all(dtype == dtypes[0] for dtype in dtypes):
        return dtypes[0]
    kinds = {_KINDS[dtype.kind] for dtype in dtypes}
    if kinds == {"integers"}:
        for study, column in filled:
            if column.dtype == np.uint64 and column.max() > INT64_MAX:
                raise TypeError(
                    f"the {axis} column {name!r} of study {study!r} holds integers "
                    "above 2**63 - 1, which int64 does not hold, and other studies "
                    "hold signed integers in it"
                )
        return np.dtype(np.int64)
    if kinds <= {"integers", "floats"}:
        return np.dtype(np.float64)
    if kinds <= _NUMBERS:
        return np.dtype(np.complex128)
    if kinds in ({"datetimes"}, {"timedeltas"}):
        return np.result_type(*dtypes)
    first_study, first = filled[0]
    other_study, other = next(
        (study, column)
        for study, column in filled
        if _KINDS[column.dtype.kind] != _KINDS[first.dtype.kind]
    )
    raise TypeError(
        f"the {axis} column {name!r} holds {_KINDS[first.dtype.kind]} "
        f"({first.dtype}) in study {first_study!r} and {_KINDS[other.dtype.kind]} "
        f"({other.dtype}) in study {other_study!r}; a column must hold one kind of "
        "values in every study: convert it in one of the cubes' tables, or rename it "
        "in one of them"
    )


def _with_missing_dtype(dtype: np.dtype) -> np.dtype:
    """The dtype of a column that needs a missing value: integers and booleans
    become float64 with NaN."""
    return np.dtype(np.float64) if dtype.kind in "biu" else dtype


def _missing_value(dtype: np.dtype) -> Any:
    if dtype.kind == "O":
        return None
    if dtype.kind in "mM":
        return np.array("NaT", dtype=dtype)
    return np.nan


def _stacked_column(
    name: str, axis: str, parts: list[tuple[str, np.ndarray | None, int]]
) -> np.ndarray:
    """A table column of every study, one after another; missing values where a
    study does not have it."""
    present = [(study, column) for study, column, _ in parts if column is not None]
    dtype = _common_dtype(name, axis, present)
    if any(column is None and rows for _, column, rows in parts):
        dtype = _with_missing_dtype(dtype)
    result = np.empty(sum(rows for _, _, rows in parts), dtype=dtype)
    start = 0
    for _, column, rows in parts:
        if column is None:
            result[start : start + rows] = _missing_value(dtype)
        elif rows:
            result[start : start + rows] = column.astype(dtype)
        start += rows
    return result


def _column_from_values(values: list[Any], dtype: np.dtype) -> np.ndarray:
    """A column of the values given, missing where a value is ``_MISSING``."""
    missing = [value is _MISSING for value in values]
    if any(missing):
        dtype = _with_missing_dtype(dtype)
    result = np.empty(len(values), dtype=dtype)
    for position, value in enumerate(values):
        result[position] = _missing_value(dtype) if missing[position] else value
    return result


# Stacking.


def _kept_positions(
    positions: np.ndarray | None, present: np.ndarray, ids: list[str], single: str
) -> np.ndarray:
    """The groups or features a selection keeps: those given, each of which a kept
    study must have, or by default those a kept study has, in order."""
    if positions is None:
        return np.flatnonzero(present.any(axis=0))
    lacking = [ids[p] for p in positions.tolist() if not present[:, p].any()]
    if lacking:
        word = "measured" if single == "feature" else "has"
        raise ValueError(
            f"no study selected {word} the {single}"
            f"{'' if len(lacking) == 1 else 's'} {quoted(lacking)}; select studies "
            f"that do, or other {single}s"
        )
    return positions


def _shared_positions(
    selector: Any,
    ids: np.ndarray,
    index: dict[str, int],
    present: np.ndarray,
    collection: CubeCollection,
    single: str,
) -> np.ndarray:
    """The groups or features to stack: those given, each of which every study must
    have, or by default those every study has, in the collection's order."""
    word = "measured" if single == "feature" else "has"
    positions = selected_positions(selector, ids, index, f"{single}s")
    if positions is None:
        shared = np.flatnonzero(present.all(axis=0))
        if shared.size == 0:
            raise ValueError(
                f"no {single} is in every study, so there is nothing to stack: a cube "
                f"holds no missing values, so stack() takes only the {single}s that "
                f"every study {word}; select studies that share {single}s "
                "(select(studies=...))"
            )
        return shared
    for position in positions.tolist():
        lacking = [
            study
            for row, study in enumerate(collection.studies)
            if not present[row, position]
        ]
        if lacking:
            raise ValueError(
                f"the {single} {ids[position]!r} is not in "
                f"{'study' if len(lacking) == 1 else 'studies'} {quoted(lacking)}: a "
                f"cube holds no missing values, so stack() takes only {single}s that "
                f"every study {word}, rather than fill the others with zeros; leave "
                f"out the {single} or those studies (select(studies=...))"
            )
    return positions


def _stacked_layers(
    value: Any, studies: tuple[str, ...], cubes: list[Cube]
) -> list[str]:
    if value is None:
        return [
            name
            for name in LAYER_NAMES
            if all(name in cube.layer_names for cube in cubes)
        ]
    reject_masked(value, "layers")
    names: list[Any] = [value] if isinstance(value, str) else list(value)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"layers must be layer names (str), got {type(name).__name__} {name!r}"
            )
        if name not in LAYER_NAMES:
            raise ValueError(
                f"unknown layer {name!r}; layers are named: {', '.join(LAYER_NAMES)}"
            )
        if names.count(name) > 1:
            raise ValueError(f"layers names {name!r} more than once")
        lacking = [
            study
            for study, cube in zip(studies, cubes, strict=True)
            if name not in cube.layer_names
        ]
        if lacking:
            shared = [
                layer
                for layer in LAYER_NAMES
                if all(layer in cube.layer_names for cube in cubes)
            ]
            raise ValueError(
                f"the cube of {'study' if len(lacking) == 1 else 'studies'} "
                f"{quoted(lacking)} has no {name!r} layer; every study has "
                f"{quoted(shared)}"
            )
    if "counts" not in names:
        raise ValueError("layers must include 'counts', which every cube holds")
    return [name for name in LAYER_NAMES if name in names]


def _shared_normalization(
    names: list[str], studies: tuple[str, ...], cubes: list[Cube]
) -> dict[str, Any] | None:
    normalized = [name for name in names if name in _NORMALIZED_LAYERS]
    if not normalized:
        return None
    first = cubes[0].normalization
    for study, cube in zip(studies[1:], cubes[1:], strict=True):
        if cube.normalization != first:
            raise ValueError(
                f"the normalized layers {quoted(normalized)} of study {study!r} were "
                f"computed with the normalization {cube.normalization!r}, and those of "
                f"study {studies[0]!r} with {first!r}: stacked, they would mix two "
                "normalizations. Stack counts and detected cells alone "
                "(layers=['counts', 'detected']), or studies normalized alike"
            )
    return first


def _stacked_layer(
    name: str, parts: list[Cube], studies: tuple[str, ...]
) -> np.ndarray:
    """A layer of every study's part, with the samples of each after another."""
    n_groups, n_features = parts[0].n_groups, parts[0].n_features
    n_samples = sum(part.n_samples for part in parts)
    dtype: np.dtype
    if name in _NORMALIZED_LAYERS:
        dtype = np.dtype(np.float64)
    else:
        dtypes = (
            [part.layer(name, 0).dtype for part in parts]
            if n_groups
            else [np.dtype(np.int64)]
        )
        if any(dtype.kind == "f" for dtype in dtypes):
            dtype = np.dtype(np.float64)
        elif all(dtype == dtypes[0] for dtype in dtypes):
            dtype = dtypes[0]
        else:
            dtype = np.dtype(np.int64)
    result = np.empty((n_groups, n_samples, n_features), dtype=dtype)
    start = 0
    for study, part in zip(studies, parts, strict=True):
        block = part.layer(name)
        if (
            dtype.kind == "f"
            and block.dtype.kind in "iu"
            and block.size
            and block.max() >= _FLOAT_LIMIT
        ):
            raise ValueError(
                f"the integer {name} of study {study!r} reach {block.max()}, and float64, "
                "which another study's counts need, holds whole numbers exactly only "
                "below 2**53"
            )
        result[:, start : start + part.n_samples] = block
        start += part.n_samples
        del block
    return result
