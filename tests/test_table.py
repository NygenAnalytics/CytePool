import copy
import pickle
import sys

import numpy as np
import pytest
from numpy.dtypes import StringDType
from numpy.testing import assert_array_equal

from cytepool import Table


@pytest.fixture(scope="module")
def table() -> Table:
    return Table(
        {
            "sample_id": ["S1", "S2", "S3"],
            "donor": ["D1", None, ""],
            "cells": np.array([120, 85, 3], dtype=np.int32),
            "age": [34.5, np.nan, 61.0],
            "treated": [True, False, True],
            "collected": np.array(["2024-01-02", "NaT", "2024-03-04"], dtype="M8[s]"),
        }
    )


def test_columns_keep_their_dtype_and_text_holds_str_or_none(table: Table) -> None:
    assert table.columns == (
        "sample_id",
        "donor",
        "cells",
        "age",
        "treated",
        "collected",
    )
    assert len(table) == 3
    assert table["cells"].dtype == np.int32
    assert table["age"].dtype == np.float64
    assert table["treated"].dtype == np.bool_
    assert table["collected"].dtype == np.dtype("M8[s]")
    assert table["donor"].dtype == object
    assert table["donor"].tolist() == ["D1", None, ""]
    assert [type(value) for value in table["sample_id"]] == [str, str, str]


def test_text_is_read_from_numpy_string_dtypes_and_object_arrays() -> None:
    table = Table(
        {
            "fixed": np.array(["x", "yy"]),
            "variable": np.array(["p", None], dtype=StringDType(na_object=None)),
            "objects": np.array(["q", np.nan], dtype=object),
            "listed": ["r", float("nan")],
        }
    )
    assert table["fixed"].tolist() == ["x", "yy"]
    assert table["variable"].tolist() == ["p", None]
    assert table["objects"].tolist() == ["q", None]
    assert table["listed"].tolist() == ["r", None]


def test_a_missing_value_is_not_an_empty_string(table: Table) -> None:
    assert (table["donor"] == "").tolist() == [False, False, True]
    assert (table["donor"] == "D1").tolist() == [True, False, False]
    assert not Table({"a": [""]}).equals(Table({"a": [None]}))


def test_numbers_and_booleans_with_none_become_float64_with_nan() -> None:
    table = Table({"numbers": [1, None, 3], "flags": [True, None, False]})
    assert table["numbers"].dtype == np.float64
    assert table["flags"].dtype == np.float64
    assert_array_equal(table["numbers"], [1.0, np.nan, 3.0])
    assert_array_equal(table["flags"], [1.0, np.nan, 0.0])


def test_pandas_missing_values_are_recognized_by_type_name() -> None:
    # pandas.NA and pandas.NaT are recognized without importing pandas; stand-ins
    # with the same type names check that rule where pandas is not installed.
    NAType = type("NAType", (), {})
    NaTType = type("NaTType", (), {})
    values = np.array([1, NAType(), 3, NaTType()], dtype=object)
    assert_array_equal(Table({"n": values})["n"], [1.0, np.nan, 3.0, np.nan])
    text = np.array(["a", NAType()], dtype=object)
    assert Table({"t": text})["t"].tolist() == ["a", None]


def test_columns_are_copied_and_read_only() -> None:
    source = np.array([1.0, 2.0])
    table = Table({"x": source})
    source[0] = 99.0
    assert table["x"].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="read-only"):
        table["x"][0] = 5.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        table["x"].flags.writeable = True
    exported = table.to_dict()["x"]
    with pytest.raises(ValueError, match="read-only"):
        exported[0] = 5.0


def test_table_attributes_cannot_be_set(table: Table) -> None:
    with pytest.raises(AttributeError, match="immutable"):
        table.columns = ("a",)  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del table._columns
    assert copy.copy(table) is table
    assert copy.deepcopy(table) is table


@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        ({"": [1]}, ValueError, "non-empty"),
        ({0: [1]}, TypeError, "column names must be str"),
        ({"a": [1, 2], "b": [1]}, ValueError, "equal lengths; 'a' has 2, 'b' has 1"),
        ({"a": [[1, 2], [3, 4]]}, ValueError, r"must be 1-D, got shape \(2, 2\)"),
        ({"a": [[1, 2], [3]]}, ValueError, "ragged"),
        ({"a": 5}, ValueError, r"must be 1-D, got shape \(\)"),
        ({"a": np.array([b"x"])}, TypeError, "holds bytes"),
        ({"a": ["x", 1]}, TypeError, "mixes values of types int, str"),
        ({"a": np.array([{"x": 1}], dtype=object)}, TypeError, "values of type dict"),
        ({"a": np.zeros(2, dtype=[("x", "f8")])}, TypeError, "unsupported dtype"),
        ([1, 2], TypeError, "mapping of column names"),
    ],
)
def test_invalid_columns_are_rejected(data: object, error: type, message: str) -> None:
    with pytest.raises(error, match=message):
        Table(data)


def test_empty_table() -> None:
    empty = Table()
    assert empty.columns == ()
    assert len(empty) == 0
    assert empty.equals(Table({}))
    assert repr(empty) == "Table with 0 rows and 0 columns"


def test_take_returns_rows_in_the_given_order(table: Table) -> None:
    taken = table.take([2, -3])
    assert taken["sample_id"].tolist() == ["S3", "S1"]
    assert taken["cells"].tolist() == [3, 120]
    assert taken["cells"].dtype == np.int32
    assert len(table.take([])) == 0
    with pytest.raises(IndexError, match="position 3 is out of range"):
        table.take([0, 3])
    with pytest.raises(IndexError, match="position -4 is out of range"):
        table.take([-4])
    with pytest.raises(TypeError, match="use filter"):
        table.take([True, False, True])
    with pytest.raises(TypeError, match="integer positions"):
        table.take([0.0])


def test_filter_keeps_rows_where_the_mask_is_true(table: Table) -> None:
    kept = table.filter(np.array([True, False, True]))
    assert kept["sample_id"].tolist() == ["S1", "S3"]
    assert len(kept) == 2
    with pytest.raises(TypeError, match="boolean mask"):
        table.filter([1, 0, 1])
    with pytest.raises(ValueError, match=r"one value per row \(3\)"):
        table.filter([True, False])


def test_filter_by_values_keeps_the_rows_a_mask_of_equalities_keeps(
    table: Table,
) -> None:
    # Each value against the mask built by hand; missing values match missing ones.
    cases = [
        ({"sample_id": "S2"}, table["sample_id"] == "S2"),
        ({"donor": None}, np.array([v is None for v in table["donor"]])),
        ({"donor": ""}, np.array([v == "" for v in table["donor"]])),
        ({"cells": 85}, table["cells"] == 85),
        ({"age": 61.0}, table["age"] == 61.0),
        ({"age": np.nan}, np.isnan(table["age"])),
        ({"cells": np.nan}, np.zeros(3, dtype=bool)),  # integers are never missing
        ({"treated": True}, table["treated"]),
        ({"collected": np.datetime64("2024-03-04")}, np.array([False, False, True])),
        ({"collected": np.datetime64("NaT", "s")}, np.array([False, True, False])),
        ({"treated": True, "cells": 3}, table["treated"] & (table["cells"] == 3)),
    ]
    for values, mask in cases:
        assert table.filter(**values).equals(table.filter(mask)), values
    # A mask and values together; the mask's keyword, as before values existed.
    both = table.filter(np.array([True, True, False]), treated=True)
    assert both["sample_id"].tolist() == ["S1"]
    assert table.filter(mask=np.array([False, True, True])).equals(
        table.filter(np.array([False, True, True]))
    )
    assert table.filter().equals(table)
    # A column named mask is filtered by its value.
    named = Table({"mask": [1.0, 2.0]})
    assert named.filter(mask=2.0)["mask"].tolist() == [2.0]


def test_filter_refuses_values_of_another_kind(table: Table) -> None:
    with pytest.raises(TypeError, match="the column holds numbers, a number, got str"):
        table.filter(cells="85")
    with pytest.raises(TypeError, match="holds numbers, a number, got NoneType None"):
        table.filter(age=None)  # NaN marks a missing number
    with pytest.raises(TypeError, match="the column holds text, a string, got int 1"):
        table.filter(donor=1)
    with pytest.raises(TypeError, match=r"numpy.isin\(table\['donor'\], values\)"):
        table.filter(donor=["D1", None])
    with pytest.raises(TypeError, match="booleans, True or False, got int"):
        table.filter(treated=1)
    with pytest.raises(TypeError, match="datetimes, a numpy.datetime64"):
        table.filter(collected="2024-03-04")
    with pytest.raises(KeyError, match="no column named 'tissue'"):
        table.filter(tissue="lung")


def test_sort_is_stable_and_puts_missing_values_last() -> None:
    rng = np.random.default_rng(3)
    values = rng.integers(0, 4, 40).astype(float)
    values[[3, 17, 30]] = np.nan
    text = np.array([f"t{v}" for v in rng.integers(0, 3, 40)], dtype=object)
    text[[5, 9]] = None
    table = Table({"row": np.arange(40), "value": values, "text": text})
    present = np.flatnonzero(~np.isnan(values))
    order = present[np.argsort(values[present], kind="stable")]
    expected = np.concatenate([order, np.flatnonzero(np.isnan(values))])
    assert_array_equal(table.sort("value")["row"], expected)
    # Descending: the largest first, equal values in their order, missing last.
    order = present[np.argsort(-values[present], kind="stable")]
    expected = np.concatenate([order, np.flatnonzero(np.isnan(values))])
    assert_array_equal(table.sort("value", descending=True)["row"], expected)
    named = [i for i in range(40) if text[i] is not None]
    by_text = sorted(named, key=lambda i: text[i]) + [5, 9]
    assert table.sort("text")["row"].tolist() == by_text
    by_text = sorted(named, key=lambda i: text[i], reverse=True) + [5, 9]
    assert table.sort("text", descending=True)["row"].tolist() == by_text
    assert table.sort("row", descending=True)["row"].tolist() == list(range(39, -1, -1))
    with pytest.raises(KeyError, match="no column named 'x'"):
        table.sort("x")
    with pytest.raises(TypeError, match="descending must be True or False"):
        table.sort("row", descending="yes")  # type: ignore[arg-type]
    assert Table({"a": []}).sort("a").equals(Table({"a": []}))


def test_select_chooses_and_orders_columns(table: Table) -> None:
    assert table.select(["age", "sample_id"]).columns == ("age", "sample_id")
    assert table.select("donor").columns == ("donor",)
    with pytest.raises(KeyError, match="no column named 'weight'"):
        table.select(["weight"])
    with pytest.raises(ValueError, match="more than once"):
        table.select(["age", "age"])


def test_with_columns_adds_and_replaces_without_changing_the_original(
    table: Table,
) -> None:
    changed = table.with_columns({"cell type": ["a", "b", "c"]}, age=[1.0, 2.0, 3.0])
    assert changed.columns == table.columns + ("cell type",)
    assert changed["age"].tolist() == [1.0, 2.0, 3.0]
    assert changed["cell type"].tolist() == ["a", "b", "c"]
    assert table["age"][0] == 34.5
    assert Table().with_columns(x=[1, 2]).columns == ("x",)
    with pytest.raises(ValueError, match="has 2 values; the table has 3 rows"):
        table.with_columns(extra=[1, 2])
    with pytest.raises(ValueError, match="given twice"):
        table.with_columns({"x": [1, 2, 3]}, x=[1, 2, 3])


def test_rename_keeps_positions(table: Table) -> None:
    renamed = table.rename({"age": "age_years"})
    assert renamed.columns[3] == "age_years"
    assert_array_equal(renamed["age_years"], table["age"])
    with pytest.raises(KeyError, match="no column named 'weight'"):
        table.rename({"weight": "mass"})
    with pytest.raises(ValueError, match="two columns named 'donor'"):
        table.rename({"age": "donor"})


def test_lookup_errors(table: Table) -> None:
    assert "age" in table and "weight" not in table
    assert list(table) == list(table.columns)
    with pytest.raises(
        KeyError, match="no column named 'weight'; columns are: sample_id"
    ):
        table["weight"]
    with pytest.raises(TypeError, match="looked up by name"):
        table[0]  # type: ignore[index]


def test_equals_compares_names_order_dtypes_and_values(table: Table) -> None:
    assert table.equals(Table(table.to_dict()))
    assert table.equals(table.take([0, 1, 2]))
    assert not table.equals(table.select(list(reversed(table.columns))))
    assert not table.equals(table.with_columns(cells=table["cells"].astype(np.int64)))
    assert not table.equals(table.with_columns(age=[34.5, 1.0, 61.0]))
    assert not table.equals(table.with_columns(donor=["D1", None, None]))
    assert not table.equals(table.to_dict())


def test_repr_shows_size_dtypes_and_the_first_and_last_rows() -> None:
    long = Table({"position": np.arange(30), "name": [f"x{i}" for i in range(30)]})
    text = repr(long)
    lines = text.splitlines()
    assert lines[0] == "Table with 30 rows and 2 columns"
    assert lines[1].split() == ["position", "name"]
    assert lines[2].split() == ["int64", "str"]
    assert lines[3].split() == ["0", "x0"]
    assert lines[8].split() == ["...", "..."]
    assert lines[-1].split() == ["29", "x29"]
    assert "x10" not in text


def test_repr_lists_columns_beyond_the_grid() -> None:
    wide = Table({f"c{i}": [i] for i in range(10)})
    assert repr(wide).splitlines()[-1] == "  2 more columns: c8 (int64), c9 (int64)"


def test_pickle_round_trip(table: Table) -> None:
    restored = pickle.loads(pickle.dumps(table))
    assert restored.equals(table)
    with pytest.raises(ValueError, match="read-only"):
        restored["age"][0] = 1.0


def test_to_pandas_explains_that_pandas_is_missing(
    table: Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match="needs pandas"):
        table.to_pandas()


def test_pandas_round_trip(table: Table) -> None:
    pd = pytest.importorskip("pandas")
    frame = table.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == list(table.columns)
    assert frame["cells"].dtype == np.int32
    assert frame["donor"].isna().tolist() == [False, True, False]
    assert frame["donor"].iloc[2] == ""
    assert Table(frame).equals(table)


def test_pandas_extension_dtypes_and_named_index() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["B", None, "B"]),
            "count": pd.array([1, None, 3], dtype="Int64"),
            "complete": pd.array([1, 2, 3], dtype="Int64"),
            "flag": pd.array([True, None, False], dtype="boolean"),
            "label": pd.array(["p", None, "q"], dtype="string"),
        },
        index=pd.Index(["S1", "S2", "S3"], name="sample_id"),
    )
    table = Table(frame)
    assert table.columns == (
        "sample_id",
        "cell_type",
        "count",
        "complete",
        "flag",
        "label",
    )
    assert table["sample_id"].tolist() == ["S1", "S2", "S3"]
    assert table["cell_type"].tolist() == ["B", None, "B"]
    assert_array_equal(table["count"], [1.0, np.nan, 3.0])
    assert table["complete"].dtype == np.int64
    assert_array_equal(table["flag"], [1.0, np.nan, 0.0])
    assert table["label"].tolist() == ["p", None, "q"]
    assert Table(frame.reset_index(drop=True)).columns[0] == "cell_type"
    # Series in a plain mapping reach numpy as object arrays holding pandas.NA.
    columns = Table({"count": frame["count"], "label": frame["label"]})
    assert_array_equal(columns["count"], [1.0, np.nan, 3.0])
    assert columns["label"].tolist() == ["p", None, "q"]
    with pytest.raises(ValueError, match="more than one column named 'a'"):
        Table(pd.DataFrame([[1, 2]], columns=["a", "a"]))
    clash = pd.DataFrame({"sample_id": ["x"]}, index=pd.Index(["S1"], name="sample_id"))
    with pytest.raises(ValueError, match="index 'sample_id' is also a column name"):
        Table(clash)
