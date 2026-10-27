---
myst:
  html_meta:
    description: Save CytePool cubes as Zarr v3 files and open them lazily, reading only the chunks that hold the groups and features in use, with numpy and the standard library only.
---

# Cube files

A cube file is a Zarr v3 group that any Zarr v3 reader can open.
{func}`~cytepool.write_cube` saves a cube and {func}`~cytepool.read_cube` opens one,
leaving its layers on disk until they are used; a read then opens only the chunks that
hold the selected groups and features. `cytepool.io.zarr` is the minimal Zarr v3
reader and writer behind them.

```{eval-rst}
.. autofunction:: cytepool.write_cube

.. autofunction:: cytepool.read_cube
```

```{eval-rst}
.. automodule:: cytepool.io.zarr
```
