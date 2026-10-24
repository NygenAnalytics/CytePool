---
myst:
  html_meta:
    description: Save CytePool cubes as Zarr v3 files and open them lazily, one group at a time, with numpy and the standard library only.
---

# Cube files

A cube file is a Zarr v3 group that any Zarr v3 reader can open.
{func}`~cytepool.io.write_cube` saves a cube and {func}`~cytepool.io.read_cube` opens
one, leaving its layers on disk until a group is read. `cytepool.io.zarr` is the
minimal Zarr v3 reader and writer behind them.

```{eval-rst}
.. automodule:: cytepool.io
```

```{eval-rst}
.. automodule:: cytepool.io.zarr
```
