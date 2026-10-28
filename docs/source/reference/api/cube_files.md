---
myst:
  html_meta:
    description: Save CytePool cubes as Zarr v3 files and open them lazily, reading only the chunks that hold the groups and features in use, with numpy and the standard library only.
---

# Cube files

Cube files, and the Zarr v3 reader and writer they use; [Cube files and provenance records](../reference/cube-files/#cube-file-format) specifies the format, and [Pseudobulk profiles and cubes](../statistics/cubes/#cubes-larger-than-memory) explains how a cube larger than memory is read.

```{eval-rst}
.. autofunction:: cytepool.write_cube

.. autofunction:: cytepool.read_cube
```

```{eval-rst}
.. automodule:: cytepool.io.zarr
```
