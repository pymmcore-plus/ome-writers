---
title: Supported File Formats
icon: lucide/file-text
---

`ome-writers` currently supports writing the following file formats:

- [OME-TIFF](tiff.md) - A widely used file format for storing multi-dimensional
  biological image data.
- [OME-Zarr](zarr.md) - A next-generation file format for bioimaging data based
  on the Zarr specification.

Selecting and output formats is as simple as changing the target
[`AcquisitionSettings.root_path`][ome_writers.AcquisitionSettings] to end in
`.tiff` or `.zarr`. Or, explicitly select a `Acquisitionsettings.format`.

See the individual format pages linked above for more details on each format and
their specific options.
