---
icon: lucide/download-cloud
---

# Installation

You can install `ome-writers` via pip. You *must* also select select at least
one backend extra:

```bash
pip install ome-writers[<backend>]
```

...where `<backend>` is a comma-separated list of one or more of the following:

- `tensorstore` — Uses [tensorstore](https://github.com/google/tensorstore), supports
  OME-Zarr v0.5.
- `acquire-zarr` — Uses
  [acquire-zarr](https://github.com/acquire-project/acquire-zarr), supports
  OME-Zarr v0.5.
- `zarr-python` — Uses [zarr-python](https://github.com/zarr-developers/zarr-python), supports
  OME-Zarr v0.5.
- `zarrs-python` — Uses [zarrs-python](https://github.com/zarrs/zarrs-python), supports
  OME-Zarr v0.5.
- `tifffile` — Uses [tifffile](https://github.com/cgohlke/tifffile), supports
  OME-TIFF.
- `all` — install all backends.

!!! Note
    All zarr-backends use [yaozarrs](https://github.com/tlambert03/yaozarrs) to generate
    OME-Zarr metadata and create zarr hierarchies (only array-writing is handled by the selected backend).
