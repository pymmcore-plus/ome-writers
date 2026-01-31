---
title: OME-Zarr
icon: lucide/boxes
---

!!! information "OME-Zarr Specification"

    We currently target `v0.5` of the official OME-Zarr specification as defined by:

    <https://ngff.openmicroscopy.org/0.5/index.html>

    As the specification evolves, we plan to add support for newer versions
    and additional features.

`ome-writers` supports writing to `OME-Zarr`: a newer file format for bioimaging
data based on the [Zarr
specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html),
with metadata structures defined by the [OME-Zarr (NGFF)
specification](https://ngff.openmicroscopy.org/0.5/index.html).

Zarr is a format designed for the storage of chunked, multi-dimensional arrays
and is optimized for cloud storage and parallel access. Data is comprised of
chunks, which are stored as individual files within a hierarchical directory
structure, (or, optionally grouped into super-chunks called "shards" in the v3
Zarr specification, with a single file per shard).

!!! tip "Further Reading"

    For more information on the Zarr format, see:

    - [Zarr Documentation "Concepts and terminology"](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#concepts-and-terminology)
    - [Xarray's Intro to Zarr](https://tutorial.xarray.dev/intermediate/intro-to-zarr.html)

    For a friendly guide to OME-Zarr, see:

    - [Yaozzars Guide to OME-Zarr](https://tlambert03.github.io/yaozarrs/ome_zarr_guide/)

## Expected Output Structure

When writing an acquisition to OME-Zarr, the expected output structure depends
on your [AcquisitionSettings][ome_writers.AcquisitionSettings].  OME-Zarr currently
supports no more than 5 dimensions (typically: `T`, `C`, `Z`, `Y`, `X`), per array
node.  Therefore, acquisitions with more than 5 dimensions will be split into
multiple arrays, each stored in a separate group within the root OME-Zarr group.

=== "Single ≤5D Image"

    Any acquisition with 5 or fewer dimensions will be stored as a single
    ["multiscales" image](https://ngff.openmicroscopy.org/0.5/index.html#multiscale-md)
    at the `AcquisitionSettings.output_path` with:

    ```
    output_path.ome.zarr/
    ├── zarr.json            # {"zarr_format": 3} group, with attributes.ome.multiscales
    └── 0/                   # Full resolution array  
        ├── zarr.json        # Array metadata (standard zarr schema)
        └── c/0/1/2/3        # Chunk files
    ```
    
    (We don't current write downsampled arrays, but this will be added in the future).

    !!! example "Example Code"
        [Writing a single ≤5D image](../examples/single_5d_image.md)

=== "Multi-Position & Other Collections"

    Acquisitions that contain multiple positions (e.g., stage positions, tiled
    images, angles on a light-sheet microscope, etc.) or that exceed 5
    dimensions in any way other will be structured as a root zarr group (currently
    following the transitional [`bioformats2raw`
    convention](https://ngff.openmicroscopy.org/0.5/index.html#bf2raw)),
    with a sub multiscales group for each position or collection member.

    ```
    output_path.ome.zarr/
    ├── zarr.json             # Contains "bioformats2raw.layout" metadata
    ├── OME                   # Special group for containing OME metadata
    │   ├── zarr.json         # Contains "series" metadata, listing all positions
    │   └── METADATA.ome.xml  # optional OME-XML file stored within the Zarr fileset
    ├── 0/                    # First image in the collection (same as 5D image above)
    ├── 1/                    # Second image in the collection
    └── ...
    ```

    !!! example "Example Code"
        [Writing multiple positions](../examples/multiposition.md)

=== "Multi-well Plates (HCS)"

    If you declare `AcquisitionSettings.plate` along with a `PositionDimension`
    containing [`Position`s][ome_writers.Position] that define `plate_well`/`plate_column`
    information, the output structure will follow the [OME-Zarr HCS
    specification](https://ngff.openmicroscopy.org/0.5/index.html#plate-md):

    ```
    output_path.ome.zarr/
    ├── zarr.json              # Plate metadata
    ├── A/                     # Row A
    │   ├── 1/                 # Well A1
    │   │   ├── zarr.json      # Well metadata
    │   │   ├── 0/             # Position 0 (Image with multiscales)
    │   │   │   ├── zarr.json  # contains "ome.multiscales" metadata
    │   │   │   ├── 0/         # Full resolution
    │   │   │   └── n/         # (Downsampled levels, not currently written)
    │   │   └── 1/             # Position 1  (Image with multiscales)
    │   └── 2/                 # Well A2
    └── B/                     # Row B
        ├── 1/                 # Well B1    
        └── ...
    ```

    !!! example "Example Code"
        [Writing plates](../examples/plate.md)

## Backends

!!! important
    All zarr format backends currently use `yaozarrs` to establish the group hierarchy
    and group-level `zarr.json` documents.  Only the array nodes are written using the
    specific backend libraries below.

- `tensorstore`: Uses Google's
  [`tensorstore`](https://google.github.io/tensorstore/) library.
- `acquire-zarr`: Uses the
  [`acquire-zarr`](https://github.com/acquire-project/acquire-zarr) library.
- `zarr-python`: Uses the reference
  [`zarr`](https://zarr.readthedocs.io/en/stable/) library (also known as
  `zarr-python` to disambiguate the Python library from the specification
  itself).
- `zarrs-python`: Uses the rust-backed
  [`zarrs-python`](https://github.com/zarrs/zarrs-python) library, on top of
  `zarr-python` to speed up array writes.
