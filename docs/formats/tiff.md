---
title: OME-TIFF
icon: lucide/file
---

!!! information "OME-TIFF Specification"

    We aim to fully comply with the official specification for OME-TIFF
    as defined by:

    <https://ome-model.readthedocs.io/en/stable/ome-tiff/index.html>

`ome-writers` supports writing to `OME-TIFF`, a stable and widely used standard
for storing multi-dimensional biological image data. `OME-TIFF` is an extension
of the standard TIFF format that includes metadata defined by the [OME-XML Data
Model](https://ome-model.readthedocs.io/en/stable/ome-xml/index.html) to
describe the higher order dimensionality and experimental context of the image
data.

## Expected Output Structure

When writing an acquisition to OME-TIFF, the expected output structure depends
on your [AcquisitionSettings][ome_writers.AcquisitionSettings].  OME-TIFF
supports no more than 5 dimensions per image (strictly: `T`, `C`, `Z`, `Y`,
`X`). Therefore, acquisitions with additional dimensions (e.g., stage position,
tiles, angles, etc.) are broken into series of ≤5D images.  The OME-TIFF
specification supports arranging those series either in a single large file, or
as multiple files in a directory structure.  `ome-writers` currently only
supports the latter approach: **positions/series are always written as separate
files in the output directory**

=== "Single ≤5D Image"

    Any acquisition with a combination of no more than "TCZXY" dimensions is
    stored as a single OME-TIFF file, directly at the
    `AcquisitionSettings.output_path`.

    !!! example "Example Code"
        [Writing a single ≤5D image](../examples/single_5d_image.md)

=== "Multi-Position & Other Collections"

    Acquisitions that contain multiple positions (e.g., stage positions, tiled
    images, angles on a light-sheet microscope, etc.) or that exceed 5
    dimensions in any way other will be structured as multiple OME-TIFF files, one per
    position/series.  For example, if `AcquisitionSettings.root_path` is
    "dest/my_file.ome.tiff" and there are 4 positions in the
    [`PositionDimension`][ome_writers.PositionDimension], the output will be:

    ```
    dest/my_file_position_p000.ome.tiff
    dest/my_file_position_p001.ome.tiff
    dest/my_file_position_p002.ome.tiff
    dest/my_file_position_p003.ome.tiff
    ```

    !!! example "Example Code"
        [Writing multiple positions](../examples/multiposition.md)

=== "Multi-well Plates (HCS)"

    If you declare `AcquisitionSettings.plate` along with a `PositionDimension`
    containing [`Position`s][ome_writers.Position] that define `plate_well`/`plate_column`
    information, the output follows the same pattern as above, but HCS metadata is included
    in the OME-XML.  **Work in progress:** full support for HCS metadata in OME-TIFF is
    still being developed.

    !!! example "Example Code"
        [Writing plates](../examples/plate.md)

## Backends

- `tifffile` (default): Uses the popular [`tifffile` Python library](https://github.com/cgohlke/tifffile/)
  to write OME-TIFF files, and [`ome-types`](https://github.com/tlambert03/ome-types) to generate the
  OME-XML metadata embedded in the TIFF header.
