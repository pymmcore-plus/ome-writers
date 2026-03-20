# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "ome-writers[all]",
#     "pymmcore-plus>=0.16.0",
# ]
#
# [tool.uv.sources]
# ome-writers = { path = "../" }
# ///
"""Example of using ome_writers to store data acquired with pymmcore-plus."""

import sys

import useq
from pymmcore_plus import CMMCorePlus

from ome_writers import AcquisitionSettings, Dimension

# Initialize pymmcore-plus core and load system configuration (null = demo config)
core = CMMCorePlus()
core.loadSystemConfiguration()

# Create a MDASequence, which will be used to run the MDA with pymmcore-plus
seq = useq.MDASequence(
    stage_positions=(
        {"x": 0, "y": 0, "name": "Pos0"},
        {"x": 1, "y": 1, "name": "Pos1"},
        {"x": 0, "y": 1, "name": "Pos2"},
    ),
    channels=(
        {"config": "DAPI", "exposure": 2},
        {"config": "FITC", "exposure": 10},
    ),
    time_plan={"interval": 0.5, "loops": 2},
    z_plan={"range": 3.5, "step": 0.5},
    axis_order="tpcz",
)

# Setup the AcquisitionSettings, converting the MDASequence to ome-writers Dimensions
# Derive format/backend from command line argument (default: auto)
FORMAT = "auto" if len(sys.argv) < 2 else sys.argv[1]

# Create AcquisitionSettings with just user preferences (chunk sizes, compression,
# format, etc.). Dimensions, dtype, and image sizes are filled in automatically
# by pymmcore-plus when running the MDA via core.mda.run(seq, output=settings).
settings = AcquisitionSettings(
    root_path="example_pymmcore_plus",
    dimensions=[
        Dimension(name="z", chunk_size=4),
    ],
    overwrite=True,
    format=FORMAT,
)

# Run the MDA — pymmcore-plus derives dimensions from the sequence and camera,
# merges in user-provided dimension overrides (e.g. chunk_size), and writes data.
core.mda.run(seq, output=settings)


if settings.format.name == "ome-zarr":
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.root_path + ".ome.zarr")
    print("Zarr store is valid")

if settings.format.name == "ome-tiff":
    from pathlib import Path

    from ome_types import from_tiff

    output_path = Path(settings.root_path)
    # single-position -> file, multi-position -> directory of .tiff files
    if output_path.with_suffix(".ome.tiff").is_file():
        files = [output_path.with_suffix(".ome.tiff")]
    else:
        files = list(output_path.glob("*.tiff"))
    for file in files:
        from_tiff(str(file))
        print(f"TIFF file {file.name} is valid")
