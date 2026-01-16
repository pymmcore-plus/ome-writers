"""Minimal test to debug resize issue."""

import tempfile
from pathlib import Path

import numpy as np
from yaozarrs import DimSpec, v05
from yaozarrs.write.v05 import prepare_image

# Create a temporary directory
tmpdir = Path(tempfile.mkdtemp())
root = tmpdir / "test.zarr"

# Create a simple image with unbounded first dimension
dims = [
    DimSpec(name="t", size=None, type="time"),  # unbounded
    DimSpec(name="y", size=256, type="space"),
    DimSpec(name="x", size=256, type="space"),
]
image = v05.Image(multiscales=[v05.Multiscale.from_dims(dims)])

# Initial shape with unbounded dim set to 1
shape = (1, 256, 256)
dtype = "uint16"

# Prepare with tensorstore backend
_, arrays = prepare_image(
    root,
    image,
    datasets=[(shape, dtype)],
    overwrite=True,
    chunks=(1, 64, 64),
    writer="tensorstore",
)

array = arrays["0"]
print(f"Initial array shape: {array.shape}")
print(f"Array type: {type(array)}")

# Try to write to index 0 (should work)
frame0 = np.zeros((256, 256), dtype=np.uint16)
array[(0,)] = frame0
print(f"After writing index 0, shape: {array.shape}")

# Now try to resize and write to index 1
print(f"\nBefore resize: {array.shape}")
new_shape = [2, 256, 256]
array.resize(new_shape)
print(f"After resize: {array.shape}")
print(f"Array object id after resize: {id(array)}")

# Try to write to index 1
frame1 = np.ones((256, 256), dtype=np.uint16)
try:
    array[(1,)] = frame1
    print("Successfully wrote to index 1")
    print(f"Final shape: {array.shape}")
except ValueError as e:
    print(f"ERROR writing to index 1: {e}")
    print(f"Array shape at error time: {array.shape}")
