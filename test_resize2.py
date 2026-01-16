"""Test different resize approaches."""

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

shape = (1, 256, 256)
dtype = "uint16"

# Test with tensorstore
print("=== Testing tensorstore ===")
_, arrays = prepare_image(
    root,
    image,
    datasets=[(shape, dtype)],
    overwrite=True,
    chunks=(1, 64, 64),
    writer="tensorstore",
)
ts_array = arrays["0"]
print(f"Type: {type(ts_array)}")
print(f"Dir: {[attr for attr in dir(ts_array) if not attr.startswith('_')]}")
print(f"Has resize attr: {hasattr(ts_array, 'resize')}")
print(f"Resize callable: {callable(getattr(ts_array, 'resize', None))}")

# Check if there's a spec or schema attribute
if hasattr(ts_array, "spec"):
    print(f"\nSpec: {ts_array.spec}")
if hasattr(ts_array, "schema"):
    print(f"Schema: {ts_array.schema}")
if hasattr(ts_array, "domain"):
    print(f"Domain: {ts_array.domain}")

# Test with zarr backend instead
print("\n=== Testing zarr backend ===")
root2 = tmpdir / "test2.zarr"
_, arrays2 = prepare_image(
    root2,
    image,
    datasets=[(shape, dtype)],
    overwrite=True,
    chunks=(1, 64, 64),
    writer="zarr",
)
zarr_array = arrays2["0"]
print(f"Type: {type(zarr_array)}")
print(f"Shape before: {zarr_array.shape}")
zarr_array.resize((2, 256, 256))
print(f"Shape after resize: {zarr_array.shape}")

# Try writing to index 1
frame1 = np.ones((256, 256), dtype=np.uint16)
try:
    zarr_array[(1,)] = frame1
    print("Successfully wrote to index 1 with zarr backend")
except Exception as e:
    print(f"Error with zarr backend: {e}")
