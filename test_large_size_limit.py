"""Test what happens when we exceed the large initial size."""

import tempfile
from pathlib import Path

import numpy as np
from yaozarrs import DimSpec, v05
from yaozarrs.write.v05 import prepare_image

tmpdir = Path(tempfile.mkdtemp())
root = tmpdir / "test.zarr"

dims = [
    DimSpec(name="t", size=None, type="time"),  # unbounded
    DimSpec(name="y", size=256, type="space"),
    DimSpec(name="x", size=256, type="space"),
]
image = v05.Image(multiscales=[v05.Multiscale.from_dims(dims)])

# Start with size 100
_, arrays = prepare_image(
    root,
    image,
    datasets=[((100, 256, 256), "uint16")],
    overwrite=True,
    chunks=(1, 64, 64),
    writer="tensorstore",
)
array = arrays["0"]
print(f"Initial shape: {array.shape}")

# Try writing at index 99 (within bounds)
try:
    array[99] = np.ones((256, 256), dtype=np.uint16)
    print("✓ Write to index 99 succeeded")
except Exception as e:
    print(f"✗ Write to index 99 failed: {e}")

# Try writing at index 100 (at the boundary)
try:
    array[100] = np.ones((256, 256), dtype=np.uint16)
    print("✓ Write to index 100 succeeded")
except Exception as e:
    print(f"✗ Write to index 100 failed: {e}")

# Try resizing to a larger size
print(f"\nTrying to resize from {array.shape[0]} to 200...")
try:
    result = array.resize([200, 256, 256])
    result.result()
    print(f"✓ Resize succeeded! New shape: {array.shape}")

    # Try writing at index 150
    array[150] = np.ones((256, 256), dtype=np.uint16)
    print("✓ Write to index 150 succeeded after resize")
except Exception as e:
    print(f"✗ Resize failed: {e}")
