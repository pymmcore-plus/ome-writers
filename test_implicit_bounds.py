"""Test different approaches to handle unbounded dimensions."""

import tempfile
from pathlib import Path

import numpy as np
import tensorstore as ts
from yaozarrs import DimSpec, v05
from yaozarrs.write.v05 import prepare_image

tmpdir = Path(tempfile.mkdtemp())

print("=== Approach 1: Try using a large initial size ===")
root1 = tmpdir / "large_init.zarr"
dims = [
    DimSpec(name="t", size=None, type="time"),  # unbounded
    DimSpec(name="y", size=256, type="space"),
    DimSpec(name="x", size=256, type="space"),
]
image = v05.Image(multiscales=[v05.Multiscale.from_dims(dims)])

# Start with a larger size for the unbounded dimension
_, arrays = prepare_image(
    root1,
    image,
    datasets=[((10000, 256, 256), "uint16")],  # Large initial size
    overwrite=True,
    chunks=(1, 64, 64),
    writer="tensorstore",
)
array1 = arrays["0"]
print(f"Domain: {array1.domain}")
print(f"Shape: {array1.shape}")

# Try writing to various indices
for i in [0, 1, 100, 1000]:
    try:
        array1[i] = np.full((256, 256), i, dtype=np.uint16)
        print(f"✓ Write to index {i} succeeded")
    except Exception as e:
        print(f"✗ Write to index {i} failed: {e}")

print("\n=== Approach 2: Check what's in the spec ===")
root2 = tmpdir / "check_spec.zarr"
_, arrays2 = prepare_image(
    root2,
    image,
    datasets=[((1, 256, 256), "uint16")],
    overwrite=True,
    chunks=(1, 64, 64),
    writer="tensorstore",
)
array2 = arrays2["0"]
spec = array2.spec()
print(f"Spec: {spec}")
print(f"\nTransform: {spec.transform}")

# Check if we can reopen with modified bounds
print("\n=== Approach 3: Reopen with modified spec ===")
try:
    # Get the original spec
    orig_spec = array2.spec().to_json()
    print(f"Original transform: {orig_spec.get('transform', {})}")

    # Try reopening without explicit bounds in transform
    modified_spec = orig_spec.copy()
    if (
        "transform" in modified_spec
        and "input_exclusive_max" in modified_spec["transform"]
    ):
        # Make the first dimension unbounded
        modified_spec["transform"]["input_exclusive_max"][0] = None

    print(f"Modified spec: {modified_spec}")
    reopened = ts.open(modified_spec).result()
    print(f"Reopened shape: {reopened.shape}")
    reopened[1] = np.ones((256, 256), dtype=np.uint16)
    print("✓ Write to index 1 succeeded with reopened array!")
except Exception as e:
    print(f"✗ Reopen approach failed: {e}")
