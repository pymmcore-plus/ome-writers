"""Test tensorstore resize behavior."""

import tempfile
from pathlib import Path

import tensorstore as ts
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
print(f"Initial shape: {array.shape}")
print(f"Initial domain: {array.domain}")

# Try different resize approaches
print("\n=== Test 1: Call resize with list ===")
result = array.resize([2, 256, 256])
print(f"Resize return value: {result}")
print(f"Shape after resize: {array.shape}")
print(f"Domain after resize: {array.domain}")

print("\n=== Test 2: Check if resize is async ===")
result = array.resize([3, 256, 256])
if hasattr(result, "result"):
    print("Result has .result() method, calling it...")
    result.result()
    print(f"Shape after result(): {array.shape}")

print("\n=== Test 3: Reopen the array ===")
# Try reopening the array
spec = array.spec()
reopened = ts.open(spec).result()
print(f"Reopened shape: {reopened.shape}")
print(f"Reopened domain: {reopened.domain}")

print("\n=== Test 4: Try with explicit inclusive_min/exclusive_max ===")
# Try to resize by creating a new array with updated bounds
try:
    resized = array[ts.d[:].translate_to[0]][ts.d[0].mark_bounds_implicit()[0:4]]
    print(f"After bound manipulation: {resized.shape}")
    print(f"Domain: {resized.domain}")
except Exception as e:
    print(f"Error: {e}")
