---
title: Scratch Format
icon: lucide/memory-stick
---

The `scratch` format provides lightweight in-memory (or disk-backed) array
storage. It is useful for testing, previewing, and real-time visualization of
acquisition data without committing to an on-disk file format.

Data is stored in numpy arrays (one per position), and can be accessed as
standard numpy-like arrays during and after acquisition.

!!!warning
    The disk-backed scratch format is **not intended for long-term storage or
    interoperability.** It is an internal format optimized for speed
    and flexibility, and may change without warning.

## Usage

Select the scratch format by setting `format="scratch"` (or the alias `format="memory"`),
or by using a [`ScratchFormat`][ome_writers.ScratchFormat] object directly.

```python
from ome_writers import AcquisitionSettings, dims_from_standard_axes, create_stream

settings = AcquisitionSettings(
    dimensions=dims_from_standard_axes({"t": 10, "c": 2, "y": 512, "x": 512}),
    format="scratch",
    dtype="uint16",
)

with create_stream(settings) as stream:
    for frame in my_frames:
        stream.append(frame)
```

No `root_path` is required for pure in-memory mode. However, if `root_path` is
provided, data is written to disk as memory-mapped files for crash recovery
(see [Disk-backed mode](#disk-backed-mode) below).  `root_path` should be a
directory path (not a file path), and will be created if it does not exist.

## Memory Limits

By default, the scratch format allows up to 4 GB of in-memory storage
(configurable via
[`ScratchFormat.max_memory_bytes`][ome_writers.ScratchFormat]). If the
estimated array size exceeds this limit:

- **With `spill_to_disk=True`** (default): data is automatically written to a
  temporary directory using memory-mapped files. A warning is emitted with the
  temporary path, and the directory is automatically cleaned up on process exit.
- **With `spill_to_disk=False`**: a `MemoryError` is raised.

You can also set `root_path` explicitly to always use disk-backed storage,
regardless of array size.

## Disk-backed Mode

When `root_path` is set (or when spilling to disk), the scratch backend writes
memory-mapped (numpy memmap) files alongside a JSON manifest:

```
root_path/
├── manifest.json          # Acquisition settings + position shapes
├── pos_0.dat              # Memory-mapped array for position 0
├── pos_1.dat              # Memory-mapped array for position 1
├── ...
└── frame_metadata.jsonl   # Per-frame metadata (if provided)
```

The `manifest.json` contains the full `AcquisitionSettings` dump plus a
`position_shapes` key listing the logical shape of each position array. Data
files can be read back with:

```python
import json
import numpy as np

manifest = json.loads(open("root_path/manifest.json").read())
shape = tuple(manifest["position_shapes"][0])
dtype = manifest["dtype"]
data = np.memmap("root_path/pos_0.dat", dtype=dtype, mode="r", shape=shape)
```
