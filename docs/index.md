---
icon: lucide/rocket
---

# Get started

## OME-writers

`ome-writers` is a Python library that provides a unified interface for writing
microscopy image data to OME-compliant formats (OME-TIFF and OME-Zarr) using
various different backends. It is designed for **streaming acquisition**:
receiving 2D camera frames one at a time and writing them to multi-dimensional
arrays with proper metadata.

The core API looks something like this (pseudocode):

```python
from ome_writers import AcquisitionSettings, create_stream

# define the dimensions of your experiment
# and storage settings such as chunk sizes, data type, etc.
settings = AcquisitionSettings( ... )

# create a stream writer based on those settings
with create_stream(settings) as stream:
    # append camera frames as they arrive
    for frame in acquisition:
        stream.append(frame)
```

For complete reference on how to build `AcquisitionSettings`, see the
[API documentation](api/index.md).
