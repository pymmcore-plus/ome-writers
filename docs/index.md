---
icon: lucide/rocket
title: Get started
---

# Getting started with OME-writers

`ome-writers` is a Python library that provides a unified interface for writing
microscopy image data to OME-compliant formats (OME-TIFF and OME-Zarr) using
various different backends. It is designed for **streaming acquisition**:
receiving 2D camera frames one at a time and writing them to multi-dimensional
arrays with proper metadata.

The core API involves creating an
[`AcquisitionSettings`][ome_writers.AcquisitionSettings] object that fully
defines the data (dimensions, data type, etc.) as it will arrive from the
microscope, and then using the
[`create_stream()`][ome_writers.create_stream] factory function to create an
[`OMEStream`][ome_writers.OMEStream] object that can accept frames via its
[`append()`][ome_writers.OMEStream.append] method.

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

## Reference

For complete reference on how to build `AcquisitionSettings`, see the
[API documentation](reference/index.md).

## Examples

For more use-case specific examples, see the examples:

- [Writing a single ≤5D image](examples/single_5d_image.md)
- [Multiple positions](examples/multiposition.md)
- [Multi-well plates](examples/plate.md)
- [Unbounded first dimension](examples/unbounded.md)
- [Transposed storage layout](examples/transposed.md)
