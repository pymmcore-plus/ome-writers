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

## Installation

See dedicated [installation instructions](install.md).

## Usage

See [Using `ome-writers`](usage.md) for a quick overview of how to use the library.

```python
from ome_writers import AcquisitionSettings, create_stream

settings = AcquisitionSettings( ... )

with create_stream(settings) as stream:
    for frame in frame_generator():
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
