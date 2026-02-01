---
icon: lucide/rocket
title: Get started
---

# Welcome to OME-writers

`ome-writers` is a Python library that provides a unified interface for writing
microscopy image data to OME-compliant formats (*either* OME-TIFF or OME-Zarr) using
various different backends. It is designed for **streaming acquisition**:
receiving 2D camera frames one at a time and writing them to multi-dimensional
arrays with proper metadata.

*We prioritize:*

- ✅ **Correctness**: Strict adherence to both OME-TIFF and OME-Zarr specifications.
- 💯 **Completeness**: We want all the metadata to go to its proper place.
- 🚀 **Performance**: Very minimal, native-backed hot-path logic when appending frames.
- 🤸‍♂️ **Flexibility**: Pick from 5+ array backends, suiting your dependency preferences.
- 📖 **Usability**: Relatively small, well organized API, with extensive documentation.
- 💪 **Stability**: Minimal dependencies, exhaustive testing and validation.

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

## Feedback

We love your feedback!

If you have usage questions (*"How do I...?"*), please post on the [image.sc
forum](https://forum.image.sc/) with tag `ome-writers` or `pymmcore-plus`.

And of course, if you have bug reports, feature requests, or any other feedback,
you can always [open an issue on
GitHub](https://github.com/pymmcore-plus/ome-writers/issues/new).
