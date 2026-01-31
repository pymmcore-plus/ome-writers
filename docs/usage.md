---
title: Usage
icon: lucide/book-open
---

# Using `ome-writers`

## Basic Pattern

Almost all functionality of `ome-writers` begins by creating an
[`AcquisitionSettings`][ome_writers.AcquisitionSettings] object, which describes
the desired output dataset. This object is then passed to
[`create_stream`][ome_writers.create_stream] to setup the output hierarchy and
prepare for receiving frames:

```python
from ome_writers import AcquisitionSettings, Dimension


settings = AcquisitionSettings(
    root_path="example.ome.zarr",
    dimensions=[
        Dimension(name="t", count=2, chunk_size=1, type="time", unit="s"),
        Dimension(name="c", count=3, chunk_size=1, type="channel"),
        Dimension(name="z", count=32, chunk_size=16, type="space", unit="um"),
        Dimension(name="y", count=1024, chunk_size=512, type="space", unit="um"),
        Dimension(name="x", count=1024, chunk_size=512, type="space", unit="um"),
    ],
    dtype="uint16",
)

with create_stream(settings) as stream:
    for frame in frame_iterator():
        stream.append(frame)
```

As you can see, the [`Dimension`][ome_writers.Dimension] objects are a key part of the
`AcquisitionSettings`, describing the shape and chunking of each
dimension in the dataset, along with other critical metadata. So it pays to understand
the `Dimension` class well.

!!! example "Shorthand for standard axis names"
    The [`dims_from_standard_axes`][ome_writers.dims_from_standard_axes] helper function
    may be used to create a list of `Dimension` objects from a dictionary
    of [`StandardAxis`][ome_writers.StandardAxis] names
    (`"p", "t", "c", "z", "y", "x"`) and their sizes:

    The dimensions list shown above could have been created like this:

    ```python
    from ome_writers import dims_from_standard_axes

    dims = dims_from_standard_axes(
        {"t": 2,"c": 3,"z": 32,"y": 1024,"x": 1024},
        chunk_shapes={"z": 16, "y": 512, "x": 512},
    )
    ```

## Modifying Output Format

Both the suffix of the `root_path` and the `format` key in the
`AcquisitionSettings` may be used to specify the output format:

```python
# Specify OME-TIFF via suffix
settings = AcquisitionSettings(
    root_path='output.ome.tiff',
    ...
)

# Specify OME-TIFF via format (let format pick suffix)
settings = AcquisitionSettings(
    root_path='output',
    format="ome-tiff",
    ...
)

# Specify OME-Zarr via suffix
settings = AcquisitionSettings(
    root_path='output.ome.zarr',
    ...
)

# Specify OME-Zarr via format (let format pick suffix)
settings = AcquisitionSettings(
    root_path='output',
    format="ome-zarr",
    ...
)
```

!!! warning "Warning: Don't omit both format and suffix"
    It is ***not*** recommended, and may be an error in the future,
    to omit the `format` key when the `root_path` does not have a suffix.

    ````python
    >>> from ome_writers import AcquisitionSettings, dims_from_standard_axes
    >>> settings = AcquisitionSettings(
    ...     root_path='root',
    ...     dimensions=dims_from_standard_axes({"x": 512, "y": 512}),
    ...     dtype="uint8",
    ... )
    ````

    ````
    UserWarning: 

    Output format could not be inferred from root_path 'root'. 
    Picking the first available format/backend: 'ome-zarr'/'tensorstore'. 
    This may not be what you want, and may be an error in future versions.
    Please specify the desired format explicitly (e.g. format='ome-zarr')
    or via the extension of `root_path`.
    ````

    It is however, a flexible, acceptable practice to omit the extension
    from `root_path` **provided** the `format` key is explicitly set.

## Specifying Array Backend

The actual writing of arrays is done by a backend.  Backends are detailed
on each format's documentation page: [OME-TIFF backends](./formats/tiff.md#backends) and
[OME-Zarr backends](./formats/zarr.md#backends).

By default, the first
available backend for the requested file format is used. To specify a
particular backend, use an expanded `format` dictionary, with a `backend`
key:

```python
settings = AcquisitionSettings(
    root_path='output',
    format={"name": "ome-zarr", "backend": "acquire-zarr"},
    ...
)
```

Or, as a shorthand (since each backend only supports one format),
you may also pass the backend name directly to the `format` key:

```python
settings = AcquisitionSettings(
    root_path='output',
    format="acquire-zarr",
    ...
)
```

## Common Dimension Setups

### Single 5D Image

```python
# at each time point, for each channel, acquire a 3D stack:
dimensions=[
    Dimension(name="t", count=2, chunk_size=1, type="time"),
    Dimension(name="c", count=3, chunk_size=1, type="channel"),
    Dimension(name="z", count=4, chunk_size=1, type="space", scale=.5, unit="um"),
    Dimension(name="y", count=2048, chunk_size=512, type="space", scale=.1, unit="um"),
    Dimension(name="x", count=2048, chunk_size=512, type="space", scale=.1, unit="um"),
]
```

Remember, order of dimensions matters!  They must be ordered from
slowest-changing to fastest-changing (i.e. C-order).  Note the difference
in semantics below, where now we acquire all channels for each plane in the stack:

```python
# for each time point, acquire ALL channels in sequence
# while visiting each plane in the 3D stack:
dimensions=[
    Dimension(name="t", count=2, chunk_size=1, type="time"),
    Dimension(name="z", count=4, chunk_size=1, type="space", scale=.5, unit="um"),
    Dimension(name="c", count=3, chunk_size=1, type="channel"),
    Dimension(name="y", count=2048, chunk_size=512, type="space", scale=.1, unit="um"),
    Dimension(name="x", count=2048, chunk_size=512, type="space", scale=.1, unit="um"),
]
```

See the [single 5D image example](../examples/single_5d_image.md).

### Multiple Positions

```python
# for each time point, visit each position, and for each channel, acquire a 3D stack:
dimensions=[
    Dimension(name="t", count=2, chunk_size=1, type="time"),
    PositionDimension(positions=["Pos0", "Pos1"]),
    Dimension(name="c", count=3, chunk_size=1, type="channel"),
    Dimension(name="z", count=4, chunk_size=1, type="space", scale=5, unit="um"),
    Dimension(name="y", count=512, chunk_size=256, type="space", scale=2, unit="um"),
    Dimension(name="x", count=512, chunk_size=256, type="space", scale=2, unit="um"),
]
```

At this time, *neither* OME-TIFF *nor* OME-Zarr support more than
5 dimensions in a single image. So these datasets will be composed
of multiple 5D images, one per position.

See the [multi-position example](../examples/multiposition.md).

### Multi-well plates (HCS)

Multi-well plates are declared by the addition of the `plate` key
to the `AcquisitionSettings`, along with `plate_row/plate_column`
definitions in the [`Position`][ome_writers.Position] objects used to
define each position.  While there may be many fields of view per well,
currently both specifications are limited to no more than 5 `TCZYX`
dimensions in any given field of view.

```python
settings = AcquisitionSettings(
    root_path="example_5d_plate",
    dimensions=[
        Dimension(name="t", count=2, chunk_size=1, type="time"),
        PositionDimension(
            # order should match stage position traversal during acquisition
            positions=[
                Position(name="fov0", plate_row="A", plate_column="1"),
                Position(name="fov0", plate_row="A", plate_column="2"),
                # note ... two fovs in same well
                Position(name="fov0", plate_row="C", plate_column="4"),
                Position(name="fov1", plate_row="C", plate_column="4"),
            ]
        ),
        Dimension(name="c", count=3, chunk_size=1, type="channel"),
        Dimension(name="z", count=4, chunk_size=1, type="space"),
        Dimension(name="y", count=256, chunk_size=64, type="space"),
        Dimension(name="x", count=256, chunk_size=64, type="space"),
    ],
    dtype="uint16",
    plate=Plate(
        name="Example Plate",
        row_names=["A", "B", "C", "D"],
        column_names=["1", "2", "3", "4", "5", "6", "7", "8"],
    ),
)
```

See the [multi-well plate example](../examples/plate.md).

### Unbounded Dimensions

If you don't know ahead of time how many frames you will acquire
along a particular dimension (e.g. time), you may declare that
dimension as unbounded by setting its `count` to `None`.  Only the
first `Dimension` may be unbounded.

```python
dimensions=[
    Dimension(name="t", count=None, type="time"),
    PositionDimension(positions=["Pos0", "Pos1"]),
    Dimension(name="y", count=256, chunk_size=64, type="space"),
    Dimension(name="x", count=256, chunk_size=64, type="space"),
]
```

See the [unbounded dimensions example](../examples/unbounded.md).

### Everything Else

If your dataset is fundamentally incompatible with the limitations of either
OME-TIFF or OME-Zarr (e.g. ragged dimensions, more than 5 dimensions per image,
non-standard axes, non-deterministic axes), we recommend following the patterns
explained in the [event-driven example](../examples/event_driven.md).

In short:

1. **Create a 3D dimensional dataset, with an unbounded first dimension**,
    that simply represents all of the frames you will acquire over the
    course of the acquisition.

    > *Unless it pains you deeply (:smile:), we suggest using a standard
    axis name like `"t"` for the unbounded dimension, so that it can be
    stored in either OME-TIFF or OME-Zarr.*

    ```python
    dimensions=[
        Dimension(name="t", count=None, type="time"),
        Dimension(name="y", count=256, chunk_size=64, type="space"),
        Dimension(name="x", count=256, chunk_size=64, type="space"),
    ]
    ```

2. **Append frames as they come, with `frame_metadata` describing context.**

    `frame_metadata` can be associated with each frame in the
    [`append`][ome_writers.OMEStream.append] function.  This metadata is stored
    in the OME-XML/OME-NGFF metadata (see the
    [example](../examples/event_driven.md) for details), and may be used to
    (manually) arrange frames into some higher-level structure after
    acquisition.

    ```python
    with create_stream(settings) as stream:
        for frame, metadata in frame_iterator():
            # anything you need to describe the frame's context
            metadata = {
                "position": position_id,
                "timestamp": time_point,
                "channel": channel_id,
                ...
            }
            stream.append(frame, metadata=metadata)
    ```

!!! danger "Even Frame Shape changing?"
    `ome-writers`, OME-TIFF, and OME-Zarr all expect that the shape
    of each frame will be constant throughout the acquisition.  If
    even your frame shape changes over time, you will unfortunately
    need to come up with a custom solution.  Please open an issue
    if this use case is particularly important to you.

## Understanding Storage Order

Microscopy acquisitions can happen in many different orders:  You might acquire
all channels at each Z-plane before moving to the next Z-plane (order `zcyx`),
or acquire all Z-planes for each channel before switching channels (order
`czxy`), etc.  Different file formats have different support for this.

While OME-TIFF is relatively flexible, supporting any permutation of `TCZ`...  
**OME-Zarr is currently *strictly* limited to TCZYX storage order.**

> [axes] MUST be ordered by "type" where the "time" axis must come first (if
> present), followed by the "channel" or custom axis (if present) and the axes
> of type "space". If there are three spatial axes where two correspond to the
> image plane ("yx") and images are stacked along the other (anisotropic) axis
> ("z"), the spatial axes SHOULD be ordered as "zyx".

!!! note "Discussions ongoing"
    [RFC-3](https://ngff.openmicroscopy.org/rfc/3/index.html) is an active
    discussion about relaxing this restriction in future versions of the
    OME-NGFF specification.

This restriction makes it extremely hard to directly write some acquisitions to
OME-Zarr without re-ordering frames in memory or on-disk after acquisition.
This can be controlled using the `storage_order` key in the `AcquisitionSettings`.

```python
settings = AcquisitionSettings(
    root_path="example.ome.zarr",
    dimensions=[
        Dimension(name="t", ...),
        Dimension(name="z", ...),
        Dimension(name="c", ...),
        Dimension(name="y", ...),
        Dimension(name="x", ...),
    ],
    dtype="uint16",
    storage_order="ome",  # the default
)
```

By default (`storage_order="ome"`), `ome-writers` always attempts to write
*spec-compliant* datasets, for both OME-TIFF and OME-Zarr, it *will* permute
frames when necessary to ensure that the output dataset is valid. So, in the
example above, frames will be re-ordered from `tzcyx` to `tczyx` when writing
OME-Zarr (since that's the only valid order for that format), but will be written
as `tzcyx` for OME-TIFF (which supports that order).

Other options for `storage_order`, both of which may produce non-compliant
datasets, are:

- `"acquisition"`:  Frames are written in the order defined by the
  `dimensions` list, *without* any re-ordering.
- `list[str]`: A list of dimension names defining the desired storage order.
  Frames will be re-ordered as necessary to match this order.

Backends may also impose additional restrictions on storage order.  If you run
into difficulties with `storage_order`, please
[open an issue](https://github.com/pymmcore-plus/ome-writers/issues/new) to
discuss your use case.

## Compressing Data

Both OME-TIFF and OME-Zarr support compression of image data (though
supported compression types vary by format and backend).  Compression
is specified via the `compression` key in the `AcquisitionSettings`.

```python
settings = AcquisitionSettings(
    root_path="example.ome.zarr",
    dimensions=[...],
    dtype="uint16",
    compression="blosc-zstd",
)
```

Using a compression type that is not supported by the selected format/backend
will raise an error when creating the `AcquisitionSettings` object.

## Where did the data go?

Different formats have different conventions for how data is stored on-disk.
It's easy with OME-Zarr to guarantee that all data is inside of a zarr group
at your `AcquisitionSettings.root_path`, but OME-TIFF has many different
valid conventions (everything in a single file, one file per position, etc...).

Similarly, various formats may add suffixes or sub-directories to the specified
`AcquisitionSettings.root_path`.

For these reasons, the
[`AcquisitionSettings.output_path`][ome_writers.AcquisitionSettings.output_path]
property exists. It resolves to **the root *container* of the actual data
written**.  That container may be a single file (e.g. `some_data.ome.tiff`), a
directory that contains multiple files (e.g. `some_data/pos0.ome.tiff`,
`some_data/pos1.ome.tiff`, ...), or a zarr group with or without suffix (e.g.
`some_data.ome.zarr/`).

`AcquisitionSettings.output_path` will *always* exist, but you cannot assume
that `AcquisitionSettings.root_path` does, and you cannot assume that
`output_path` will always be a file or a directory (though these things *can* be
determined by and gleaned from your `format` settings).
