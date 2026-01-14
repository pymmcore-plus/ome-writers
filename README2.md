# ome-writers Architecture

## Purpose

ome-writers is a Python library for writing microscopy image data to
OME-compliant formats (OME-TIFF and OME-Zarr). It is designed for **streaming
acquisition**: receiving 2D camera frames one at a time and writing them to
multi-dimensional arrays with proper metadata.

The core problem ome-writers solves:

> Map a **stream of 2D frames** (arriving in acquisition order) to **storage
> locations** in multi-dimensional arrays, while generating **OME-compliant
> metadata** for both TIFF and Zarr formats.

## High-Level Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌───────────────────────┐
│  StorageSchema  │─────▶│   FrameRouter   │─────▶│  ArrayBackend         │
│                 │      │                 │      │                       │
│  Declarative    │      │  __next__() ->  │      │  write(pos,idx,frame) │
│  exp definition │      │    (pos, idx)   │      │  finalize()           │
└─────────────────┘      └─────────────────┘      └───────────────────────┘
```

### StorageSchema (`schema.py`)

The schema is the **declarative description** of what to create.  In addition to
other storage details such as data types, chunking, compression, and other
metadata, it must fully describe the dimensionality of the data and the exact
order in which frames will arrive.

> **Explicit non-goal**: `ome-writers` does *not* attempt to handle non-deterministic
> acquisition patterns (e.g., event-driven acquisitions where data shape is unknown ahead
> of time).

It answers:

- What dimensions exist? (T, C, Z, Y, X, positions, plates, etc.)
- What is the acquisition order? (how will frames arrive)
- What is the storage order? (how should axes be arranged on disk)
- Data types, chunking, compression, sharding, etc.

The schema separates **acquisition order** (the order dimensions appear in the
`dimensions` list) from **storage order** (controlled by the `storage_order`
field). This allows data to arrive in one order (e.g., TZCYX) but be stored in
another (e.g., TCZYX for NGFF compliance).

### FrameRouter

The router is the **stateful iterator** that maps frame numbers to storage locations. It:

1. Reads the schema to understand both acquisition and storage order
2. Maintains iteration state (which frame are we on?)
3. Computes the permutation from acquisition order to storage order
4. Yields `(position_key, storage_index)` tuples for each frame

The router is the **only component that knows about both orderings**. It
iterates in acquisition order (because that's how frames arrive) and emits
storage-order indices (because that's what backends need).

### ArrayBackend

Backends are **format-specific writers** that handle the actual I/O. They:

1. Create arrays/files based on the schema
2. Write frames to specified locations
3. Generate format-appropriate metadata
4. Handle finalization (flushing, closing)

Supported backends:

- **tensorstore** / **zarr-python** — OME-Zarr v0.5 via yaozarrs (random-access writes)
- **acquire-zarr** — OME-Zarr v3 (sequential writes only)
- **tifffile** — OME-TIFF (sequential writes)

Backends receive indices in storage order and don't need to know about acquisition order.

## Design Principles

1. **Schema is declarative** — describes the target structure, not how to build it
2. **Router handles the mapping** — single place for acquisition→storage order logic
3. **Backends are simple adapters** — receive storage-order indices, write bytes
4. **Position is a meta-dimension** — appears in iteration but becomes separate arrays/files, not an array axis

## Why this level of abstraction?

The separation of schema, router, and backend allows us to leave the performance-critical
tasks to C++ libraries (like tensorstore, acquire-zarr), while keeping "fiddly" metadata
logic and frame routing in Python (where it's easier to maintain).

The API of this library is heavily inspired by the acquire-zarr API
(declare deterministic experiment with schema, append frames with single `append()` calls).
But we also:

- want to support non-zarr formats (OME-TIFF)
- want to take advantage of Python for metadata management (e.g. `ome-types` for
  OME-XML generation and `yaozarrs` for OME-Zarr metadata)
- want to support other zarr array libraries, such as tensorstore.

## Supported Use Cases

### Well Supported

- **Single 5D image** (TCZYX or any permutation) — the common case
- **Multi-position acquisition** — separate arrays/files per stage position
- **Well plates** — hierarchical plate/well/field structure with explicit acquisition order

### Challenging Edge Cases

- Jagged arrays: E.g.
  - one channel does Z-stacks while another does single planes
  - different positions have different shapes (nT, nZ, etc)
- Multi-camera setups, particularly with different image shapes or data types.
