# Contributing to ome-writers

This is a work in progress; we absolutely welcome and appreciate contributions!
If you have suggestions, improvements, or bug fixes, please open an issue or
submit a pull request.

## Scope of the project

To give you a general sense for whats "in-scope" for this project:

**`ome-writers` aims to provide a consistent API for *writing* data as it streams
from acquisition hardware, or is processed in memory, to standardized bioimaging
formats, specifically OME-TIFF and OME-Zarr.**

The actual writing of data is delegated to backend implementations, such as
(currently) `tifffile` for OME-TIFF and `tensorstore` or `acquire-zarr` for
OME-Zarr. `ome-writers` provides a common interface to these backends, and the
goal is for the user to be able to switch between backends while retaining the
same high-level API, which is roughly:

```python
import ome_writers

with ome_writers.create_stream(..., backend='some-backend') as stream:
    for frame in acquisition:
        stream.append(frame)
```

Anything that falls within this relatively broad scope is welcome, including,
but not limited to:

- New backends for writing OME-TIFF or OME-Zarr
- Performance improvements to existing backends
- Extended support for functionality provided by one or more backend
- Extended support for metadata, e.g. OME-XML, NGFF metadata

> In any API abstraction, the question arises of "how much to abstract",
> and whether we aim to cover the union or the intersection of functionality
> provided by the backends.  We don't have a strict policy on this, but we *do*
> want to be able to take advantage of unique features of a given backend, so
> backend-specific options will likely be considered.

## Dependency policy

We want to keep the core of `ome-writers` as lightweight as possible, but each
backend may have its own dependencies, which are declared under the
corresponding "extra" in the `[project.optional-dependencies]` table in
`pyproject.toml`.

> [!TIP]
> Though the repo is within the `pymmcore-plus` organization, `ome-writers` does
> *not* depend on `pymmcore-plus` or make any assumptions about where the data
> is coming from.  (`pymmcore-plus` may be a consumer of `ome-writers`, but
> `ome-writers` is not dependent on `pymmcore-plus`).

## Development

While not mandatory, we generally use [`uv`](https://docs.astral.sh/uv/) to
manage environments and dependencies, and instructions below assume you are
using `uv`.

```bash
# Clone the repo
git clone https://github.com/pymmcore-plus/ome-writers

# setup dev environment, with ome-writers installed in editable mode
uv sync
```

### Testing

```sh
uv run pytest
```

If you want to test *exactly* the dependencies for a specific extra, you can
use:

```sh
uv run --exact --only-group <backend> pytest
```

where `<backend>` is one of `tensorstore`, `acquire-zarr`, or `tifffile` (or any
future added backend.)

### Pre-commit (linting and formatting)

```sh
uv run pre-commit run -a

# or, to install the git hooks
uv run pre-commit install
```
