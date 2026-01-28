# Contributing to ome-writers

This is a work in progress; we absolutely welcome and appreciate contributions!
If you have suggestions, improvements, or bug fixes, please open an issue or
submit a pull request.

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
