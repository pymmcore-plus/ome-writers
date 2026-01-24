---
icon: lucide/gauge
title: Performance
---

# Performance Considerations

There are many considerations when it comes to performance in writing OME-Zarr
data. These include the choice of backend, chunk and shard sizes, compression
settings, and the overall data shape.  It's difficult for `ome-writers` to
strike a balance that works optimally with no additional configuration for all
use cases, so it's recommended to experiment with different settings for your
specific use case.

In particular, chunk sizes, sharding, and compression settings can have a major
impact on performance for zarr-based formats like OME-Zarr.  Write performance
is generally improved with larger chunk sizes and shards, though this may come
at the cost of read performance if the chunks are too large for typical access
patterns.  Different backends may also perform better or worse depending on the
specific data shape and chunking/sharding strategy used.

## Benchmarking

The `tools/benchmark.py` script can be used to benchmark the performance of
different `ome-writers` backends, with flexible acquisition settings.
Currently, it requires that you clone the repository and run it locally:

```bash
git clone https://github.com/pymmcore-plus/ome-writers
cd ome-writers
```

Then use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to run
the benchmark script with the desired options (uv will install the required
dependencies in an isolated environment in `.venv`). For more information on the
available options, run:

```bash
uv run tools/benchmark.py --help
```

The most important parameter is the `--dims`/`-d` argument, which specifies the
shape, chunking, and sharding of the data to be written.  The format is a
comma-separated list of dimension specifications, where each specification is
`name:size[:chunk_size[:shard_size]]` (chunk and shard sizes are optional).  For
example, to benchmark writing a 20-frame timelapse of 1024x1024 images with 256x256
chunks and no sharding, you would use:

```bash
uv run tools/benchmark.py -d t:20,y:1024:256,x:1024:256
```

By default, all available backends will be benchmarked.  You can specify a subset
of backends to test using the `--backends`/`-b` argument, it may be used multiple
times:

```bash
uv run tools/benchmark.py -d t:20,y:1024:256,x:1024:256 -b tensorstore -b acquire-zarr -b zarrs-python
```

Run `--help` for more options, including compression settings and output formats.

### Example Results

```
Benchmark Configuration
  Backends: tensorstore, acquire-zarr, zarrs-python
  Dimensions: 'tyx' (20, 1024, 1024)
  Chunk shape: (1, 256, 256)
  Total frames: 20
  Dtype: uint16
  Compression: None
  Warmups: 1
  Iterations: 30

Benchmarking tensorstore
  Running 1 warmup(s)...
  Running 30 iteration(s)...
  Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02
✓ tensorstore complete

Benchmarking acquire-zarr
  Running 1 warmup(s)...
  Running 30 iteration(s)...
  Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
✓ acquire-zarr complete

Benchmarking zarrs-python
  Running 1 warmup(s)...
  Running 30 iteration(s)...
  Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
✓ zarrs-python complete


Benchmark Results

Test Conditions:
  Total shape: (20, 1024, 1024)
  Frame shape: (1024, 1024)
  Number of frames: 20
  Data type: uint16
  Chunk shape: (1, 256, 256)
  MB per chunk: 0.125
  Total data: 0.039 GB
  Compression: none

┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric              ┃   tensorstore ┃  acquire-zarr ┃  zarrs-python ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ create (mean±std s) │ 0.001 ± 0.000 │ 0.001 ± 0.000 │ 0.002 ± 0.000 │
│ write  (mean±std s) │ 0.047 ± 0.005 │ 0.043 ± 0.001 │ 0.035 ± 0.003 │
│ throughput    (fps) │         423.5 │         467.9 │         575.4 │
│ bandwidth    (GB/s) │         0.888 │         0.981 │         1.207 │
└─────────────────────┴───────────────┴───────────────┴───────────────┘
```
