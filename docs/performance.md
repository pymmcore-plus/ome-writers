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

**In particular: chunk size, sharding, and compression settings will have a
major impact on performance for zarr-based formats like OME-Zarr.**

Write performance is generally improved with larger chunk sizes and shards,
though this may come at the cost of read performance if the chunks are too large
for typical access patterns.  Different backends may also perform better or
worse depending on the specific data shape and chunking/sharding strategy used.

!!! info "See Also"
    The documentation for zarr-python also has some [tips for chunk
    optimization](https://zarr.readthedocs.io/en/latest/user-guide/performance/#chunk-optimizations)

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

## Profiling with cProfile

There is also a `tools/profile.py` script that can be used to profile the performance
of different `ome-writers` backends using the built-in `cProfile` module.  Similar to the
benchmark script, it requires cloning the repository and running it locally:

```bash
git clone https://github.com/pymmcore-plus/ome-writers
cd ome-writers
```

You can run the profiling script with `uv`:

```bash
uv run tools/profiler.py --help
```

As with benchmarking, the most important parameter is the `--dims`/`-d`
argument, which specifies the shape, chunking, and sharding of the data to be
written.  In this case, you must specify a single backend to profile using the
`--backend`/`-b` argument:

```bash
uv run tools/profiler.py -d t:20,y:1024:256,x:1024:256 -b tensorstore
```

### Example Results

```
Profiling tensorstore
  Shape: (20, 1024, 1024)
  Chunks: (1, 256, 256)
  Frames: 20

Generating frames...
Setting up stream...
Profiling append + finalize...

                                      Top 20 functions by time                                       
┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Calls ┃   Time ┃ Cumulative ┃ Location                                                            ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    20 │ 0.000s │     0.000s │ src/ome_writers/_backends/_tensorstore.py:24(_write)                │
│     1 │ 0.000s │     0.000s │ ~:0(<method 'disable' of '_lsprof.Profiler' objects>)               │
│     2 │ 0.000s │     0.000s │ ~:0(<method 'clear' of 'list' objects>)                             │
│    20 │ 0.000s │     0.000s │ src/ome_writers/_router.py:161(__next__)                            │
│     1 │ 0.000s │     0.000s │ src/ome_writers/_backends/_yaozarrs.py:352(finalize)                │
│    20 │ 0.000s │     0.000s │ src/ome_writers/_backends/_yaozarrs.py:222(write)                   │
│    20 │ 0.000s │     0.000s │ src/ome_writers/_router.py:188(_increment_indices)                  │
│    20 │ 0.000s │     0.000s │ ~:0(<built-in method builtins.next>)                                │
│    20 │ 0.000s │     0.000s │ ~:0(<method 'pop' of 'list' objects>)                               │
│    40 │ 0.000s │     0.000s │ src/ome_writers/_router.py:180(<genexpr>)                           │
│    20 │ 0.000s │     0.000s │ ~:0(<method 'append' of 'list' objects>)                            │
│     1 │ 0.000s │     0.000s │ src/ome_writers/_stream.py:51(append)                               │
│    20 │ 0.000s │     0.000s │ ~:0(<built-in method builtins.max>)                                 │
│     1 │ 0.000s │     0.000s │ src/ome_writers/_backends/_yaozarrs.py:360(_finalize_chunk_buffers) │
│    20 │ 0.000s │     0.000s │ <stdlib>/typing.py:2371(cast)                                       │
└───────┴────────┴────────────┴─────────────────────────────────────────────────────────────────────┘
```

## Profiling with py-spy

You can also use [`py-spy`](https://github.com/benfred/py-spy).  You can add it to your
environment with `uv add --dev py-spy`.  For complete usage tips, see the
[`py-spy` readme](https://github.com/benfred/py-spy?tab=readme-ov-file#usage),
but here is a quick example of how to use it with ome-writers:

!!! tip
    py-spy will require sudo access on most systems to attach to the Python process.

Create a Python script that uses ome-writers to perform some operations,
such as one of the examples in the `examples/` directory.  For example:

```sh
sudo uv run py-spy record -o profile.svg -f flamegraph -- python examples/single_5d_image.py zarr-python
```

- the `-o profile` flag specifies the output file name
- the `-f flamgegraph` flag specifies the output format. Other options include `speedscope` and `raw`
- the `--` separates the `py-spy` arguments from the Python script to be profiled

This will create a `profile.svg` file that you can open in a web browser to
navigate the profiling results.

!!! important
    The structure of the script matters.  If you want to profile
    raw write performance, make sure that the frames are pre-generated outside
    of the append loop.  The example scripts do *not* do this, so you may see
    some amount of time spent in numpy frame generation interspersed with the
    backend write time.

### Understanding the flamegraph

`py-spy` is a *sampling* profiler, which means it periodically samples the call stack of the
running Python process.  (as opposed to a deterministic profiler like `cProfile` that records
every function call).  This means that the results will vary slightly each time you run it.

The flamegraph output shows the call stack over time, with the x-axis
representing time and the y-axis representing the call stack depth. Wider blocks
indicate functions that took more time to execute.  You can hover over blocks to
see more information about the function, including the number of samples
collected in that function.
