# Notes on the #139 rework (TEMPORARY: delete before merging)

Branch `fix/tiff-multifile-metadata-rework` on tlambert03/ome-writers: Aaron's
#139 commit (`8bfce4c`), the rework on top, and merges of main (latest after
#143, which this branch conflicted with only in test imports).
Nothing has been pushed to Aaron's branch, and #139 on GitHub is untouched.

- Everything vs. current main: `git diff upstream/main HEAD -- src tests`
- Rework only, vs. Aaron's commit merged onto main:
  `git diff $(git merge-tree --write-tree upstream/main 8bfce4c) HEAD -- src tests`

Proposed, NOT decided yet: close #139 with a thank-you comment, and open a new
PR from this branch with Aaron's commit kept in history, so the squash merge
credits him as co-author (check the `Co-authored-by:` line when merging).
Mention "based on #139 by aaronkanzer" in the body. Claude also suggested
splitting the writer-timeout change (item 6 below) into its own PR; Talley
has not weighed in on that.

## The original bug (#139)

In multi-file OME-TIFF modes (`redundant`, `master-tiff`, `companion-file`),
`finalize` only corrected SizeT and PlaneCount for `images[0]` when the outer
dimension was unbounded or the run stopped early. Confirmed on main with 3
timepoints x 2 positions, unbounded t: the full model reported SizeT
companion `[1, 1]`, master `[3, 1]`, redundant `[3, 1]`. A bounded run that
stops early keeps the declared SizeT (e.g. 5).

Terms used below: the "full OME models" are the mirrors with
`model.binary_only is None` (found with `TiffBackend._global_target_pos_idxs()`):
every TIFF in redundant mode, p000 in master-tiff mode, the companion file in
companion mode. All other TIFFs hold BinaryOnly stubs.

Root cause: data is per position (one TIFF each) but the full metadata is not.
The old code updated "the position's own mirror", which only matches where
the metadata lives in single-file mode.

## Spec check (2026-11-08)

Spec: https://ome-model.readthedocs.io/en/stable/ome-tiff/specification.html
Key rules: embedded copies are "identical metadata apart from the UUID"; IFDs
beyond SizeZ*SizeC*SizeT "are extraneous and should be ignored"; planes not
covered by TiffData "are unspecified and hence lost" (readers zero-fill);
PlaneCount is NonNegativeInt (0 is valid); BinaryOnly stubs carry nothing else.

Scripts in the session scratchpad wrote 2 positions x 2 channels x unbounded t
(complete / stopped after 7 frames / B never got a frame) with frame metadata,
in all 3 modes, for main and for this branch, then checked those rules and
opened every file with Bio-Formats. Findings on main:

- Stopped runs: frames silently dropped by readers (companion: all positions;
  master: B; redundant: copies disagree, p001 has PlaneCount 3 > Z*C*T = 2 and
  a Plane with TheT >= SizeT, and Bio-Formats crashes opening it).
- Frame metadata wrong even in complete runs: companion had 0 annotations,
  master had A's only, redundant copies each had only their own position's.
- Empty position: every full copy claims planes in a 0-IFD file, Bio-Formats
  cannot open the dataset at all.

This branch passes every check in every scenario, except the 0-IFD TIFF left
for an empty position (see open decisions).

## What the rework changes (vs. Aaron's commit)

In `src/ome_writers/_backends/_tifffile.py` unless noted:

1. `finalize` writes every position's frames written into *every* full OME
   model, in one place on `TiffBackend` (removed `PositionManager.join` and
   `PositionManager.finalize`).
2. The outer dimension size rounds *up* (`-(-n // inner)`), so a partially
   written last timepoint is still declared. Rounding down (Aaron, main)
   either declares more planes than exist or makes readers ignore written
   frames. Matches `LiveTiffArray`.
3. Positions with zero frames get PlaneCount=0 and size 1 (Aaron's version
   skipped them, leaving the declared size).
4. `_append_frame_metadata` writes the per-frame `Plane`/`MapAnnotation` into
   every full OME model, not the position's own (possibly stub) model.
   Removed dead `self._frame_metadata`.
5. `PositionManager.flush()` skips a TIFF with no frames (`tiffcomment` raises
   IndexError on a 0-IFD file) or one whose writer thread is still alive. In
   Aaron's version an empty position in redundant mode crashed `finalize`.
6. Writer threads: was `join(timeout=5)`, which on a long but healthy backlog
   gave up and then raced `tiffcomment` against the running writer. Now
   `_join_writers` waits in `WRITER_STALL_TIMEOUT` (60 s) slices while
   `frames_written` keeps increasing, gives up only on a writer that wrote
   nothing for a whole slice, flushes everything else, and raises
   `TimeoutError` naming the stalled files.
7. `finalize` returns early if there are no position managers (`create_stream`
   calls `backend.finalize()` after `prepare` raises FileExistsError), and sets
   `_finalized` in a `finally`.
8. `finalize` serializes the full OME models once and swaps in each file's
   root UUID, reusing #141's `initial_tiff_xml` (renamed `tiff_xmls` in
   `_ome_xml.py`; `OmeXMLMirror.flush` and `PositionManager.flush` take an
   optional pre-serialized `xml`). Valid only while the full models differ by
   root UUID alone: true from `prepare` until `finalize`, NOT after a user
   `update_metadata`, so post-finalize writes still serialize each model.

9. `finalize` only marks a full model changed (and so rewrites it) when a frame
   count or size actually differs, or something else already changed it (frame
   metadata, global metadata). Complete acquisitions no longer rewrite the
   OME-XML at close; main always did for bounded ones. The companion mirror
   starts dirty in `prepare`, since no writer thread writes its initial XML.
   Removed the dead `PositionManager._metadata_dirty`. `OmeXMLMirror` gained a
   read-only `dirty` property.

Tests (`tests/test_tiff_metadata.py`):

- `_get_full_ome` in redundant mode reads *every* TIFF with frames, asserts
  they match (minus UUID), and that each file's root UUID is the one its
  TiffData references (checked to fail if the UUID swap is skipped or replaces
  every occurrence).
- `test_multiposition_frames_written` replaces `test_unbounded_multiposition_sizes`:
  all 3 modes x (complete timepoints, partial timepoint, empty position,
  bounded but stopped early).
- `test_frame_metadata_multiposition` runs in all 3 modes and checks each
  plane references its own frame's annotation.
- `test_finalize_waits_for_slow_writer` / `test_finalize_abandons_stalled_writer`
  monkeypatch `tifffile.TiffWriter.write` and `_tifffile.WRITER_STALL_TIMEOUT`.
  Both fail against the broken variant they target.
- `test_initial_tiff_xml` renamed `test_tiff_xmls`.

## Benchmarks on main (#144), local, min of rounds (what CodSpeed should show)

| benchmark | main | this branch |
|---|---|---|
| acquisition[16_positions-tifffile] | 128.8 ms | 56.9 ms |
| acquisition[unbounded_t-tifffile] | 10.4 ms | 10.6 ms |
| append[3d_chunking-tifffile] | 6.1 ms | 4.6 ms |
| append[large_2d-tifffile] | 1.0 ms | 0.9 ms |
| append[t5_c2_z3_xy256-tifffile] | 2.2 ms | 0.9 ms |
| append[time_lapse_small_2d-tifffile] | 5.3 ms | 3.9 ms |

(Before item 9, `large_2d-tifffile` was 2.45x slower than main: it rewrote
unchanged OME-XML at close.) On CI, 16_positions-tifffile is ~1.3 s on main,
10x slower than locally, mostly waiting rather than CPU; unexplained (GIL
contention between writer threads and the main thread is the guess).

## Performance (96 positions x 10 timepoints, 32x32 px frames, close time)

| mode, frame metadata | main | rework before item 8 | this branch |
|---|---|---|---|
| redundant, no        | 1.93 s | 1.91 s  | 0.10 s |
| redundant, yes       | 2.15 s | 13.28 s | 0.20 s |
| master-tiff, yes     | 0.08 s | 0.18 s  | 0.18 s |
| companion-file, yes  | 0.07 s | 0.19 s  | 0.18 s |

Master/companion are slower only because they now write frame metadata main
dropped. Redundant + frame metadata files grow 10.6 MB -> 32.4 MB (pixels are
only 2 MB at 32x32; the spec itself warns embedded per-plane metadata gets
large). Appending costs ~50 us more per frame at 96 positions (each plane goes
into every full model; caching `_global_target_pos_idxs()` would help).
Create time in redundant mode (~1.1 s) is the per-file deep copy in
`prepare_metadata`, unchanged.

## Trade-offs (not strictly better than Aaron's)

1. Rounding up declares planes that were never written (readers zero-fill).
2. Writes to an empty or still-open TIFF are skipped silently, including user
   `update_metadata` / `set_global_metadata` after close. Before, they raised.
   Undecided whether this should warn.
3. Dropped defensive checks (`pos_idx >= len(images)`, `if tiff_data_blocks`).
   Safe because user `update_metadata` is only allowed after `finalize`.
4. Bigger change than Aaron's minimal fix; `close()` can now raise
   `TimeoutError`.
5. `_tifffile.py` is 635 lines (main 624).

## Open decisions / not done

- Empty positions: Talley leans toward deleting the file (not final; would be
  a separate PR after this one). Measured on this branch's output: keeping the
  0-IFD file,
  tifffile raises KeyError opening p000; deleting the file *alone* makes
  Bio-Formats refuse the whole dataset ("Missing file ... associated with
  UUID"); a placeholder frame makes tifffile show a fake all-zero image.
  Deleting the file AND dropping that position's Image from every full model
  reads cleanly in both. Empty positions are always a suffix of the position
  order, so remaining Image indices don't shift. Still to handle: plate
  WellSample ImageRefs to dropped Images, `get_arrays()` / `get_metadata()` /
  `update_metadata()` for dropped positions, a delete failing on Windows (file
  open elsewhere), and the zero-frames-at-all case.
- `OMEStream.close()` (`src/ome_writers/_stream.py`) does not shut down the
  callback executor if `backend.finalize()` raises. Pre-existing, but the
  `TimeoutError` adds a way to hit it. A `try/finally` would fix it.
- `WRITER_STALL_TIMEOUT` is a module constant, not user-configurable.

## Related PRs

- #140 (merged): companion file named `{stem}.companion.ome` by default
  (Bio-Formats needs the `.companion.ome` suffix). Added `tests/_bf_reader.py`.
- #141 (merged): redundant mode serializes the full OME-XML once at `prepare`
  (384 positions: create_stream 44 s -> 5 s). Item 8 reuses it at close.
- #143 (merged): BinaryOnly stubs now name their metadata file instead of
  its absolute path, so Bio-Formats follows them. Its Bio-Formats test opens
  every file of a dataset, stubs included.
- #144 (merged): benchmarks write to /dev/shm (disk made rounds bimodal), and
  `test_bench_acquisition` times whole acquisitions with frame metadata.

## Follow-ups

- `_assert_bioformats_reads_ome_tiff` (`tests/test_integration.py`) globs
  `output_path.parent`, so multi-position outputs are never checked with
  Bio-Formats.
- `prepare_metadata` still deep-copies the model per file in redundant mode.
- Post-close `set_global_metadata` in redundant mode still serializes each
  file's model (O(n^2)); could use `tiff_xmls` if models are known identical.
- Local branch `fix/companion-ome-suffix` (merged as #140) can be deleted.

## Gotchas for whoever picks this up

- Running tests in a git worktree with the main checkout's venv imports
  `ome_writers` from the main checkout (editable install). Use
  `PYTHONPATH=$PWD/src uv run pytest ...`.
- Bio-Formats checks: `uv run --group bioformats pytest ...`; scyjava fetches
  a JDK and Bio-Formats on first use. CI runs these in the `bioformats` matrix
  group, which can fail on a maven.scijava.org timeout (infra, not code).
- `read_series_with_bioformats` returns the reader class as a Java string:
  `.split(".")` is Java's regex split, use `str(reader)` first.
- The `typos` pre-commit hook rejects some made-up identifiers and short commit
  hashes (it flagged a test variable name and `git diff <short sha>` here).
