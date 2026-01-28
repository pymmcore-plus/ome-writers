import pytest

from ome_writers._util import dims_from_useq

try:
    import useq
except ImportError:
    pytest.skip("useq not installed", allow_module_level=True)


def test_useq_to_dims() -> None:
    seq = useq.MDASequence(
        axis_order="tpcz",
        stage_positions=[(0.0, 0.0), (10.0, 10.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "Cy5"],
        z_plan={"range": 2, "step": 1.0},
    )

    dims = dims_from_useq(
        seq,
        image_width=64,
        image_height=64,
        units={"t": (0.1, "second")},
        pixel_size_um=0.103,
        chunk_shapes={"z": 2, "y": 32, "x": 32},
        shard_shapes={"y": 2, "x": 2},
    )
    assert [dim.name for dim in dims] == ["t", "p", "c", "z", "y", "x"]
    assert [dim.unit for dim in dims] == ["second", None, None] + ["micrometer"] * 3
    assert [dim.scale for dim in dims] == [0.1, 1, 1, 1.0, 0.103, 0.103]
    chunk_sizes = [getattr(dim, "chunk_size", None) for dim in dims]
    assert chunk_sizes == [None, None, None, 2, 32, 32]
    shard_sizes = [getattr(dim, "shard_size_chunks", None) for dim in dims]
    assert shard_sizes == [None, None, None, None, 2, 2]
