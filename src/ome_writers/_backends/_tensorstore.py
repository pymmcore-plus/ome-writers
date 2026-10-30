from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ome_writers._backends._yaozarrs import YaozarrsBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from tensorstore import Future

    from ome_writers._backends._chunk_buffer import ChunkBuffer


class TensorstoreBackend(YaozarrsBackend):
    """OME-Zarr writer using tensorstore via yaozarrs."""

    def _get_yaozarrs_writer(self) -> Literal["tensorstore"]:
        return "tensorstore"

    def __init__(self) -> None:
        super().__init__()
        self._futures: list[Future] = []

    def _write(self, array: Any, index: tuple[int, ...], frame: np.ndarray) -> None:
        """Write frame to array at specified index, async for tensorstore."""
        self._futures.append(array[index].write(frame))

    def _after_flush(self, buffer: ChunkBuffer, coords: tuple[int, ...]) -> None:
        """Release a chunk once its async write commits, so reads never miss it."""
        self._futures[-1].add_done_callback(lambda _: buffer.end_flush(coords))

    def _resize(self, array: Any, new_shape: Sequence[int]) -> None:
        """Resize array to new shape, using exclusive_max for tensorstore."""
        array.resize(exclusive_max=new_shape).result()

    def finalize(self) -> None:
        """Flush and release resources."""
        self._finalize_chunk_buffers()  # may queue writes, so it must run first
        while self._futures:
            self._futures.pop().result()
        super().finalize()
