from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ome_writers._backends._yaozarrs import YaozarrsBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from tensorstore import Future


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

    def _resize(self, array: Any, new_shape: Sequence[int]) -> None:
        """Resize array to new shape, using exclusive_max for tensorstore."""
        array.resize(exclusive_max=new_shape).result()

    def finalize(self) -> None:
        """Flush and release resources."""
        while self._futures:
            self._futures.pop().result()
        super().finalize()
