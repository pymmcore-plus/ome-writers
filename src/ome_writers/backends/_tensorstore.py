from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ome_writers.backends._yaozarrs import YaozarrsBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from ome_writers.schema import AcquisitionSettings


class TensorstoreBackend(YaozarrsBackend):
    """OME-Zarr writer using tensorstore via yaozarrs."""

    def _get_writer(self) -> Literal["tensorstore"]:
        return "tensorstore"

    def __init__(self) -> None:
        super().__init__()
        self._futures: list[Any] = []

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        if not settings.root_path.endswith(".zarr"):  # pragma: no cover
            return "Root path must end with .zarr for TensorstoreBackend."
        return False

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
