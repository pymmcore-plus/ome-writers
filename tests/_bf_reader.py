from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import scyjava
import scyjava.config
from scyjava import jimport

scyjava.config.endpoints.append("ome:formats-gpl:RELEASE")
# suppress slf4j logging output
scyjava.config.endpoints.append("org.slf4j:slf4j-nop:2.0.9")


@cache
def ome_xml_service() -> Any:
    ServiceFactory = jimport("loci.common.services.ServiceFactory")
    OMEXMLService = jimport("loci.formats.services.OMEXMLService")
    return ServiceFactory().getInstance(OMEXMLService)


@dataclass
class CoreMeta:
    size_t: int
    size_c: int
    effective_size_c: int
    size_z: int
    size_y: int
    size_x: int
    rgb_channel_count: int
    dtype: np.dtype
    series_count: int
    dimension_order: int
    resolution_count: int
    is_rgb: int
    is_interleaved: int
    format_name: str
    reader_class: str


def pixtype2dtype(pixeltype: int, little_endian: bool) -> np.dtype:
    """Convert a loci.formats PixelType integer into a numpy dtype."""
    FormatTools = scyjava.jimport("loci.formats.FormatTools")

    fmt2type: dict[int, str] = {
        FormatTools.INT8: "i1",
        FormatTools.UINT8: "u1",
        FormatTools.INT16: "i2",
        FormatTools.UINT16: "u2",
        FormatTools.INT32: "i4",
        FormatTools.UINT32: "u4",
        FormatTools.FLOAT: "f4",
        FormatTools.DOUBLE: "f8",
    }
    return np.dtype(("<" if little_endian else ">") + fmt2type[pixeltype])


def read_core_meta_with_bioformats(path: str) -> CoreMeta:
    """Read an image using Bio-Formats via scyjava."""
    ImageReader = jimport("loci.formats.ImageReader")
    reader = ImageReader()
    metadata_store = ome_xml_service().createOMEXMLMetadata()
    reader.setMetadataStore(metadata_store)
    reader.setId(path)
    reader.setSeries(0)

    meta = CoreMeta(
        size_t=reader.getSizeT(),
        effective_size_c=reader.getEffectiveSizeC(),
        size_c=reader.getSizeC(),
        size_z=reader.getSizeZ(),
        size_y=reader.getSizeY(),
        size_x=reader.getSizeX(),
        rgb_channel_count=reader.getRGBChannelCount(),
        dtype=pixtype2dtype(reader.getPixelType(), reader.isLittleEndian()),
        series_count=reader.getSeriesCount(),
        dimension_order=reader.getDimensionOrder(),
        resolution_count=reader.getResolutionCount(),
        is_rgb=reader.isRGB(),
        is_interleaved=reader.isInterleaved(),
        format_name=reader.getFormat(),
        reader_class=reader.getReader().getClass().getSimpleName(),
    )

    reader.close()
    return meta
