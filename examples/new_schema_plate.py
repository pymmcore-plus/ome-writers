"""Basic example of using ome_writers to write a multi-well plate."""

from ome_writers import schema_pydantic as schema

# every FOV will use the same array settings
array_settings = schema.dims_from_standard_axes(
    sizes={"t": 10, "c": 2, "z": 5, "y": 512, "x": 512},
    chunk_shapes={"y": 64, "x": 64},
)

# settings = schema.AcquisitionSettings(
#     root_path="output.ome.zarr",
#     plate=schema.Plate(
#         row_names=["A", "B", "C", "D"],
#         column_names=["1", "2", "3", "4", "5", "6", "7", "8"],
#         images=[
#             schema.FOV(row_index=r, column_index=c, array_settings=array_settings)
#             for (r, c) in product(range(2), range(3))
#         ],
#     ),
#     overwrite=True,
#     backend="auto",
# )


# stream = create_stream(settings)  # type: ignore
# for frame in data:
#     stream.append(frame)
