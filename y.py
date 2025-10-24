import ndv
import zarr

path = "/Users/fdrgsp/Desktop/acq_z.zarr/"


gp = zarr.open_group(path, mode="r")
print(gp.tree())  # Print the structure of the Zarr store


d = gp["96-well"]
f0 = d["A"]["01"]["fov0"]
print(f0.shape)


print(round(f0[0, 0].std(), 2))
print(round(f0[1, 0].std(), 2))
print(round(f0[2, 0].std(), 2))

print(round(f0[0, 1].std(), 2))
print(round(f0[1, 1].std(), 2))
print(round(f0[2, 1].std(), 2))


ndv.imshow(f0)


