import xarray as xr
import numpy as np

def main():
    # ncfile = "dataset/weatherbench/constants.nc"
    # ds_xr = xr.open_dataset(ncfile)
    # print(ds_xr)
    # var_xr = ds_xr.variables["z"]
    # print(var_xr.values.shape)
    # print(var_xr.values[545][0][10])

    # print("\n")

    # npfile = np.load('dataset/processed/train/2015_0.npz')
    # print(npfile)
    # print(npfile.files)
    # print(npfile['geopotential_50'].shape)
    # print(npfile['geopotential_50'][545][0][10])


    # ncfile = "dataset/copernicus/geopotential/geopotential_50_250_2015_raw.nc"
    # ds_xr = xr.open_dataset(ncfile)
    # print(ds_xr)
    # var_xr = ds_xr.variables["z"]
    # print(var_xr.values.shape)
    # print(var_xr.values[24][0][10])

    # print("\n")

    npfile = np.load('dataset/processed/normalize_mean.npz')
    print(npfile)
    print(npfile.files)
    # print(npfile['geopotential_50'].shape)
    # print(npfile['geopotential_50'][0][0][10])


if __name__ == "__main__":
    main()
