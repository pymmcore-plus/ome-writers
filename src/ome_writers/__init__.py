"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ome-writers")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"
