"""Datasets loading module."""
from .b3fd import get_b3fd_dataset
from .utkface import get_utkface_dataset

__all__ = ["get_b3fd_dataset", "get_utkface_dataset"]
