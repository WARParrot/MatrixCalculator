import math
import numpy as np


def validate_number(value_str: str):
    """Validate if a string is a valid number and convert it."""
    try:
        val = float(value_str)
        if math.isinf(val) or math.isnan(val):
            return None, False
        return val, True
    except ValueError:
        return None, False


def to_ndarray(matrix_data):
    """Convert a list of lists to a NumPy array."""
    if not matrix_data:
        return np.array([])
    return np.array(matrix_data, dtype=float)


def get_system_info():
    """Return info about available GPU."""
    info = {"has_gpu": False}
    try:
        import cupy as cp
        info["has_gpu"] = True
        info["cupy_version"] = cp.__version__
    except ImportError:
        pass
    return info
