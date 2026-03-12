import os
import psutil
import numpy as np
from enum import Enum
from dataclasses import dataclass

# CPU threads configuration
CPU_COUNT = psutil.cpu_count(logical=True)
os.environ['OPENBLAS_NUM_THREADS'] = str(CPU_COUNT)
os.environ['MKL_NUM_THREADS'] = str(CPU_COUNT)


# -- Configuration Limits and Thresholds --
MAX_MATRIX_DIM = 1000000
VIRTUAL_MODE_THRESHOLD = 1000000
CACHE_MAX_ITEMS = 100
CACHE_MAX_MB = 1024
SCROLL_DEBOUNCE_MS = 50
SIZE_CHANGE_DELAY_MS = 300
CONDITION_THRESHOLD_WARN = 1e12
PROGRESS_UPDATE_INTERVAL = 100
MAX_FILE_SIZE_MB = 100  # Limit for file uploads
CACHE_FULL_HASH_MAX_ELEMS = 10000  # For small matrices, hash entire array


# -- Configuration Enums and Data Classes --
class ComputePrecision(Enum):
    AUTO = "auto"
    FP64 = "float64"
    FP32 = "float32"


class MatrixStructure(Enum):
    UNKNOWN = "unknown"
    DENSE = "dense"
    SPARSE = "sparse"
    DIAGONAL = "diagonal"


class ComputeDevice(Enum):
    CPU = "cpu"
    GPU_CUDA = "gpu_cuda"


@dataclass
class MatrixInfo:
    rows: int
    cols: int
    dtype: np.dtype
    structure: MatrixStructure
    sparsity: float
    condition_number: float = 1.0
    memory_mb: float = 0.0
    is_square: bool = False
    recommended_device: ComputeDevice = ComputeDevice.CPU


class Config:
    """Configuration class containing all app settings"""
    
    MAX_MATRIX_DIM = MAX_MATRIX_DIM
    VIRTUAL_MODE_THRESHOLD = VIRTUAL_MODE_THRESHOLD
    CACHE_MAX_ITEMS = CACHE_MAX_ITEMS
    CACHE_MAX_MB = CACHE_MAX_MB
    SCROLL_DEBOUNCE_MS = SCROLL_DEBOUNCE_MS
    SIZE_CHANGE_DELAY_MS = SIZE_CHANGE_DELAY_MS
    CONDITION_THRESHOLD_WARN = CONDITION_THRESHOLD_WARN
    PROGRESS_UPDATE_INTERVAL = PROGRESS_UPDATE_INTERVAL
    MAX_FILE_SIZE_MB = MAX_FILE_SIZE_MB
    CACHE_FULL_HASH_MAX_ELEMS = CACHE_FULL_HASH_MAX_ELEMS
    
    PRECISION_OPTIONS = [ComputePrecision.AUTO.value, ComputePrecision.FP64.value, ComputePrecision.FP32.value]
    
    @classmethod
    def is_within_limits(cls, matrix_rows: int, matrix_cols: int) -> bool:
        """Check if matrix dimensions are within configured limits"""
        max_dim = max(matrix_rows, matrix_cols)
        return max_dim <= cls.MAX_MATRIX_DIM
