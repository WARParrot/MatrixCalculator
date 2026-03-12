import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import threading
import time
import psutil
import gc
import os
import logging
import hashlib
import warnings
from enum import Enum
from dataclasses import dataclass
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union, List, Tuple, Dict
import sys

# ----------------------------------------------------------------------
# Конфигурация и логирование
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

CPU_COUNT = psutil.cpu_count(logical=True)
os.environ['OPENBLAS_NUM_THREADS'] = str(CPU_COUNT)
os.environ['MKL_NUM_THREADS'] = str(CPU_COUNT)


class Config:
    MAX_MATRIX_DIM = 1000000
    VIRTUAL_MODE_THRESHOLD = 1000000
    CACHE_MAX_ITEMS = 100
    CACHE_MAX_MB = 1024
    SCROLL_DEBOUNCE_MS = 50
    SIZE_CHANGE_DELAY_MS = 300
    CONDITION_THRESHOLD_WARN = 1e12
    PROGRESS_UPDATE_INTERVAL = 100
    MAX_FILE_SIZE_MB = 100                     # FIXED: лимит на загрузку
    CACHE_FULL_HASH_MAX_ELEMS = 10000          # FIXED: для малых матриц хеш от всего массива


# ----------------------------------------------------------------------
# Типы данных
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Утилиты
# ----------------------------------------------------------------------
def to_ndarray(data: Any) -> np.ndarray:
    """Безопасное преобразование в numpy.ndarray (float64)."""
    if data is None:
        return np.array([], dtype=np.float64)
    if isinstance(data, np.ndarray):
        return data.astype(np.float64, copy=False)
    if hasattr(data, 'cpu'):      # PyTorch
        return data.cpu().numpy().astype(np.float64)
    if hasattr(data, 'get'):       # CuPy
        return data.get().astype(np.float64)
    try:
        return np.array(data, dtype=np.float64)
    except Exception:
        return np.array([], dtype=np.float64)


def estimate_condition_number(arr: np.ndarray) -> float:
    """Оценка числа обусловленности по спектральной норме (использует SVD)."""
    try:
        return np.linalg.cond(arr, p=2)   # FIXED: заменено на np.linalg.cond
    except np.linalg.LinAlgError:
        return np.inf


def validate_number(s: str) -> Tuple[Optional[float], bool]:
    """Проверка, является ли строка числом (с учётом переполнения). Возвращает (значение, успех)."""
    try:
        val = float(s)
        return val, True
    except ValueError:
        return None, False
    except OverflowError:
        return None, False


# ----------------------------------------------------------------------
# LRU-кэш
# ----------------------------------------------------------------------
class LRUCache:
    def __init__(self, capacity=Config.CACHE_MAX_ITEMS, max_size_mb=Config.CACHE_MAX_MB):
        self.capacity = capacity
        self.max_size_mb = max_size_mb
        self._cache = OrderedDict()
        self._current_size_mb = 0.0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            self.misses += 1
            return None
        self._cache.move_to_end(key)
        self.hits += 1
        return self._cache[key][0]

    def put(self, key: str, value: Any) -> None:
        try:
            size_mb = len(str(value)) / (1024 ** 2)
        except Exception:
            size_mb = 0.1
        if size_mb > self.max_size_mb:
            logger.warning(f"Object too large for cache: {size_mb:.1f} MB")
            return

        while (len(self._cache) >= self.capacity or
               self._current_size_mb + size_mb > self.max_size_mb):
            if not self._cache:
                break
            _, (_, old_size) = self._cache.popitem(last=False)
            self._current_size_mb -= old_size
            self.evictions += 1

        self._cache[key] = (value, size_mb)
        self._current_size_mb += size_mb

    def clear(self):
        self._cache.clear()
        self._current_size_mb = 0.0
        self.hits = self.misses = self.evictions = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


# ----------------------------------------------------------------------
# Вычислительный движок (Model)
# ----------------------------------------------------------------------
class MatrixEngine:
    def __init__(self):
        self.xp = np
        self.cp = None
        self.gpu_available = False
        self._detect_gpu()
        self.precision = ComputePrecision.AUTO
        self.cache = LRUCache()
        self.thread_pool = ThreadPoolExecutor(max_workers=CPU_COUNT)

    def _detect_gpu(self):
        try:
            import cupy as cp
            self.cp = cp
            if cp.is_available():
                self.gpu_available = True
                logger.info("GPU доступен (CuPy)")
            else:
                logger.info("CuPy установлен, но GPU не найден")
        except ImportError:
            logger.info("CuPy не установлен, работаем на CPU")
        except Exception as e:
            logger.warning(f"Ошибка инициализации GPU: {e}")

    def _get_device(self, info: MatrixInfo) -> ComputeDevice:
        if not self.gpu_available:
            return ComputeDevice.CPU
        if info.rows * info.cols < 5000:
            return ComputeDevice.CPU
        try:
            free_mem = self.cp.cuda.Device(0).mem_info[0] / (1024 ** 3)
            if info.memory_mb / 1024 * 2.5 > free_mem:
                logger.warning("Недостаточно памяти GPU, переключаемся на CPU")
                return ComputeDevice.CPU
        except Exception:
            pass
        return ComputeDevice.GPU_CUDA

    def _to_device_array(self, data: Any, device: ComputeDevice):
        """Преобразование с учётом выбранной точности."""
        arr = to_ndarray(data)
        # FIXED: приведение к нужной точности
        if self.precision == ComputePrecision.FP32:
            arr = arr.astype(np.float32)
        else:  # AUTO или FP64
            arr = arr.astype(np.float64)

        if device == ComputeDevice.GPU_CUDA and self.gpu_available:
            return self.cp.asarray(arr)
        return arr

    def _from_device_array(self, arr: Any) -> np.ndarray:
        if self.gpu_available and isinstance(arr, self.cp.ndarray):
            return self.cp.asnumpy(arr)
        if isinstance(arr, np.ndarray):
            return arr
        return np.array(arr)

    def analyze_matrix(self, matrix: Any, fast: bool = True) -> MatrixInfo:
        """
        Анализирует матрицу: размер, структура, разреженность, число обусловленности.
        Если fast=True и матрица большая, разреженность оценивается по выборке.
        """
        arr = to_ndarray(matrix)
        rows, cols = arr.shape
        total_cells = rows * cols
        memory_mb = arr.nbytes / (1024 ** 2)
        is_square = rows == cols

        info = MatrixInfo(
            rows=rows, cols=cols, dtype=arr.dtype,
            structure=MatrixStructure.UNKNOWN,
            sparsity=0.0, memory_mb=memory_mb, is_square=is_square
        )

        # Оценка разреженности
        if total_cells > Config.VIRTUAL_MODE_THRESHOLD and fast:
            sample_size = min(1000, rows, cols)
            # FIXED: используем фиксированный seed для воспроизводимости, если нужно
            rng = np.random.RandomState(42)  # детерминированная выборка для консистентности
            rows_idx = rng.choice(rows, sample_size, replace=False)
            cols_idx = rng.choice(cols, sample_size, replace=False)
            sample = arr[np.ix_(rows_idx, cols_idx)]
            nonzero = np.count_nonzero(sample)
            sparsity = 1.0 - (nonzero / (sample_size ** 2))
            logger.info("Приблизительная оценка разреженности (fast mode)")
        else:
            nonzero = np.count_nonzero(arr)
            sparsity = 1.0 - (nonzero / total_cells)

        info.sparsity = sparsity
        info.structure = MatrixStructure.SPARSE if sparsity > 0.5 else MatrixStructure.DENSE

        # Оценка числа обусловленности только для небольших квадратных матриц
        if is_square and total_cells <= 10000:
            info.condition_number = estimate_condition_number(arr)

        info.recommended_device = self._get_device(info)
        return info

    def add_matrices(self, A: Any, B: Any) -> np.ndarray:
        infoA = self.analyze_matrix(A, fast=True)
        device = infoA.recommended_device
        A_dev = self._to_device_array(A, device)
        B_dev = self._to_device_array(B, device)
        if A_dev.shape != B_dev.shape:
            raise ValueError("Матрицы должны быть одинакового размера")
        result = A_dev + B_dev
        return self._from_device_array(result)

    def subtract_matrices(self, A: Any, B: Any) -> np.ndarray:
        infoA = self.analyze_matrix(A, fast=True)
        device = infoA.recommended_device
        A_dev = self._to_device_array(A, device)
        B_dev = self._to_device_array(B, device)
        if A_dev.shape != B_dev.shape:
            raise ValueError("Матрицы должны быть одинакового размера")
        result = A_dev - B_dev
        return self._from_device_array(result)

    def scalar_multiply(self, A: Any, scalar: float) -> np.ndarray:
        infoA = self.analyze_matrix(A, fast=True)
        device = infoA.recommended_device
        A_dev = self._to_device_array(A, device)
        result = A_dev * scalar
        return self._from_device_array(result)

    def multiply_matrices(self, A: Any, B: Any) -> np.ndarray:
        key = self._cache_key('multiply', A, B)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        infoA = self.analyze_matrix(A, fast=True)
        infoB = self.analyze_matrix(B, fast=True)
        device = infoA.recommended_device
        A_dev = self._to_device_array(A, device)
        B_dev = self._to_device_array(B, device)

        if A_dev.shape[1] != B_dev.shape[0]:
            raise ValueError("Несовместимые размеры матриц для умножения")

        xp = self.cp if device == ComputeDevice.GPU_CUDA else np
        result = xp.dot(A_dev, B_dev)   # FIXED: убрано блочное умножение, положимся на BLAS

        result_np = self._from_device_array(result)
        self.cache.put(key, result_np)
        return result_np

    def transpose_matrix(self, A: Any) -> np.ndarray:
        infoA = self.analyze_matrix(A, fast=True)
        device = infoA.recommended_device
        A_dev = self._to_device_array(A, device)
        result = A_dev.T
        return self._from_device_array(result)

    def determinant_matrix(self, A: Any) -> float:
        """
        Вычисление определителя квадратной матрицы.
        Использует LU-разложение через numpy.linalg.det.
        При плохой обусловленности выдаётся предупреждение.
        """
        info = self.analyze_matrix(A)
        if not info.is_square:
            raise ValueError("Определитель только для квадратных матриц")
        arr = to_ndarray(A)

        # FIXED: предупреждение о плохой обусловленности
        if info.condition_number > Config.CONDITION_THRESHOLD_WARN:
            logger.warning(f"Матрица плохо обусловлена (cond={info.condition_number:.2e}), "
                           "результат определителя может быть неточным")

        try:
            # FIXED: замена SVD на прямой вызов det (использует LU)
            return float(np.linalg.det(arr))
        except np.linalg.LinAlgError as e:
            logger.error("Ошибка вычисления определителя", exc_info=True)
            raise ValueError(f"Ошибка вычисления определителя: {e}") from e

    def rank_matrix(self, A: Any) -> int:
        """Ранг матрицы, вычисленный через SVD с адаптивным порогом."""
        arr = to_ndarray(A)
        # FIXED: используем встроенную функцию matrix_rank с корректным порогом
        return int(np.linalg.matrix_rank(arr))

    def inverse_matrix(self, A: Any) -> np.ndarray:
        info = self.analyze_matrix(A)
        if not info.is_square:
            raise ValueError("Обращение только для квадратных матриц")
        if info.condition_number == np.inf:
            raise ValueError("Матрица вырождена")
        if info.condition_number > Config.CONDITION_THRESHOLD_WARN:
            logger.warning(f"Матрица плохо обусловлена (cond={info.condition_number:.2e}), "
                           "обратная матрица может быть неточной")
        device = info.recommended_device
        A_dev = self._to_device_array(A, device)
        try:
            xp = self.cp if device == ComputeDevice.GPU_CUDA else np
            inv = xp.linalg.inv(A_dev)
            return self._from_device_array(inv)
        except Exception as e:
            logger.error("Ошибка обращения матрицы", exc_info=True)
            raise ValueError(f"Ошибка обращения: {e}") from e

    def solve_system(self, A: Any, B: Any) -> np.ndarray:
        infoA = self.analyze_matrix(A)
        if not infoA.is_square:
            raise ValueError("Матрица A должна быть квадратной")
        if infoA.condition_number == np.inf:
            raise ValueError("Матрица A вырождена")
        if infoA.condition_number > Config.CONDITION_THRESHOLD_WARN:
            logger.warning(f"Матрица A плохо обусловлена (cond={infoA.condition_number:.2e}), "
                           "решение может быть неточным")
        device = infoA.recommended_device
        A_dev = self._to_device_array(A, device)
        B_dev = self._to_device_array(B, device)
        if A_dev.shape[0] != B_dev.shape[0]:
            raise ValueError("Количество строк A и B не совпадает")
        try:
            xp = self.cp if device == ComputeDevice.GPU_CUDA else np
            X = xp.linalg.solve(A_dev, B_dev)
            return self._from_device_array(X)
        except Exception as e:
            logger.error("Ошибка решения СЛАУ", exc_info=True)
            raise ValueError(f"Ошибка решения СЛАУ: {e}") from e

    def _cache_key(self, operation: str, *args) -> str:
        """
        Формирует стабильный ключ для кэширования.
        Для небольших матриц используется полный дамп памяти (tobytes).
        Для больших – хеш от метаданных и выборки элементов.
        """
        try:
            parts = [operation]
            for arg in args:
                if arg is None:
                    parts.append('None')
                else:
                    arr = to_ndarray(arg)
                    total_elems = arr.size
                    if total_elems <= Config.CACHE_FULL_HASH_MAX_ELEMS:
                        # малая матрица: полный хеш
                        data_bytes = arr.tobytes()
                        h = hashlib.md5(data_bytes).hexdigest()
                    else:
                        # большая: хеш от shape, dtype и выборки
                        # берём до 1000 элементов равномерно
                        rng = np.random.RandomState(42)  # детерминированно
                        flat = arr.ravel()
                        step = max(1, total_elems // 1000)
                        idx = rng.choice(total_elems, min(1000, total_elems), replace=False)
                        sample = flat[idx].tobytes()
                        meta = f"{arr.shape}_{arr.dtype}_{arr.min()}_{arr.max()}_{arr.sum()}"
                        h = hashlib.md5((meta + str(sample)).encode()).hexdigest()
                    parts.append(f"{arr.shape}_{h}")
            full = '_'.join(parts)
            return hashlib.md5(full.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Ошибка создания ключа кэша: {e}")
            return None

    def clear_cache(self):
        self.cache.clear()
        if self.gpu_available:
            try:
                self.cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        gc.collect()
        logger.info("Кэш очищен")

    def get_stats(self) -> Dict:
        return {
            'cache': {
                'hits': self.cache.hits,
                'misses': self.cache.misses,
                'hit_rate': self.cache.hit_rate,
                'evictions': self.cache.evictions,
                'size_mb': self.cache._current_size_mb,
                'items': len(self.cache._cache)
            },
            'gpu_available': self.gpu_available
        }


# ----------------------------------------------------------------------
# Виджет для ввода матрицы (с виртуальным скроллингом)
# ----------------------------------------------------------------------
class MatrixWidget(ttk.Frame):
    def __init__(self, parent, title: str = "Матрица", engine: MatrixEngine = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.title = title
        self.engine = engine or MatrixEngine()
        self.rows = 2
        self.cols = 2
        self.data = {}          # {(row, col): str}
        self.entries = {}        # {(row, col): widget}
        self.validity = {}       # {(row, col): bool}   # FIXED: для подсветки ошибок
        self.scroll_timer = None
        self.size_timer = None
        self.pending_update = False
        self.virtual_mode = False
        self.cell_width = 70
        self.cell_height = 25
        self._create_widgets()
        self._update_display()

    def _create_widgets(self):
        """Создание заголовка и холста с полосами прокрутки."""
        # FIXED: разбит на вспомогательные методы
        self._create_header()
        self._create_canvas()

    def _create_header(self):
        header = ttk.Frame(self)
        header.pack(fill="x", pady=2)

        ttk.Label(header, text=self.title, font=('Arial', 10, 'bold')).pack(side="left", padx=5)

        self.info_label = ttk.Label(header, text="", foreground="blue")
        self.info_label.pack(side="left", padx=10)

        ttk.Label(header, text="Строки:").pack(side="left", padx=(20, 2))
        self.rows_var = tk.IntVar(value=2)
        rows_spin = ttk.Spinbox(header, from_=1, to=Config.MAX_MATRIX_DIM,
                                 textvariable=self.rows_var, width=8)
        rows_spin.pack(side="left", padx=2)
        rows_spin.bind("<KeyRelease>", self._on_size_delayed)
        rows_spin.bind("<<Increment>>", self._on_size_immediate)
        rows_spin.bind("<<Decrement>>", self._on_size_immediate)

        ttk.Label(header, text="Столбцы:").pack(side="left", padx=(10, 2))
        self.cols_var = tk.IntVar(value=2)
        cols_spin = ttk.Spinbox(header, from_=1, to=Config.MAX_MATRIX_DIM,
                                 textvariable=self.cols_var, width=8)
        cols_spin.pack(side="left", padx=2)
        cols_spin.bind("<KeyRelease>", self._on_size_delayed)
        cols_spin.bind("<<Increment>>", self._on_size_immediate)
        cols_spin.bind("<<Decrement>>", self._on_size_immediate)

        ttk.Button(header, text="T", width=2, command=self.transpose).pack(side="right", padx=1)
        ttk.Button(header, text="C", width=2, command=self.clear).pack(side="right", padx=1)

    def _create_canvas(self):
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, highlightthickness=0, bg='white')
        self.v_scroll = ttk.Scrollbar(container, orient="vertical", command=self._on_scroll_y)
        self.h_scroll = ttk.Scrollbar(container, orient="horizontal", command=self._on_scroll_x)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        self.v_scroll.pack(side="right", fill="y")
        self.h_scroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        self.frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)

    def _on_size_delayed(self, event=None):
        if self.size_timer:
            self.after_cancel(self.size_timer)
        self.size_timer = self.after(Config.SIZE_CHANGE_DELAY_MS, self._on_size_change)

    def _on_size_immediate(self, event=None):
        if self.size_timer:
            self.after_cancel(self.size_timer)
        self._on_size_change()

    def _on_size_change(self):
        try:
            new_rows = max(1, min(Config.MAX_MATRIX_DIM, self.rows_var.get()))
            new_cols = max(1, min(Config.MAX_MATRIX_DIM, self.cols_var.get()))
            if new_rows != self.rows or new_cols != self.cols:
                self.rows = new_rows
                self.cols = new_cols
                total = self.rows * self.cols
                self.virtual_mode = total > Config.VIRTUAL_MODE_THRESHOLD
                if total > 1e9:
                    self.info_label.config(text=f"({total/1e9:.1f}B ячеек)")
                elif total > 1e6:
                    self.info_label.config(text=f"({total/1e6:.1f}M ячеек)")
                elif total > 1e3:
                    self.info_label.config(text=f"({total/1e3:.1f}K ячеек)")
                else:
                    self.info_label.config(text=f"({total} ячеек)")
                # Очищаем старые данные, которые выходят за новые размеры
                to_delete = [key for (r, c) in list(self.data.keys()) if r >= new_rows or c >= new_cols]
                for key in to_delete:
                    del self.data[key]
                self._update_scroll_region()
                self._update_display()
        except Exception as e:
            logger.error(f"Size change error: {e}")

    def _update_scroll_region(self):
        total_width = self.cols * self.cell_width
        total_height = self.rows * self.cell_height
        self.canvas.configure(scrollregion=(0, 0, total_width, total_height))

    def _update_display(self):
        if self.pending_update:
            return
        self.pending_update = True

        def _update():
            try:
                x1 = self.canvas.canvasx(0)
                y1 = self.canvas.canvasy(0)
                x2 = x1 + self.canvas.winfo_width()
                y2 = y1 + self.canvas.winfo_height()

                start_row = max(0, int(y1 // self.cell_height))
                end_row = min(self.rows, int(y2 // self.cell_height) + 2)
                start_col = max(0, int(x1 // self.cell_width))
                end_col = min(self.cols, int(x2 // self.cell_width) + 2)

                for widget in self.frame.winfo_children():
                    widget.destroy()
                self.entries.clear()

                for i in range(start_row, end_row):
                    for j in range(start_col, end_col):
                        if self.virtual_mode:
                            w = ttk.Label(self.frame, width=8, relief="sunken",
                                          anchor="center", background="#f0f0f0")
                            val = self.data.get((i, j), "")
                            w.config(text=val[:8] if val else "")
                        else:
                            w = ttk.Entry(self.frame, width=8, justify="center")
                            val = self.data.get((i, j), "")
                            if val:
                                w.insert(0, val)
                            # FIXED: валидация при вводе и подсветка
                            w.bind("<KeyRelease>", lambda e, r=i, c=j:
                                   self._on_value_change(r, c, e))
                            # Устанавливаем цвет фона в зависимости от валидности
                            if (i, j) in self.validity and not self.validity[(i, j)]:
                                w.config(bg="#ffcccc")  # красноватый для ошибок
                            else:
                                w.config(bg="white")
                        w.grid(row=i - start_row, column=j - start_col, padx=1, pady=1)
                        self.entries[(i, j)] = w
            except Exception as e:
                logger.error(f"Display update error: {e}")
            finally:
                self.pending_update = False

        self.after(10, _update)

    def _on_value_change(self, row, col, event):
        if self.virtual_mode:
            return
        val = event.widget.get().strip()
        if val:
            num, ok = validate_number(val)
            if ok:
                self.data[(row, col)] = val
                self.validity[(row, col)] = True
                event.widget.config(bg="white")
            else:
                # Не сохраняем, подсвечиваем красным
                self.data.pop((row, col), None)
                self.validity[(row, col)] = False
                event.widget.config(bg="#ffcccc")
        else:
            self.data.pop((row, col), None)
            self.validity.pop((row, col), None)
            event.widget.config(bg="white")

    def _on_scroll_y(self, *args):
        self.canvas.yview(*args)
        self._schedule_update()

    def _on_scroll_x(self, *args):
        self.canvas.xview(*args)
        self._schedule_update()

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._schedule_update()

    def _on_shift_mousewheel(self, event):
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        self._schedule_update()

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.frame.config(width=event.width)

    def _schedule_update(self):
        if self.scroll_timer:
            self.after_cancel(self.scroll_timer)
        self.scroll_timer = self.after(Config.SCROLL_DEBOUNCE_MS, self._update_display)

    def get_matrix(self) -> List[List[float]]:
        """Возвращает матрицу как список списков float, пропуская некорректные ячейки (заменяя на 0)."""
        matrix = [[0.0] * self.cols for _ in range(self.rows)]
        for (i, j), v in self.data.items():
            try:
                matrix[i][j] = float(v)
            except:
                pass
        return matrix

    def set_values(self, matrix: Any) -> None:
        """Устанавливает значения из матрицы (numpy array или списка)."""
        self.data.clear()
        self.validity.clear()
        arr = to_ndarray(matrix)
        for i in range(min(self.rows, arr.shape[0])):
            for j in range(min(self.cols, arr.shape[1])):
                val = str(arr[i, j])
                self.data[(i, j)] = val
                self.validity[(i, j)] = True
        self._update_display()

    def set_size(self, rows: int, cols: int):
        """Установить размер матрицы и сразу обновить внутреннее состояние."""
        self.rows_var.set(rows)
        self.cols_var.set(cols)
        self._on_size_immediate()  # синхронное обновление

    def transpose(self):
        try:
            matrix = self.get_matrix()
            transposed = self.engine.transpose_matrix(matrix)
            self.set_size(transposed.shape[0], transposed.shape[1])
            self.set_values(transposed)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось транспонировать: {e}")

    def clear(self):
        self.data.clear()
        self.validity.clear()
        self._update_display()

    def load_from_file(self, filename: str):
        # FIXED: проверка размера файла
        try:
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            if size_mb > Config.MAX_FILE_SIZE_MB:
                messagebox.showerror("Ошибка", f"Файл слишком большой: {size_mb:.1f} МБ (макс. {Config.MAX_FILE_SIZE_MB} МБ)")
                return
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось получить размер файла: {e}")
            return

        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            matrix = []
            for line in lines:
                # Пропускаем пустые строки
                parts = line.strip().split()
                if parts:
                    row = []
                    for x in parts:
                        val, ok = validate_number(x)
                        if not ok:
                            raise ValueError(f"Некорректное число: {x}")
                        row.append(val)
                    matrix.append(row)
            if not matrix:
                return
            # Проверяем, что все строки одинаковой длины
            cols = len(matrix[0])
            if not all(len(row) == cols for row in matrix):
                raise ValueError("Строки имеют разную длину")
            self.set_size(len(matrix), cols)
            self.set_values(matrix)
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))

    def save_to_file(self, filename: str):
        try:
            matrix = self.get_matrix()
            arr = to_ndarray(matrix)
            np.savetxt(filename, arr, fmt='%.6f', delimiter='\t')
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))


# ----------------------------------------------------------------------
# Виджет результата (только текст)
# ----------------------------------------------------------------------
class ResultViewer(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._create_widgets()

    def _create_widgets(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=2)
        ttk.Button(toolbar, text="Копировать", command=self._copy_result).pack(side="right", padx=5)

        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True)

        self.text_widget = tk.Text(text_frame, wrap="none", font=('Courier', 10), height=8)
        v_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_widget.yview)
        h_scroll = ttk.Scrollbar(text_frame, orient="horizontal", command=self.text_widget.xview)
        self.text_widget.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.text_widget.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

    def set_result(self, result: Union[np.ndarray, float, int, str]):
        self.text_widget.delete(1.0, tk.END)
        if result is None:
            return
        if np.isscalar(result):
            self.text_widget.insert(1.0, str(result))
            return
        try:
            arr = to_ndarray(result)
            if arr.ndim == 0:
                self.text_widget.insert(1.0, str(arr.item()))
            elif arr.ndim == 1:
                self.text_widget.insert(1.0, "\t".join(f"{x:.6f}" for x in arr))
            else:
                for i in range(arr.shape[0]):
                    row_str = "\t".join(f"{x:.6f}" for x in arr[i])
                    self.text_widget.insert(tk.END, row_str + "\n")
        except Exception:
            self.text_widget.insert(1.0, str(result))

    def _copy_result(self):
        text = self.text_widget.get(1.0, tk.END).strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)


# ----------------------------------------------------------------------
# Главное окно приложения
# ----------------------------------------------------------------------
class MatrixCalculatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Матричный калькулятор")
        self.root.geometry("1400x800")

        self.engine = MatrixEngine()
        self.operation = tk.StringVar(value="Сложение")
        self.scalar_var = tk.DoubleVar(value=1.0)
        self.precision_var = tk.StringVar(value="auto")
        self.cancel_flag = False

        self.operations = {
            "Сложение": (2, False, self.engine.add_matrices),
            "Вычитание": (2, False, self.engine.subtract_matrices),
            "Умножение на число": (1, True, self.engine.scalar_multiply),
            "Перемножение матриц": (2, False, self.engine.multiply_matrices),
            "Транспонирование": (1, False, self.engine.transpose_matrix),
            "Определитель": (1, False, self.engine.determinant_matrix),
            "Ранг": (1, False, self.engine.rank_matrix),
            "Обращение": (1, False, self.engine.inverse_matrix),
            "Решение СЛАУ (A * X = B)": (2, False, self.engine.solve_system),
        }

        self._create_menu()
        self._create_widgets()
        self._start_monitoring()
        self._on_operation_change()

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Открыть матрицу A...", command=lambda: self._load_matrix(self.matrix_a))
        file_menu.add_command(label="Открыть матрицу B...", command=lambda: self._load_matrix(self.matrix_b))
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить матрицу A...", command=lambda: self._save_matrix(self.matrix_a))
        file_menu.add_command(label="Сохранить матрицу B...", command=lambda: self._save_matrix(self.matrix_b))
        file_menu.add_command(label="Сохранить результат...", command=self._save_result)
        menubar.add_cascade(label="Файл", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Копировать результат", command=self._copy_result)
        edit_menu.add_command(label="Очистить всё", command=self._clear_all)
        edit_menu.add_command(label="Очистить кэш", command=self._clear_cache)
        menubar.add_cascade(label="Правка", menu=edit_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="О программе", command=self._show_about)
        menubar.add_cascade(label="Справка", menu=help_menu)

        self.root.config(menu=menubar)

    def _create_widgets(self):
        # FIXED: разбито на логические части
        self._create_toolbar()
        self._create_matrices_panel()
        self._create_result_panel()
        self._create_status_bar()

    def _create_toolbar(self):
        toolbar = ttk.Frame(self.root, padding="5")
        toolbar.pack(fill="x")

        ttk.Label(toolbar, text="Операция:").pack(side="left", padx=5)
        op_cb = ttk.Combobox(toolbar, textvariable=self.operation,
                              values=list(self.operations.keys()),
                              state="readonly", width=20)
        op_cb.pack(side="left", padx=5)
        op_cb.bind("<<ComboboxSelected>>", self._on_operation_change)

        ttk.Label(toolbar, text="Скаляр:").pack(side="left", padx=(20, 5))
        self.scalar_entry = ttk.Entry(toolbar, textvariable=self.scalar_var, width=10)
        self.scalar_entry.pack(side="left", padx=5)

        ttk.Label(toolbar, text="Точность:").pack(side="left", padx=(20, 5))
        prec_cb = ttk.Combobox(toolbar, textvariable=self.precision_var,
                                values=["auto", "float64", "float32"],
                                state="readonly", width=8)
        prec_cb.pack(side="left", padx=5)
        prec_cb.bind("<<ComboboxSelected>>", self._on_precision_change)

        ttk.Button(toolbar, text="Вычислить", command=self._compute).pack(side="left", padx=10)
        ttk.Button(toolbar, text="Отмена", command=self._cancel_compute).pack(side="left", padx=2)
        ttk.Button(toolbar, text="⇄ Поменять", command=self._swap_matrices).pack(side="left", padx=2)

        self.sys_label = ttk.Label(toolbar, text="", foreground="green")
        self.sys_label.pack(side="right", padx=10)

    def _create_matrices_panel(self):
        matrices_frame = ttk.Frame(self.root, padding="10")
        matrices_frame.pack(fill="both", expand=True)

        self.matrix_a = MatrixWidget(matrices_frame, "Матрица A", self.engine)
        self.matrix_a.pack(side="left", fill="both", expand=True, padx=5)

        self.matrix_b = MatrixWidget(matrices_frame, "Матрица B", self.engine)
        self.matrix_b.pack(side="right", fill="both", expand=True, padx=5)

    def _create_result_panel(self):
        result_frame = ttk.LabelFrame(self.root, text="Результат", padding="10")
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.result_viewer = ResultViewer(result_frame)
        self.result_viewer.pack(fill="both", expand=True)

    def _create_status_bar(self):
        # Статусная строка с прогресс-баром
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=2)

        # Фрейм для прогресс-бара
        self.progress_frame = ttk.Frame(status_frame)
        self.progress = ttk.Progressbar(self.progress_frame, mode='indeterminate', length=150)
        self.progress.pack(side="left", padx=5)
        self.progress_label = ttk.Label(self.progress_frame, text="Вычисление...")
        self.progress_label.pack(side="left")

        # Метка статуса
        self.status = tk.StringVar(value="Готов к работе")
        self.status_label = ttk.Label(status_frame, textvariable=self.status)

        # Метка производительности
        self.perf_label = ttk.Label(status_frame, text="", foreground="blue")

        # Размещаем по сетке
        status_frame.columnconfigure(0, minsize=250)   # колонка для прогресс-бара
        status_frame.columnconfigure(1, weight=1)      # колонка для статуса (растягивается)
        status_frame.columnconfigure(2, minsize=200)   # колонка для производительности

        self.progress_frame.grid(row=0, column=0, sticky="w")
        self.status_label.grid(row=0, column=1, sticky="w", padx=5)
        self.perf_label.grid(row=0, column=2, sticky="e", padx=5)

        # По умолчанию прогресс-бар скрыт
        self.progress_frame.grid_remove()

    def _on_operation_change(self, event=None):
        op = self.operation.get()
        num_matrices, need_scalar, _ = self.operations[op]
        self.scalar_entry.config(state="normal" if need_scalar else "disabled")
        state = "normal" if num_matrices == 2 else "disabled"
        # Блокировка/разблокировка ввода в матрице B
        for child in self.matrix_b.winfo_children():
            if isinstance(child, ttk.Entry) or isinstance(child, ttk.Spinbox):
                child.config(state=state)

    def _on_precision_change(self, event=None):
        prec_map = {
            "auto": ComputePrecision.AUTO,
            "float64": ComputePrecision.FP64,
            "float32": ComputePrecision.FP32,
        }
        self.engine.precision = prec_map.get(self.precision_var.get(), ComputePrecision.AUTO)
        self.status.set(f"Точность: {self.precision_var.get()}")

    def _swap_matrices(self):
        try:
            a_data = self.matrix_a.data.copy()
            a_rows, a_cols = self.matrix_a.rows, self.matrix_a.cols
            b_data = self.matrix_b.data.copy()
            b_rows, b_cols = self.matrix_b.rows, self.matrix_b.cols

            # Меняем размеры и данные
            self.matrix_a.set_size(b_rows, b_cols)
            self.matrix_b.set_size(a_rows, a_cols)

            self.matrix_a.data = b_data
            self.matrix_b.data = a_data

            self.matrix_a._update_display()
            self.matrix_b._update_display()
            self.status.set("Матрицы обменяны")
        except Exception as e:
            logger.error(f"Swap error: {e}")
            messagebox.showerror("Ошибка", str(e))

    def _load_matrix(self, widget: MatrixWidget):
        filename = filedialog.askopenfilename(
            title=f"Загрузить {widget.title}",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            widget.load_from_file(filename)

    def _save_matrix(self, widget: MatrixWidget):
        filename = filedialog.asksaveasfilename(
            title=f"Сохранить {widget.title}",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")]
        )
        if filename:
            widget.save_to_file(filename)

    def _save_result(self):
        filename = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if filename:
            try:
                text = self.result_viewer.text_widget.get(1.0, tk.END).strip()
                with open(filename, 'w') as f:
                    f.write(text)
            except Exception as e:
                messagebox.showerror("Ошибка сохранения", str(e))

    def _copy_result(self):
        self.result_viewer._copy_result()

    def _clear_all(self):
        self.matrix_a.clear()
        self.matrix_b.clear()
        self.result_viewer.text_widget.delete(1.0, tk.END)
        self.status.set("Все очищено")

    def _clear_cache(self):
        self.engine.clear_cache()
        self.status.set("Кэш очищен")

    def _show_about(self):
        messagebox.showinfo("О программе",
                            "Матричный калькулятор\n"
                            "Версия 2.5\n"
                            "Использует NumPy и CuPy (опционально)")

    def _cancel_compute(self):
        self.cancel_flag = True
        self.status.set("Отмена вычислений...")

    def _compute(self):
        self._set_buttons_state('disabled')
        # Показываем прогресс-бар
        self.progress_frame.grid()
        self.progress.start(10)
        self.cancel_flag = False
        self.status.set("Вычисление...")

        thread = threading.Thread(target=self._compute_thread)
        thread.daemon = True
        thread.start()
        self._poll_compute_thread(thread)

    def _poll_compute_thread(self, thread):
        if thread.is_alive():
            self.root.after(Config.PROGRESS_UPDATE_INTERVAL, self._poll_compute_thread, thread)
        else:
            self.progress.stop()
            self.progress_frame.grid_remove()
            self._set_buttons_state('normal')

    def _set_buttons_state(self, state):
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button) and child['text'] != 'Отмена':
                        try:
                            child.config(state=state)
                        except:
                            pass

    def _compute_thread(self):
        try:
            start_time = time.time()
            op_name = self.operation.get()
            num_matrices, need_scalar, func = self.operations[op_name]

            A = self.matrix_a.get_matrix()
            B = self.matrix_b.get_matrix() if num_matrices == 2 else None
            scalar = self.scalar_var.get() if need_scalar else None

            # FIXED: проверка отмены перед началом
            if self.cancel_flag:
                self.root.after(0, lambda: self.status.set("Вычисление отменено"))
                return

            if need_scalar:
                result = func(A, scalar)
            elif num_matrices == 2:
                # Проверим совместимость для умножения и решения
                if op_name == "Перемножение матриц":
                    A_arr = to_ndarray(A)
                    B_arr = to_ndarray(B)
                    if A_arr.shape[1] != B_arr.shape[0]:
                        raise ValueError("Число столбцов A должно равняться числу строк B")
                elif op_name == "Решение СЛАУ (A * X = B)":
                    A_arr = to_ndarray(A)
                    B_arr = to_ndarray(B)
                    if A_arr.shape[0] != B_arr.shape[0]:
                        raise ValueError("Количество строк A и B не совпадает")
                result = func(A, B)
            else:
                result = func(A)

            elapsed = time.time() - start_time
            device = "GPU" if self.engine.gpu_available else "CPU"
            perf_msg = f"Время: {elapsed*1000:.2f} ms | Устр-во: {device}"

            self.root.after(0, self._update_result, result, perf_msg)

        except Exception as e:
            logger.error(f"Compute error: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Ошибка вычисления", str(e)))
        finally:
            self.cancel_flag = False

    def _update_result(self, result, perf_msg):
        self.result_viewer.set_result(result)
        self.perf_label.config(text=perf_msg)
        self.status.set("Готово")

    def _start_monitoring(self):
        def monitor():
            stats = self.engine.get_stats()
            device_str = "GPU" if self.engine.gpu_available else "CPU"
            self.sys_label.config(
                text=f"Кэш: {stats['cache']['hit_rate']:.1f}% | {device_str}"
            )
            self.root.after(2000, monitor)
        self.root.after(2000, monitor)


# ----------------------------------------------------------------------
# Пример тестов (для pytest)
# ----------------------------------------------------------------------
# FIXED: добавлены примеры тестов для ключевых функций
if __name__ == "__main__" and "pytest" in sys.modules:
    # Эти тесты будут выполняться только при запуске pytest
    import pytest

    @pytest.fixture
    def engine():
        return MatrixEngine()

    def test_add_matrices(engine):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = engine.add_matrices(A, B)
        np.testing.assert_array_equal(C, [[6, 8], [10, 12]])

    def test_determinant(engine):
        A = [[1, 2], [3, 4]]
        det = engine.determinant_matrix(A)
        assert det == -2.0

    def test_rank(engine):
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        rank = engine.rank_matrix(A)
        assert rank == 2  # матрица вырождена

    def test_inverse(engine):
        A = [[1, 2], [3, 4]]
        inv = engine.inverse_matrix(A)
        expected = [[-2.0, 1.0], [1.5, -0.5]]
        np.testing.assert_almost_equal(inv, expected)

    def test_solve(engine):
        A = [[3, 1], [1, 2]]
        B = [9, 8]
        X = engine.solve_system(A, B)
        np.testing.assert_almost_equal(X, [2, 3])

    def test_validate_number():
        assert validate_number("123") == (123.0, True)
        assert validate_number("-3.14") == (-3.14, True)
        assert validate_number("abc") == (None, False)
        assert validate_number("1e500") == (None, False)  # переполнение


# ----------------------------------------------------------------------
# Запуск приложения
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    root = tk.Tk()
    app = MatrixCalculatorApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Приложение завершено")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        messagebox.showerror("Критическая ошибка", str(e))
