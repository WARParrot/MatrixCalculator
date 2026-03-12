import numpy as np
from typing import Optional
import config
from localization import Language

class MatrixEngine:
    def __init__(self):
        self.precision = config.ComputePrecision.AUTO
        self._gpu_available = False
        try:
            import cupy as cp
            self._gpu_available = True
        except ImportError:
            pass
        self.device = "GPU (CuPy)" if self._gpu_available else "CPU (NumPy)"
        self.cancel_requested = False

    @property
    def gpu_available(self):
        return self._gpu_available

    def set_precision(self, prec: config.ComputePrecision):
        self.precision = prec

    @property
    def _current_dtype(self):
        if self.precision == config.ComputePrecision.FP32:
            return np.float32
        elif self.precision == config.ComputePrecision.FP64:
            return np.float64
        else:  # AUTO
            return np.float64

    def clear_cache(self):
        """Stub for cache clearing."""
        pass

    def _validate_square_matrix(self, A: np.ndarray, operation: str = "operation"):
        if A.shape[0] != A.shape[1]:
            raise ValueError(Language.tr('err_square_matrix',
                                        operation=operation, shape=A.shape))

    def _validate_shapes(self, A: np.ndarray, B: np.ndarray, operation: str):
        if A.shape != B.shape:
            raise ValueError(Language.tr('err_same_shape',
                                        operation=operation, shape1=A.shape, shape2=B.shape))

    # --- Inverse with steps (1‑based indices) ---
    def _calculate_inverse_steps(self, A: np.ndarray) -> tuple:
        n = A.shape[0]
        aug = np.hstack([A.astype(self._current_dtype), np.eye(n, dtype=self._current_dtype)])
        steps_log = []
        current_step = 0

        steps_log.append({
            'step': current_step,
            'desc': Language.tr('step_initial_aug'),
            'state': aug.copy()
        })
        current_step += 1

        for col in range(n):
            max_row_index = col + np.argmax(np.abs(aug[col:, col]))
            if not np.isclose(aug[max_row_index, col], 0):
                if max_row_index != col:
                    aug[[col, max_row_index]] = aug[[max_row_index, col]]
                    steps_log.append({
                        'step': current_step,
                        'desc': Language.tr('step_swap', row1=col+1, row2=max_row_index+1),
                        'state': aug.copy()
                    })
                    current_step += 1

            pivot_val = aug[col, col]
            if np.isclose(pivot_val, 0):
                raise np.linalg.LinAlgError(Language.tr('err_singular_matrix'))

            aug[col] /= pivot_val
            steps_log.append({
                'step': current_step,
                'desc': Language.tr('step_normalize', row=col+1, value=pivot_val),
                'state': aug.copy()
            })
            current_step += 1

            for i in range(n):
                if i != col:
                    factor = aug[i, col]
                    if not np.isclose(factor, 0):
                        aug[i] -= factor * aug[col]
                        steps_log.append({
                            'step': current_step,
                            'desc': Language.tr('step_eliminate', col=col+1, row=i+1, factor=factor),
                            'state': aug.copy()
                        })
                        current_step += 1

        inverse = aug[:, n:]
        return inverse, steps_log

    def inverse_matrix(self, A, show_steps=False):
        if not isinstance(A, np.ndarray):
            A = np.array(A)
        self._validate_square_matrix(A, "Matrix inversion")
        if show_steps:
            result, log = self._calculate_inverse_steps(A.astype(self._current_dtype))
            return result, log
        else:
            try:
                if self.gpu_available:
                    import cupy as cp
                    A_gpu = cp.asarray(A)
                    result = cp.linalg.inv(A_gpu).get()
                else:
                    result = np.linalg.inv(A.astype(self._current_dtype))
                return result, None
            except Exception as e:
                raise ValueError(Language.tr('err_inversion_failed', msg=str(e)))

    # --- Determinant with steps (1‑based indices) ---
    def _calculate_determinant_steps(self, A: np.ndarray) -> tuple:
        A = A.astype(self._current_dtype, copy=True)
        n = A.shape[0]
        steps_log = []
        current_step = 0
        sign = 1

        steps_log.append({
            'step': current_step,
            'desc': Language.tr('step_initial'),
            'state': A.copy()
        })
        current_step += 1

        for col in range(n):
            pivot_row = np.argmax(np.abs(A[col:, col])) + col
            if np.isclose(A[pivot_row, col], 0):
                steps_log.append({
                    'step': current_step,
                    'desc': Language.tr('step_pivot_zero', col=col+1),
                    'state': A.copy()
                })
                steps_log.append({
                    'step': current_step + 1,
                    'desc': Language.tr('step_singular'),
                    'state': None
                })
                return 0.0, steps_log

            if pivot_row != col:
                A[[col, pivot_row]] = A[[pivot_row, col]]
                sign *= -1
                steps_log.append({
                    'step': current_step,
                    'desc': Language.tr('step_swap', row1=col+1, row2=pivot_row+1),
                    'state': A.copy()
                })
                current_step += 1

            pivot_val = A[col, col]
            for i in range(col + 1, n):
                if not np.isclose(A[i, col], 0):
                    factor = A[i, col] / pivot_val
                    A[i, col:] -= factor * A[col, col:]
                    steps_log.append({
                        'step': current_step,
                        'desc': Language.tr('step_eliminate', col=col+1, row=i+1, factor=factor),
                        'state': A.copy()
                    })
                    current_step += 1

        diag_product = np.prod(np.diag(A))
        det = sign * diag_product

        steps_log.append({
            'step': current_step,
            'desc': Language.tr('step_diag_product', product=diag_product, sign=sign),
            'state': A.copy()
        })
        steps_log.append({
            'step': current_step + 1,
            'desc': Language.tr('step_det_result', det=det),
            'state': None
        })

        return det, steps_log

    def determinant_matrix(self, A, show_steps=False):
        A = np.array(A)
        self._validate_square_matrix(A, "Determinant calculation")
        if show_steps:
            return self._calculate_determinant_steps(A)
        else:
            return np.linalg.det(A.astype(self._current_dtype)), None

    # --- Rank with steps (1‑based indices) ---
    def _calculate_rank_steps(self, A: np.ndarray) -> tuple:
        A = A.astype(self._current_dtype, copy=True)
        n_rows, n_cols = A.shape
        steps_log = []
        current_step = 0
        rank = 0
        row = 0
        col = 0

        steps_log.append({
            'step': current_step,
            'desc': Language.tr('step_initial'),
            'state': A.copy()
        })
        current_step += 1

        while row < n_rows and col < n_cols:
            pivot_row = np.argmax(np.abs(A[row:, col])) + row
            if np.isclose(A[pivot_row, col], 0):
                col += 1
                continue

            if pivot_row != row:
                A[[row, pivot_row]] = A[[pivot_row, row]]
                steps_log.append({
                    'step': current_step,
                    'desc': Language.tr('step_swap', row1=row+1, row2=pivot_row+1),
                    'state': A.copy()
                })
                current_step += 1

            for i in range(row + 1, n_rows):
                if not np.isclose(A[i, col], 0):
                    factor = A[i, col] / A[row, col]
                    A[i, col:] -= factor * A[row, col:]
                    steps_log.append({
                        'step': current_step,
                        'desc': Language.tr('step_eliminate', col=col+1, row=i+1, factor=factor),
                        'state': A.copy()
                    })
                    current_step += 1

            rank += 1
            row += 1
            col += 1

        steps_log.append({
            'step': current_step,
            'desc': Language.tr('step_rank_final', rank=rank),
            'state': None
        })
        return rank, steps_log

    def rank_matrix(self, A, show_steps=False):
        A = np.array(A)
        if show_steps:
            return self._calculate_rank_steps(A)
        else:
            return np.linalg.matrix_rank(A), None

    # --- Solve system with steps (1‑based indices) ---
    def _solve_system_steps(self, A: np.ndarray, B: np.ndarray) -> tuple:
        A = A.astype(self._current_dtype, copy=True)
        B = B.astype(self._current_dtype, copy=True)
        n = A.shape[0]
        steps_log = []
        current_step = 0

        vector_rhs = (B.ndim == 1)
        if vector_rhs:
            B = B.reshape(-1, 1)
        m = B.shape[1]

        aug = np.hstack([A, B])
        steps_log.append({
            'step': current_step,
            'desc': Language.tr('step_initial_aug'),
            'state': aug.copy()
        })
        current_step += 1

        for col in range(n):
            pivot_row = np.argmax(np.abs(aug[col:, col])) + col
            if np.isclose(aug[pivot_row, col], 0):
                raise np.linalg.LinAlgError(Language.tr('err_singular_matrix'))

            if pivot_row != col:
                aug[[col, pivot_row]] = aug[[pivot_row, col]]
                steps_log.append({
                    'step': current_step,
                    'desc': Language.tr('step_swap', row1=col+1, row2=pivot_row+1),
                    'state': aug.copy()
                })
                current_step += 1

            for i in range(col + 1, n):
                if not np.isclose(aug[i, col], 0):
                    factor = aug[i, col] / aug[col, col]
                    aug[i, col:] -= factor * aug[col, col:]
                    steps_log.append({
                        'step': current_step,
                        'desc': Language.tr('step_eliminate', col=col+1, row=i+1, factor=factor),
                        'state': aug.copy()
                    })
                    current_step += 1

        U = aug[:, :n]
        rhs = aug[:, n:]
        X = np.zeros((n, m), dtype=aug.dtype)

        for i in reversed(range(n)):
            sum_ax = np.zeros(m)
            for j in range(i+1, n):
                sum_ax += U[i, j] * X[j]
            X[i] = (rhs[i] - sum_ax) / U[i, i]
            steps_log.append({
                'step': current_step,
                'desc': Language.tr('step_back_subst', i=i+1, value=X[i]),
                'state': X.copy() if m == 1 else None
            })
            current_step += 1

        if vector_rhs:
            X = X.flatten()
        return X, steps_log

    def solve_system(self, A, B, show_steps=False):
        A = np.array(A)
        B = np.array(B)
        self._validate_square_matrix(A, "System solution")
        if A.shape[0] != B.shape[0]:
            raise ValueError(Language.tr('err_system_rows'))
        if show_steps:
            return self._solve_system_steps(A, B)
        else:
            try:
                return np.linalg.solve(A.astype(self._current_dtype),
                                       B.astype(self._current_dtype)), None
            except np.linalg.LinAlgError as e:
                raise ValueError(Language.tr('err_no_unique_solution', msg=str(e)))

    # --- Basic operations (no steps) ---
    def add_matrices(self, A, B):
        A, B = np.array(A), np.array(B)
        self._validate_shapes(A, B, "Matrix addition")
        return A + B, None

    def subtract_matrices(self, A, B):
        A, B = np.array(A), np.array(B)
        self._validate_shapes(A, B, "Matrix subtraction")
        return A - B, None

    def multiply_matrices(self, A, B):
        A, B = np.array(A), np.array(B)
        if A.shape[1] != B.shape[0]:
            raise ValueError(Language.tr('err_incompatible_mul', shape1=A.shape, shape2=B.shape))
        return A @ B, None

    def scalar_multiply(self, A, scalar):
        A = np.array(A)
        return A * scalar, None

    def transpose_matrix(self, A):
        A = np.array(A)
        return A.T, None

    def stats(self):
        return {
            "device": self.device,
            "precision": self.precision.value,
            "cache_hits": 0,
            "cache_misses": 0
        }