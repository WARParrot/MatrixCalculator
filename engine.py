import numpy as np
from typing import Optional
import config

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
        self.cancel_requested = False   # for thread cancellation

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
        else:  # AUTO – use float64 by default
            return np.float64

    def clear_cache(self):
        """Stub for cache clearing (not implemented)."""
        pass

    def _validate_square_matrix(self, A: np.ndarray, operation: str = "operation"):
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"{operation} requires a square matrix. Got shape {A.shape}")

    def _validate_shapes(self, A: np.ndarray, B: np.ndarray, operation: str):
        if A.shape != B.shape:
            raise ValueError(f"{operation} requires matrices of the same shape. "
                           f"Got {A.shape} and {B.shape}")

    # --- Step‑by‑step Gauss‑Jordan inversion ---
    def _calculate_inverse_steps(self, A: np.ndarray) -> tuple:
        n = A.shape[0]
        aug = np.hstack([A.astype(self._current_dtype), np.eye(n, dtype=self._current_dtype)])
        steps_log = []
        current_step = 0

        for col in range(n):
            # Pivot selection (partial pivoting)
            max_row_index = col + np.argmax(np.abs(aug[col:, col]))
            if not np.isclose(aug[max_row_index, col], 0):
                if max_row_index != col:
                    aug[[col, max_row_index]] = aug[[max_row_index, col]]
                    steps_log.append({
                        'step': current_step,
                        'desc': f"Swapped Row {col + 1} and Row {max_row_index + 1}",
                        'state': aug.copy()
                    })
                    current_step += 1

            pivot_val = aug[col, col]
            if np.isclose(pivot_val, 0):
                raise np.linalg.LinAlgError("Matrix is singular (cannot be inverted)")

            aug[col] /= pivot_val
            steps_log.append({
                'step': current_step,
                'desc': f"Normalized Row {col + 1} by dividing by {pivot_val:.2f}",
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
                            'desc': f"Eliminated column {col + 1} from Row {i + 1}",
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
            # The steps are returned; the caller (GUI) will display them.
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
                raise ValueError(f"Inversion failed: {str(e)}")

    # --- Other operations (all return (result, steps) where steps may be None) ---
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
            raise ValueError(f"Incompatible shapes for multiplication: {A.shape} and {B.shape}")
        return A @ B, None

    def scalar_multiply(self, A, scalar):
        A = np.array(A)
        return A * scalar, None

    def transpose_matrix(self, A):
        A = np.array(A)
        return A.T, None

    def determinant_matrix(self, A, show_steps=False):
        A = np.array(A)
        self._validate_square_matrix(A, "Determinant calculation")
        if show_steps:
            # Simplified step logging – in a real app you would log elimination steps
            det = np.linalg.det(A.astype(self._current_dtype))
            # Return dummy steps for now
            steps = [{'step': 0, 'desc': 'Determinant computed via NumPy', 'state': A}]
            return det, steps
        else:
            return np.linalg.det(A.astype(self._current_dtype)), None

    def _calculate_rank_steps(self, A: np.ndarray) -> tuple:
        """
        Performs Gaussian elimination (forward elimination) and logs steps.
        Returns (rank, list_of_step_dictionaries).
        """
        A = A.astype(self._current_dtype, copy=True)
        n_rows, n_cols = A.shape
        steps_log = []
        current_step = 0
        rank = 0
        row = 0
        col = 0

        while row < n_rows and col < n_cols:
            # Find pivot (max absolute in this column from current row downwards)
            pivot_row = np.argmax(np.abs(A[row:, col])) + row
            if np.isclose(A[pivot_row, col], 0):
                # No pivot in this column, move to next column
                col += 1
                continue

            # Swap if needed
            if pivot_row != row:
                A[[row, pivot_row]] = A[[pivot_row, row]]
                steps_log.append({
                    'step': current_step,
                    'desc': f"Swapped row {row} and row {pivot_row}",
                    'state': A.copy()
                })
                current_step += 1

            # Eliminate below
            for i in range(row + 1, n_rows):
                if not np.isclose(A[i, col], 0):
                    factor = A[i, col] / A[row, col]
                    A[i, col:] -= factor * A[row, col:]
                    steps_log.append({
                        'step': current_step,
                        'desc': f"Eliminated column {col} from row {i}",
                        'state': A.copy()
                    })
                    current_step += 1

            rank += 1
            row += 1
            col += 1

        return rank, steps_log

    def rank_matrix(self, A, show_steps=False):
        A = np.array(A)
        if show_steps:
            rank, steps = self._calculate_rank_steps(A)
            return rank, steps
        else:
            return np.linalg.matrix_rank(A), None

    def _solve_system_steps(self, A: np.ndarray, B: np.ndarray) -> tuple:
        """
        Solves Ax = B using Gaussian elimination with partial pivoting.
        Logs each step (row swaps, elimination, back substitution).
        Returns (X, list_of_step_dictionaries).
        """
        A = A.astype(self._current_dtype, copy=True)
        B = B.astype(self._current_dtype, copy=True)
        n = A.shape[0]
        steps_log = []
        current_step = 0

        # Check if B is a vector or a matrix
        vector_rhs = (B.ndim == 1)
        if vector_rhs:
            B = B.reshape(-1, 1)   # make column vector
        m = B.shape[1]              # number of RHS

        # Build augmented matrix [A | B]
        aug = np.hstack([A, B])
        steps_log.append({
            'step': current_step,
            'desc': "Начальная расширенная матрица [A | B]",
            'state': aug.copy()
        })
        current_step += 1

        # Forward elimination (to upper triangular form)
        for col in range(n):
            # Partial pivoting
            pivot_row = np.argmax(np.abs(aug[col:, col])) + col
            if np.isclose(aug[pivot_row, col], 0):
                raise np.linalg.LinAlgError("Matrix is singular (pivot zero)")

            if pivot_row != col:
                aug[[col, pivot_row]] = aug[[pivot_row, col]]
                steps_log.append({
                    'step': current_step,
                    'desc': f"Swapped row {col} and row {pivot_row}",
                    'state': aug.copy()
                })
                current_step += 1

            # Eliminate below
            for i in range(col + 1, n):
                if not np.isclose(aug[i, col], 0):
                    factor = aug[i, col] / aug[col, col]
                    aug[i, col:] -= factor * aug[col, col:]
                    steps_log.append({
                        'step': current_step,
                        'desc': f"Eliminated column {col} from row {i} (factor = {factor:.4f})",
                        'state': aug.copy()
                    })
                    current_step += 1

        # At this point aug is upper triangular (first n columns)
        # Extract upper triangular part and RHS
        U = aug[:, :n]
        rhs = aug[:, n:]   # modified RHS after elimination

        # Back substitution
        X = np.zeros((n, m), dtype=aug.dtype)
        for i in reversed(range(n)):
            sum_ax = np.zeros(m)
            for j in range(i+1, n):
                sum_ax += U[i, j] * X[j]
            X[i] = (rhs[i] - sum_ax) / U[i, i]

            steps_log.append({
                'step': current_step,
                'desc': f"Back substitution: x[{i}] = {X[i]}",
                'state': X.copy()   # show current solution vector(s)
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
            raise ValueError("Matrix A and RHS B must have the same number of rows")

        if show_steps:
            return self._solve_system_steps(A, B)
        else:
            # Fast path
            try:
                return np.linalg.solve(A.astype(self._current_dtype),
                                       B.astype(self._current_dtype)), None
            except np.linalg.LinAlgError as e:
                raise ValueError(f"System has no unique solution: {e}")

    def stats(self):
        return {
            "device": self.device,
            "precision": self.precision.value,
            "cache_hits": 0,
            "cache_misses": 0
        }