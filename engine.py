import numpy as np
from typing import Optional, Union, List, Tuple, Any
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

    # ----------------------------------------------------------------------
    # Vector utility methods
    # ----------------------------------------------------------------------
    def _as_vector(self, v: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """Convert input to 1D float array with current precision."""
        v = np.asarray(v, dtype=self._current_dtype)
        if v.ndim > 1:
            v = v.flatten()
        if v.ndim != 1:
            raise ValueError(Language.tr('err_vector_1d', shape=v.shape))
        return v

    def _validate_vectors_same_length(self, v1: np.ndarray, v2: np.ndarray, op: str):
        if v1.shape != v2.shape:
            raise ValueError(Language.tr('err_vector_same_len', op=op, len1=len(v1), len2=len(v2)))

    # ----------------------------------------------------------------------
    # Vector addition / subtraction / scalar multiplication (with steps)
    # ----------------------------------------------------------------------
    def _format_vector(self, v: np.ndarray) -> str:
        """Преобразует 1D массив в строку вида '(1.2, 3.4, 5.6)'."""
        if v.ndim == 0:
            return str(v.item())
        return "(" + ", ".join(f"{x:.4g}" for x in v) + ")"

    # ----------------------------------------------------------------------
    # Векторные операции (замените существующие методы)
    # ----------------------------------------------------------------------
    def vector_add(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "addition")
        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_vector_add_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2)),
                          'state': None})
            result = v1 + v2
            steps.append({'step': 1, 'desc': Language.tr('step_vector_add_result',
                                                         res=self._format_vector(result)), 'state': result.copy()})
            return result, steps
        return v1 + v2, None

    def vector_subtract(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "subtraction")
        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_vector_sub_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2)),
                          'state': None})
            result = v1 - v2
            steps.append({'step': 1, 'desc': Language.tr('step_vector_sub_result',
                                                         res=self._format_vector(result)), 'state': result.copy()})
            return result, steps
        return v1 - v2, None

    def vector_scalar_multiply(self, v, scalar, show_steps=False):
        v = self._as_vector(v)
        scalar = float(scalar)
        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_vector_scale_init',
                                                         v=self._format_vector(v), scalar=scalar), 'state': None})
            result = v * scalar
            steps.append({'step': 1, 'desc': Language.tr('step_vector_scale_result',
                                                         res=self._format_vector(result)), 'state': result.copy()})
            return result, steps
        return v * scalar, None

    def vector_dot(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "dot product")
        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_dot_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2)),
                          'state': None})
            products = v1 * v2
            steps.append({'step': 1, 'desc': Language.tr('step_dot_products',
                                                         prods=self._format_vector(products)),
                          'state': products.copy()})
            dot_val = float(np.sum(products))
            steps.append({'step': 2, 'desc': Language.tr('step_dot_sum', sum=dot_val), 'state': None})
            return dot_val, steps
        return float(np.dot(v1, v2)), None

    def vector_cross(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        if len(v1) != 3 or len(v2) != 3:
            raise ValueError(Language.tr('err_cross_3d', len1=len(v1), len2=len(v2)))
        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_cross_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2)),
                          'state': None})
            x = v1[1] * v2[2] - v1[2] * v2[1]
            y = v1[2] * v2[0] - v1[0] * v2[2]
            z = v1[0] * v2[1] - v1[1] * v2[0]
            steps.append({'step': 1, 'desc': Language.tr('step_cross_components', x=x, y=y, z=z), 'state': None})
            result = np.array([x, y, z], dtype=self._current_dtype)
            steps.append({'step': 2, 'desc': Language.tr('step_cross_result',
                                                         res=self._format_vector(result)), 'state': result.copy()})
            return result, steps
        return np.cross(v1, v2), None

    def vector_norm(self, v, show_steps=False):
        v = self._as_vector(v)
        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_norm_init',
                                                         v=self._format_vector(v)), 'state': None})
            squares = v ** 2
            steps.append({'step': 1, 'desc': Language.tr('step_norm_squares',
                                                         squares=self._format_vector(squares)),
                          'state': squares.copy()})
            sum_sq = float(np.sum(squares))
            steps.append({'step': 2, 'desc': Language.tr('step_norm_sum_sq', sum_sq=sum_sq), 'state': None})
            norm = np.sqrt(sum_sq)
            steps.append({'step': 3, 'desc': Language.tr('step_norm_result', norm=norm), 'state': None})
            return norm, steps
        return float(np.linalg.norm(v)), None

    def vector_normalize(self, v, show_steps=False):
        v = self._as_vector(v)
        norm = np.linalg.norm(v)
        if np.isclose(norm, 0.0):
            raise ValueError(Language.tr('err_normalize_zero'))
        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_normalize_init',
                                                         v=self._format_vector(v)), 'state': None})
            steps.append({'step': 1, 'desc': Language.tr('step_normalize_norm', norm=norm), 'state': None})
            unit = v / norm
            steps.append({'step': 2, 'desc': Language.tr('step_normalize_result',
                                                         unit=self._format_vector(unit)), 'state': unit.copy()})
            return unit, steps
        return v / norm, None

    def vector_projection(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "projection")
        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_proj_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2)),
                          'state': None})
            dot = float(np.dot(v1, v2))
            steps.append({'step': 1, 'desc': Language.tr('step_proj_dot', dot=dot), 'state': None})
            norm_sq = float(np.dot(v2, v2))
            steps.append({'step': 2, 'desc': Language.tr('step_proj_norm_sq', norm_sq=norm_sq), 'state': None})
            if np.isclose(norm_sq, 0.0):
                raise ValueError(Language.tr('err_projection_zero_vec'))
            scalar = dot / norm_sq
            steps.append({'step': 3, 'desc': Language.tr('step_proj_scalar', scalar=scalar), 'state': None})
            proj = scalar * v2
            steps.append({'step': 4, 'desc': Language.tr('step_proj_result',
                                                         proj=self._format_vector(proj)), 'state': proj.copy()})
            return proj, steps
        else:
            norm_sq = np.dot(v2, v2)
            if np.isclose(norm_sq, 0.0):
                raise ValueError(Language.tr('err_projection_zero_vec'))
            return (np.dot(v1, v2) / norm_sq) * v2, None

    def vector_angle(self, v1, v2, show_steps=False, degrees=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "angle")
        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_angle_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2)),
                          'state': None})
            dot = float(np.dot(v1, v2))
            steps.append({'step': 1, 'desc': Language.tr('step_angle_dot', dot=dot), 'state': None})
            norm1 = float(np.linalg.norm(v1))
            norm2 = float(np.linalg.norm(v2))
            steps.append({'step': 2, 'desc': Language.tr('step_angle_norms', norm1=norm1, norm2=norm2), 'state': None})
            if np.isclose(norm1, 0.0) or np.isclose(norm2, 0.0):
                raise ValueError(Language.tr('err_angle_zero_vec'))
            cos_theta = dot / (norm1 * norm2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            steps.append({'step': 3, 'desc': Language.tr('step_angle_cos', cos=cos_theta), 'state': None})
            theta_rad = float(np.arccos(cos_theta))
            steps.append({'step': 4, 'desc': Language.tr('step_angle_rad', rad=theta_rad), 'state': None})
            if degrees:
                theta_deg = float(np.degrees(theta_rad))
                steps.append({'step': 5, 'desc': Language.tr('step_angle_deg', deg=theta_deg), 'state': None})
                return theta_deg, steps
            return theta_rad, steps
        else:
            dot = np.dot(v1, v2)
            norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
            if np.isclose(norm_prod, 0.0):
                raise ValueError(Language.tr('err_angle_zero_vec'))
            cos_theta = np.clip(dot / norm_prod, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            return float(np.degrees(theta)) if degrees else float(theta), None

    def vector_triple_scalar(self, v1, v2, v3, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        v3 = self._as_vector(v3)
        if not (len(v1) == len(v2) == len(v3) == 3):
            raise ValueError(Language.tr('err_triple_3d'))
        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_triple_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2),
                                                         v3=self._format_vector(v3)), 'state': None})
            cross = np.cross(v2, v3)
            steps.append({'step': 1, 'desc': Language.tr('step_triple_cross',
                                                         cross=self._format_vector(cross)), 'state': cross.copy()})
            dot = float(np.dot(v1, cross))
            steps.append({'step': 2, 'desc': Language.tr('step_triple_dot', dot=dot), 'state': None})
            return dot, steps
        else:
            return float(np.dot(v1, np.cross(v2, v3))), None
