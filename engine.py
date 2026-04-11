import numpy as np
from typing import Optional, Union, List, Tuple, Any
import config
from localization import Language
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

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

        # Symbolic mode flag
        self._symbolic_mode = False

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
        else:
            return np.float64

    # ------------------------------------------------------------------
    # Symbolic mode management
    # ------------------------------------------------------------------
    def set_symbolic_mode(self, enabled: bool):
        self._symbolic_mode = enabled

    def get_symbolic_mode(self) -> bool:
        return self._symbolic_mode

    def _parse_expression(self, value):
        """Convert input to float (numeric) or SymPy expression (symbolic)."""
        if isinstance(value, (int, float)):
            return float(value) if not self._symbolic_mode else sp.sympify(value)
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return 0.0 if not self._symbolic_mode else sp.Integer(0)
            try:
                if not self._symbolic_mode:
                    return float(value)
                else:
                    transformations = (standard_transformations +
                                      (implicit_multiplication_application, convert_xor))
                    return parse_expr(value, transformations=transformations)
            except Exception as e:
                raise ValueError(f"Cannot parse expression '{value}': {e}")
        return value

    def _to_numpy(self, matrix, subs_dict=None):
        if isinstance(matrix, (sp.Matrix, sp.Expr)):
            if subs_dict:
                matrix = matrix.subs(subs_dict)
            return np.array(matrix).astype(self._current_dtype)
        return np.asarray(matrix, dtype=self._current_dtype)

    def _to_sympy_matrix(self, data):
        sym_data = []
        for row in data:
            sym_row = [self._parse_expression(cell) for cell in row]
            sym_data.append(sym_row)
        return sp.Matrix(sym_data)

    def _as_vector(self, v):
        if self._symbolic_mode:
            if not isinstance(v, sp.Matrix):
                v = sp.Matrix([self._parse_expression(x) for x in v])
            if v.shape[1] != 1:
                v = v.reshape(v.rows, 1)
            return v
        else:
            v = np.asarray(v, dtype=self._current_dtype)
            if v.ndim > 1:
                v = v.flatten()
            if v.ndim != 1:
                raise ValueError(Language.tr('err_vector_1d', shape=v.shape))
            return v

    def _validate_vectors_same_length(self, v1, v2, op: str):
        if len(v1) != len(v2):
            raise ValueError(Language.tr('err_vector_same_len', op=op, len1=len(v1), len2=len(v2)))

    def _validate_square_matrix(self, A, operation="operation"):
        if A.shape[0] != A.shape[1]:
            raise ValueError(Language.tr('err_square_matrix', operation=operation, shape=A.shape))

    def _validate_shapes(self, A, B, operation: str):
        if A.shape != B.shape:
            raise ValueError(Language.tr('err_same_shape', operation=operation, shape1=A.shape, shape2=B.shape))

    # ------------------------------------------------------------------
    # Matrix operations (symbolic extended)
    # ------------------------------------------------------------------
    def _calculate_inverse_steps(self, A_data):
        if self._symbolic_mode:
            A = sp.Matrix(A_data)
        else:
            A = np.array(A_data, dtype=self._current_dtype)
        n = A.shape[0]
        if self._symbolic_mode:
            aug = A.row_join(sp.eye(n))
        else:
            aug = np.hstack([A, np.eye(n, dtype=self._current_dtype)])
        steps_log = []
        current_step = 0
        steps_log.append({'step': current_step, 'desc': Language.tr('step_initial_aug'), 'state': aug})
        current_step += 1

        for col in range(n):
            if self._symbolic_mode:
                pivot_val = aug[col, col]
                if pivot_val == 0:
                    raise sp.MatrixError(Language.tr('err_singular_matrix'))
            else:
                max_row_index = col + np.argmax(np.abs(aug[col:, col]))
                if not np.isclose(aug[max_row_index, col], 0):
                    if max_row_index != col:
                        aug[[col, max_row_index]] = aug[[max_row_index, col]]
                        steps_log.append({'step': current_step, 'desc': Language.tr('step_swap', row1=col+1, row2=max_row_index+1), 'state': aug})
                        current_step += 1
                pivot_val = aug[col, col]
                if np.isclose(pivot_val, 0):
                    raise np.linalg.LinAlgError(Language.tr('err_singular_matrix'))

            aug[col, :] = aug[col, :] / pivot_val
            steps_log.append({'step': current_step, 'desc': Language.tr('step_normalize', row=col+1, value=pivot_val), 'state': aug})
            current_step += 1

            for i in range(n):
                if i != col:
                    factor = aug[i, col]
                    if self._symbolic_mode and factor == 0:
                        continue
                    if not self._symbolic_mode and np.isclose(factor, 0):
                        continue
                    aug[i, :] = aug[i, :] - factor * aug[col, :]
                    steps_log.append({'step': current_step, 'desc': Language.tr('step_eliminate', col=col+1, row=i+1, factor=factor), 'state': aug})
                    current_step += 1

        inverse = aug[:, n:]
        return inverse, steps_log

    def inverse_matrix(self, A, show_steps=False):
        if not isinstance(A, (np.ndarray, sp.Matrix, list)):
            A = np.array(A)
        if self._symbolic_mode:
            A = sp.Matrix(A)
        else:
            A = np.array(A)
        self._validate_square_matrix(A, "Matrix inversion")
        if show_steps:
            result, log = self._calculate_inverse_steps(A)
            return result, log
        else:
            try:
                if self._symbolic_mode:
                    return A.inv(), None
                else:
                    if self.gpu_available:
                        import cupy as cp
                        A_gpu = cp.asarray(A)
                        result = cp.linalg.inv(A_gpu).get()
                    else:
                        result = np.linalg.inv(A.astype(self._current_dtype))
                    return result, None
            except Exception as e:
                raise ValueError(Language.tr('err_inversion_failed', msg=str(e)))

    def _calculate_determinant_steps(self, A_data):
        if self._symbolic_mode:
            A = sp.Matrix(A_data)
        else:
            A = np.array(A_data, dtype=self._current_dtype)
        n = A.shape[0]
        steps_log = []
        current_step = 0
        sign = 1
        steps_log.append({'step': current_step, 'desc': Language.tr('step_initial'), 'state': A})
        current_step += 1

        if self._symbolic_mode:
            det = A.det()
            steps_log.append({'step': current_step, 'desc': Language.tr('step_det_result', det=det), 'state': None})
            return det, steps_log

        for col in range(n):
            pivot_row = np.argmax(np.abs(A[col:, col])) + col
            if np.isclose(A[pivot_row, col], 0):
                steps_log.append({'step': current_step, 'desc': Language.tr('step_pivot_zero', col=col+1), 'state': A})
                steps_log.append({'step': current_step+1, 'desc': Language.tr('step_singular'), 'state': None})
                return 0.0, steps_log
            if pivot_row != col:
                A[[col, pivot_row]] = A[[pivot_row, col]]
                sign *= -1
                steps_log.append({'step': current_step, 'desc': Language.tr('step_swap', row1=col+1, row2=pivot_row+1), 'state': A})
                current_step += 1
            pivot_val = A[col, col]
            for i in range(col+1, n):
                if not np.isclose(A[i, col], 0):
                    factor = A[i, col] / pivot_val
                    A[i, col:] -= factor * A[col, col:]
                    steps_log.append({'step': current_step, 'desc': Language.tr('step_eliminate', col=col+1, row=i+1, factor=factor), 'state': A})
                    current_step += 1

        diag_product = np.prod(np.diag(A))
        det = sign * diag_product
        steps_log.append({'step': current_step, 'desc': Language.tr('step_diag_product', product=diag_product, sign=sign), 'state': A})
        steps_log.append({'step': current_step+1, 'desc': Language.tr('step_det_result', det=det), 'state': None})
        return det, steps_log

    def determinant_matrix(self, A, show_steps=False):
        if self._symbolic_mode:
            A = sp.Matrix(A)
        else:
            A = np.array(A)
        self._validate_square_matrix(A, "Determinant calculation")
        if show_steps:
            return self._calculate_determinant_steps(A)
        else:
            if self._symbolic_mode:
                return A.det(), None
            else:
                return np.linalg.det(A.astype(self._current_dtype)), None

    def _calculate_rank_steps(self, A):
        A = A.astype(self._current_dtype, copy=True)
        n_rows, n_cols = A.shape
        steps_log = []
        current_step = 0
        rank = 0
        row = 0
        col = 0
        steps_log.append({'step': current_step, 'desc': Language.tr('step_initial'), 'state': A.copy()})
        current_step += 1
        while row < n_rows and col < n_cols:
            pivot_row = np.argmax(np.abs(A[row:, col])) + row
            if np.isclose(A[pivot_row, col], 0):
                col += 1
                continue
            if pivot_row != row:
                A[[row, pivot_row]] = A[[pivot_row, row]]
                steps_log.append({'step': current_step, 'desc': Language.tr('step_swap', row1=row+1, row2=pivot_row+1), 'state': A.copy()})
                current_step += 1
            for i in range(row+1, n_rows):
                if not np.isclose(A[i, col], 0):
                    factor = A[i, col] / A[row, col]
                    A[i, col:] -= factor * A[row, col:]
                    steps_log.append({'step': current_step, 'desc': Language.tr('step_eliminate', col=col+1, row=i+1, factor=factor), 'state': A.copy()})
                    current_step += 1
            rank += 1
            row += 1
            col += 1
        steps_log.append({'step': current_step, 'desc': Language.tr('step_rank_final', rank=rank), 'state': None})
        return rank, steps_log

    def rank_matrix(self, A, show_steps=False):
        if self._symbolic_mode:
            A = sp.Matrix(A)
            return A.rank(), None
        else:
            A = np.array(A)
            if show_steps:
                return self._calculate_rank_steps(A)
            else:
                return np.linalg.matrix_rank(A), None

    def _solve_system_steps(self, A, B):
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
        steps_log.append({'step': current_step, 'desc': Language.tr('step_initial_aug'), 'state': aug.copy()})
        current_step += 1

        for col in range(n):
            pivot_row = np.argmax(np.abs(aug[col:, col])) + col
            if np.isclose(aug[pivot_row, col], 0):
                raise np.linalg.LinAlgError(Language.tr('err_singular_matrix'))
            if pivot_row != col:
                aug[[col, pivot_row]] = aug[[pivot_row, col]]
                steps_log.append({'step': current_step, 'desc': Language.tr('step_swap', row1=col+1, row2=pivot_row+1), 'state': aug.copy()})
                current_step += 1
            for i in range(col+1, n):
                if not np.isclose(aug[i, col], 0):
                    factor = aug[i, col] / aug[col, col]
                    aug[i, col:] -= factor * aug[col, col:]
                    steps_log.append({'step': current_step, 'desc': Language.tr('step_eliminate', col=col+1, row=i+1, factor=factor), 'state': aug.copy()})
                    current_step += 1

        U = aug[:, :n]
        rhs = aug[:, n:]
        X = np.zeros((n, m), dtype=aug.dtype)
        for i in reversed(range(n)):
            sum_ax = np.zeros(m)
            for j in range(i+1, n):
                sum_ax += U[i, j] * X[j]
            X[i] = (rhs[i] - sum_ax) / U[i, i]
            steps_log.append({'step': current_step, 'desc': Language.tr('step_back_subst', i=i+1, value=X[i]), 'state': X.copy() if m == 1 else None})
            current_step += 1
        if vector_rhs:
            X = X.flatten()
        return X, steps_log

    def solve_system(self, A, B, show_steps=False):
        if self._symbolic_mode:
            A = sp.Matrix(A)
            B = sp.Matrix(B)
            if show_steps:
                sol = A.LUsolve(B)
                steps = [{'step': 0, 'desc': 'Symbolic solution', 'state': sol}]
                return sol, steps
            else:
                return A.LUsolve(B), None
        else:
            A = np.array(A)
            B = np.array(B)
            self._validate_square_matrix(A, "System solution")
            if A.shape[0] != B.shape[0]:
                raise ValueError(Language.tr('err_system_rows'))
            if show_steps:
                return self._solve_system_steps(A, B)
            else:
                try:
                    return np.linalg.solve(A.astype(self._current_dtype), B.astype(self._current_dtype)), None
                except np.linalg.LinAlgError as e:
                    raise ValueError(Language.tr('err_no_unique_solution', msg=str(e)))

    def add_matrices(self, A, B):
        if self._symbolic_mode:
            A = sp.Matrix(A)
            B = sp.Matrix(B)
            self._validate_shapes(A, B, "Matrix addition")
            return A + B, None
        else:
            A, B = np.array(A), np.array(B)
            self._validate_shapes(A, B, "Matrix addition")
            return A + B, None

    def subtract_matrices(self, A, B):
        if self._symbolic_mode:
            A = sp.Matrix(A)
            B = sp.Matrix(B)
            self._validate_shapes(A, B, "Matrix subtraction")
            return A - B, None
        else:
            A, B = np.array(A), np.array(B)
            self._validate_shapes(A, B, "Matrix subtraction")
            return A - B, None

    def multiply_matrices(self, A, B):
        if self._symbolic_mode:
            A = sp.Matrix(A)
            B = sp.Matrix(B)
            if A.cols != B.rows:
                raise ValueError(Language.tr('err_incompatible_mul', shape1=A.shape, shape2=B.shape))
            return A * B, None
        else:
            A, B = np.array(A), np.array(B)
            if A.shape[1] != B.shape[0]:
                raise ValueError(Language.tr('err_incompatible_mul', shape1=A.shape, shape2=B.shape))
            return A @ B, None

    def scalar_multiply(self, A, scalar):
        if self._symbolic_mode:
            A = sp.Matrix(A)
            scalar = self._parse_expression(scalar)
            return scalar * A, None
        else:
            A = np.array(A)
            return A * scalar, None

    def transpose_matrix(self, A):
        if self._symbolic_mode:
            A = sp.Matrix(A)
            return A.T, None
        else:
            A = np.array(A)
            return A.T, None

    # ------------------------------------------------------------------
    # Vector operations (symbolic extended)
    # ------------------------------------------------------------------
    def _format_vector(self, v):
        if self._symbolic_mode:
            return str(v).replace('Matrix([[', '(').replace(']])', ')')
        else:
            return "(" + ", ".join(f"{x:.4g}" for x in v) + ")"

    def vector_add(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "addition")
        if show_steps:
            steps = [{'step': 0, 'desc': Language.tr('step_vector_add_init',
                                                     v1=self._format_vector(v1), v2=self._format_vector(v2)),
                      'state': None}]
            result = v1 + v2
            steps.append({'step': 1, 'desc': Language.tr('step_vector_add_result',
                                                         res=self._format_vector(result)), 'state': result})
            return result, steps
        return v1 + v2, None

    def vector_subtract(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "subtraction")
        if show_steps:
            steps = [{'step': 0, 'desc': Language.tr('step_vector_sub_init',
                                                     v1=self._format_vector(v1), v2=self._format_vector(v2)),
                      'state': None}]
            result = v1 - v2
            steps.append({'step': 1, 'desc': Language.tr('step_vector_sub_result',
                                                         res=self._format_vector(result)), 'state': result})
            return result, steps
        return v1 - v2, None

    def vector_scalar_multiply(self, v, scalar, show_steps=False):
        v = self._as_vector(v)
        scalar = self._parse_expression(scalar)
        if show_steps:
            steps = [{'step': 0, 'desc': Language.tr('step_vector_scale_init',
                                                     v=self._format_vector(v), scalar=scalar), 'state': None}]
            result = scalar * v
            steps.append({'step': 1, 'desc': Language.tr('step_vector_scale_result',
                                                         res=self._format_vector(result)), 'state': result})
            return result, steps
        return scalar * v, None

    def vector_dot(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "dot product")
        if self._symbolic_mode:
            dot_val = v1.dot(v2)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_dot_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2)),
                          'state': None}]
                products = [v1[i] * v2[i] for i in range(len(v1))]
                steps.append({'step': 1, 'desc': Language.tr('step_dot_products',
                                                             prods=str(products)), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_dot_sum', sum=dot_val), 'state': None})
                return dot_val, steps
            return dot_val, None
        else:
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_dot_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2)),
                          'state': None}]
                # Show element-wise multiplication with indices
                products = v1 * v2
                prod_desc = " + ".join([f"({v1[i]:.4g}·{v2[i]:.4g})" for i in range(len(v1))])
                steps.append({'step': 1, 'desc': Language.tr('step_dot_products_detail',
                                                             detail=prod_desc, prods=self._format_vector(products)),
                              'state': products})
                dot_val = float(np.sum(products))
                steps.append({'step': 2, 'desc': Language.tr('step_dot_sum', sum=dot_val), 'state': None})
                return dot_val, steps
            return float(np.dot(v1, v2)), None

    def vector_cross(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        if len(v1) != 3 or len(v2) != 3:
            raise ValueError(Language.tr('err_cross_3d', len1=len(v1), len2=len(v2)))

        if self._symbolic_mode:
            cross = v1.cross(v2)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_cross_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2)),
                          'state': None}, {'step': 1, 'desc': Language.tr('step_cross_determinant',
                                                                          i=v1[1] * v2[2] - v1[2] * v2[1],
                                                                          j=v1[2] * v2[0] - v1[0] * v2[2],
                                                                          k=v1[0] * v2[1] - v1[1] * v2[0]),
                                           'state': None}]
                # Show determinant formula
                # Show each component calculation
                x = v1[1] * v2[2] - v1[2] * v2[1]
                y = v1[2] * v2[0] - v1[0] * v2[2]
                z = v1[0] * v2[1] - v1[1] * v2[0]
                steps.append({'step': 2, 'desc': Language.tr('step_cross_components',
                                                             x=x, y=y, z=z), 'state': None})
                steps.append({'step': 3, 'desc': Language.tr('step_cross_result',
                                                             res=self._format_vector(cross)), 'state': cross})
                return cross, steps
            return cross, None
        else:
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_cross_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2)),
                          'state': None}, {'step': 1, 'desc': Language.tr('step_cross_determinant_numeric',
                                                                          a1=v1[0], a2=v1[1], a3=v1[2],
                                                                          b1=v2[0], b2=v2[1], b3=v2[2]), 'state': None}]
                # Show determinant expansion
                # Calculate components with formulas
                x = v1[1] * v2[2] - v1[2] * v2[1]
                y = v1[2] * v2[0] - v1[0] * v2[2]
                z = v1[0] * v2[1] - v1[1] * v2[0]
                steps.append({'step': 2, 'desc': Language.tr('step_cross_components_calc',
                                                             x_expr=f"{v1[1]:.4g}·{v2[2]:.4g} - {v1[2]:.4g}·{v2[1]:.4g} = {x:.4f}",
                                                             y_expr=f"{v1[2]:.4g}·{v2[0]:.4g} - {v1[0]:.4g}·{v2[2]:.4g} = {y:.4f}",
                                                             z_expr=f"{v1[0]:.4g}·{v2[1]:.4g} - {v1[1]:.4g}·{v2[0]:.4g} = {z:.4f}"),
                              'state': None})
                result = np.array([x, y, z], dtype=self._current_dtype)
                steps.append({'step': 3, 'desc': Language.tr('step_cross_result',
                                                             res=self._format_vector(result)), 'state': result})
                return result, steps
            return np.cross(v1, v2), None

    def vector_norm(self, v, show_steps=False):
        v = self._as_vector(v)
        if self._symbolic_mode:
            norm = sp.sqrt(v.dot(v))
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_norm_init', v=self._format_vector(v)), 'state': None}]
                squares = [x**2 for x in v]
                steps.append({'step': 1, 'desc': Language.tr('step_norm_squares', squares=str(squares)), 'state': None})
                sum_sq = sum(squares)
                steps.append({'step': 2, 'desc': Language.tr('step_norm_sum_sq', sum_sq=sum_sq), 'state': None})
                steps.append({'step': 3, 'desc': Language.tr('step_norm_result', norm=norm), 'state': None})
                return norm, steps
            return norm, None
        else:
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_norm_init', v=self._format_vector(v)), 'state': None}]
                squares = v ** 2
                steps.append({'step': 1, 'desc': Language.tr('step_norm_squares', squares=self._format_vector(squares)), 'state': squares})
                sum_sq = float(np.sum(squares))
                steps.append({'step': 2, 'desc': Language.tr('step_norm_sum_sq', sum_sq=sum_sq), 'state': None})
                norm = np.sqrt(sum_sq)
                steps.append({'step': 3, 'desc': Language.tr('step_norm_result', norm=norm), 'state': None})
                return norm, steps
            return float(np.linalg.norm(v)), None

    def vector_normalize(self, v, show_steps=False):
        v = self._as_vector(v)
        if self._symbolic_mode:
            norm = sp.sqrt(v.dot(v))
            if norm == 0:
                raise ValueError(Language.tr('err_normalize_zero'))
            unit = v / norm
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_normalize_init', v=self._format_vector(v)), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_normalize_norm', norm=norm), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_normalize_result', unit=self._format_vector(unit)), 'state': unit})
                return unit, steps
            return unit, None
        else:
            norm = np.linalg.norm(v)
            if np.isclose(norm, 0.0):
                raise ValueError(Language.tr('err_normalize_zero'))
            unit = v / norm
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_normalize_init', v=self._format_vector(v)), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_normalize_norm', norm=norm), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_normalize_result', unit=self._format_vector(unit)), 'state': unit})
                return unit, steps
            return unit, None

    def vector_projection(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "projection")
        if self._symbolic_mode:
            dot = v1.dot(v2)
            norm_sq = v2.dot(v2)
            if norm_sq == 0:
                raise ValueError(Language.tr('err_projection_zero_vec'))
            scalar = dot / norm_sq
            proj = scalar * v2
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_proj_init', v1=self._format_vector(v1), v2=self._format_vector(v2)), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_proj_dot', dot=dot), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_proj_norm_sq', norm_sq=norm_sq), 'state': None})
                steps.append({'step': 3, 'desc': Language.tr('step_proj_scalar', scalar=scalar), 'state': None})
                steps.append({'step': 4, 'desc': Language.tr('step_proj_result', proj=self._format_vector(proj)), 'state': proj})
                return proj, steps
            return proj, None
        else:
            dot = float(np.dot(v1, v2))
            norm_sq = float(np.dot(v2, v2))
            if np.isclose(norm_sq, 0.0):
                raise ValueError(Language.tr('err_projection_zero_vec'))
            scalar = dot / norm_sq
            proj = scalar * v2
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_proj_init', v1=self._format_vector(v1), v2=self._format_vector(v2)), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_proj_dot', dot=dot), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_proj_norm_sq', norm_sq=norm_sq), 'state': None})
                steps.append({'step': 3, 'desc': Language.tr('step_proj_scalar', scalar=scalar), 'state': None})
                steps.append({'step': 4, 'desc': Language.tr('step_proj_result', proj=self._format_vector(proj)), 'state': proj})
                return proj, steps
            return proj, None

    def vector_angle(self, v1, v2, show_steps=False, degrees=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "angle")
        if self._symbolic_mode:
            dot = v1.dot(v2)
            norm1 = sp.sqrt(v1.dot(v1))
            norm2 = sp.sqrt(v2.dot(v2))
            if norm1 == 0 or norm2 == 0:
                raise ValueError(Language.tr('err_angle_zero_vec'))
            cos_theta = dot / (norm1 * norm2)
            theta_rad = sp.acos(cos_theta)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_angle_init', v1=self._format_vector(v1), v2=self._format_vector(v2)), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_angle_dot', dot=dot), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_angle_norms', norm1=norm1, norm2=norm2), 'state': None})
                steps.append({'step': 3, 'desc': Language.tr('step_angle_cos', cos=cos_theta), 'state': None})
                steps.append({'step': 4, 'desc': Language.tr('step_angle_rad', rad=theta_rad), 'state': None})
                if degrees:
                    theta_deg = sp.deg(theta_rad)
                    steps.append({'step': 5, 'desc': Language.tr('step_angle_deg', deg=theta_deg), 'state': None})
                    return theta_deg, steps
                return theta_rad, steps
            return theta_rad if not degrees else sp.deg(theta_rad), None
        else:
            dot = float(np.dot(v1, v2))
            norm1 = float(np.linalg.norm(v1))
            norm2 = float(np.linalg.norm(v2))
            if np.isclose(norm1, 0.0) or np.isclose(norm2, 0.0):
                raise ValueError(Language.tr('err_angle_zero_vec'))
            cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
            theta_rad = float(np.arccos(cos_theta))
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_angle_init', v1=self._format_vector(v1), v2=self._format_vector(v2)), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_angle_dot', dot=dot), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_angle_norms', norm1=norm1, norm2=norm2), 'state': None})
                steps.append({'step': 3, 'desc': Language.tr('step_angle_cos', cos=cos_theta), 'state': None})
                steps.append({'step': 4, 'desc': Language.tr('step_angle_rad', rad=theta_rad), 'state': None})
                if degrees:
                    theta_deg = np.degrees(theta_rad)
                    steps.append({'step': 5, 'desc': Language.tr('step_angle_deg', deg=theta_deg), 'state': None})
                    return theta_deg, steps
                return theta_rad, steps
            return np.degrees(theta_rad) if degrees else theta_rad, None

    def vector_triple_scalar(self, v1, v2, v3, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        v3 = self._as_vector(v3)
        if not (len(v1) == len(v2) == len(v3) == 3):
            raise ValueError(Language.tr('err_triple_3d'))

        if self._symbolic_mode:
            cross = v2.cross(v3)
            dot = v1.dot(cross)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_triple_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2),
                                                         v3=self._format_vector(v3)), 'state': None},
                         {'step': 1, 'desc': Language.tr('step_triple_cross_start',
                                                         v2=self._format_vector(v2), v3=self._format_vector(v3)),
                          'state': None}]
                # Step 1: Show cross product of v2 and v3
                # Cross product components (symbolic)
                cx = v2[1] * v3[2] - v2[2] * v3[1]
                cy = v2[2] * v3[0] - v2[0] * v3[2]
                cz = v2[0] * v3[1] - v2[1] * v3[0]
                steps.append({'step': 2, 'desc': Language.tr('step_triple_cross_components',
                                                             x=cx, y=cy, z=cz), 'state': None})
                steps.append({'step': 3, 'desc': Language.tr('step_triple_cross_result',
                                                             cross=self._format_vector(cross)), 'state': cross})
                # Step 4: Dot product with v1
                steps.append({'step': 4, 'desc': Language.tr('step_triple_dot_start',
                                                             v1=self._format_vector(v1),
                                                             cross=self._format_vector(cross)), 'state': None})
                # Element-wise products
                products = [v1[i] * cross[i] for i in range(3)]
                steps.append({'step': 5, 'desc': Language.tr('step_triple_dot_products',
                                                             prods=str(products)), 'state': None})
                steps.append({'step': 6, 'desc': Language.tr('step_triple_dot_sum',
                                                             sum=dot), 'state': None})
                # Final result
                steps.append({'step': 7, 'desc': Language.tr('step_triple_result',
                                                             result=dot), 'state': None})
                return dot, steps
            return dot, None
        else:
            cross = np.cross(v2, v3)
            dot = float(np.dot(v1, cross))
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_triple_init',
                                                         v1=self._format_vector(v1), v2=self._format_vector(v2),
                                                         v3=self._format_vector(v3)), 'state': None},
                         {'step': 1, 'desc': Language.tr('step_triple_cross_start_numeric',
                                                         v2=self._format_vector(v2), v3=self._format_vector(v3)),
                          'state': None}]
                # Step 1: Cross product calculation
                # Compute cross components with formulas
                x = v2[1] * v3[2] - v2[2] * v3[1]
                y = v2[2] * v3[0] - v2[0] * v3[2]
                z = v2[0] * v3[1] - v2[1] * v3[0]
                steps.append({'step': 2, 'desc': Language.tr('step_triple_cross_components_calc',
                                                             x_expr=f"{v2[1]:.4g}·{v3[2]:.4g} - {v2[2]:.4g}·{v3[1]:.4g} = {x:.4f}",
                                                             y_expr=f"{v2[2]:.4g}·{v3[0]:.4g} - {v2[0]:.4g}·{v3[2]:.4g} = {y:.4f}",
                                                             z_expr=f"{v2[0]:.4g}·{v3[1]:.4g} - {v2[1]:.4g}·{v3[0]:.4g} = {z:.4f}"),
                              'state': None})
                steps.append({'step': 3, 'desc': Language.tr('step_triple_cross_result',
                                                             cross=self._format_vector(cross)), 'state': cross})
                # Step 4: Dot product with v1
                steps.append({'step': 4, 'desc': Language.tr('step_triple_dot_start_numeric',
                                                             v1=self._format_vector(v1),
                                                             cross=self._format_vector(cross)), 'state': None})
                # Show element-wise multiplication
                prod_expr = " + ".join([f"({v1[i]:.4g}·{cross[i]:.4g})" for i in range(3)])
                steps.append({'step': 5, 'desc': Language.tr('step_triple_dot_products_detail',
                                                             detail=prod_expr, prods=self._format_vector(v1 * cross)),
                              'state': v1 * cross})
                steps.append({'step': 6, 'desc': Language.tr('step_triple_dot_sum',
                                                             sum=dot), 'state': None})
                # Final result with volume interpretation
                steps.append({'step': 7, 'desc': Language.tr('step_triple_result_volume',
                                                             result=dot, volume=abs(dot)), 'state': None})
                return dot, steps
            return dot, None

    # ----------------------------------------------------------------------
    # Phase 2: Vector Special Relations, Basis, Geometry
    # ----------------------------------------------------------------------

    def are_collinear(self, v1, v2) -> bool:
        """Check if two vectors are collinear (2D or 3D)."""
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        if len(v1) != len(v2):
            return False
        if self._symbolic_mode:
            # Cross product zero vector
            if len(v1) == 2:
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                return cross == 0
            else:
                cross = v1.cross(v2)
                return cross == sp.Matrix([0, 0, 0])
        else:
            if len(v1) == 2:
                return np.isclose(v1[0] * v2[1] - v1[1] * v2[0], 0)
            else:
                return np.allclose(np.cross(v1, v2), 0)

    def find_collinearity_parameter(self, v1_expr, v2_expr, param_name='λ', show_steps=False):
        """
        Given two vectors (lists of strings/expressions) containing a parameter,
        find parameter values making them collinear.
        Returns solutions and steps.
        """
        # Parse expressions
        v1 = [self._parse_expression(comp) for comp in v1_expr]
        v2 = [self._parse_expression(comp) for comp in v2_expr]
        param = sp.Symbol(param_name)
        steps = []

        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_collinear_init',
                                                         v1=str(v1), v2=str(v2)), 'state': None})

        if len(v1) == 2:
            eq = v1[0] * v2[1] - v1[1] * v2[0]
            if show_steps:
                steps.append({'step': 1, 'desc': Language.tr('step_collinear_2d_eq',
                                                             eq=f"{eq} = 0"), 'state': None})
        else:
            cross = sp.Matrix(v1).cross(sp.Matrix(v2))
            # Find first non-zero component equation
            non_zero = [comp for comp in cross if comp != 0]
            if non_zero:
                eq = non_zero[0]
            else:
                eq = sp.S.Zero
            if show_steps:
                steps.append({'step': 1, 'desc': Language.tr('step_collinear_3d_cross',
                                                             cross=str(cross)), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_collinear_3d_eq',
                                                             eq=f"{eq} = 0"), 'state': None})

        solutions = sp.solve(eq, param)
        if show_steps:
            steps.append({'step': len(steps), 'desc': Language.tr('step_collinear_solve',
                                                                  param=param, solutions=str(solutions)),
                          'state': None})

        return solutions, steps

    def are_coplanar(self, v1, v2, v3) -> bool:
        """Check if three 3D vectors are coplanar (scalar triple product == 0)."""
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        v3 = self._as_vector(v3)
        if not (len(v1) == len(v2) == len(v3) == 3):
            raise ValueError(Language.tr('err_coplanar_3d'))
        if self._symbolic_mode:
            triple = v1.dot(v2.cross(v3))
            return triple == 0
        else:
            return np.isclose(np.dot(v1, np.cross(v2, v3)), 0)

    def is_orthogonal(self, v1, v2) -> bool:
        """Check if two vectors are orthogonal (dot product == 0)."""
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "orthogonality")
        if self._symbolic_mode:
            return v1.dot(v2) == 0
        else:
            return np.isclose(np.dot(v1, v2), 0)

    def is_basis(self, vectors) -> bool:
        """Check if set of vectors forms a basis (linearly independent)."""
        if self._symbolic_mode:
            M = sp.Matrix(vectors).T  # columns are vectors
            return M.det() != 0
        else:
            M = np.column_stack(vectors)
            return not np.isclose(np.linalg.det(M), 0)

    def decompose_vector(self, v, basis, show_steps=False):
        """
        Express vector v as linear combination of basis vectors.
        Returns coefficients (coordinates in the basis) and steps.
        """
        v = self._as_vector(v)
        basis = [self._as_vector(b) for b in basis]
        dim = len(v)
        if any(len(b) != dim for b in basis):
            raise ValueError(Language.tr('err_basis_dimension'))
        if len(basis) != dim:
            raise ValueError(Language.tr('err_basis_count', got=len(basis), expected=dim))

        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_decompose_init',
                                                         v=self._format_vector(v),
                                                         basis=str([self._format_vector(b) for b in basis])),
                          'state': None})

        if self._symbolic_mode:
            # Build matrix with basis vectors as columns
            M = sp.Matrix(basis).T
            if show_steps:
                steps.append({'step': 1, 'desc': Language.tr('step_decompose_matrix',
                                                             matrix=str(M)), 'state': M})
            # Solve M * x = v
            coeffs = M.LUsolve(sp.Matrix(v))
            if show_steps:
                steps.append({'step': 2, 'desc': Language.tr('step_decompose_solve',
                                                             system=f"{M}·x = {self._format_vector(v)}"),
                              'state': None})
                steps.append({'step': 3, 'desc': Language.tr('step_decompose_result',
                                                             coeffs=str(coeffs)), 'state': None})
            return list(coeffs), steps
        else:
            M = np.column_stack(basis)
            if show_steps:
                steps.append({'step': 1, 'desc': Language.tr('step_decompose_matrix_numeric',
                                                             matrix=str(M)), 'state': M})
            coeffs = np.linalg.solve(M, v)
            if show_steps:
                steps.append({'step': 2, 'desc': Language.tr('step_decompose_solve_numeric',
                                                             system=f"{M}·x = {self._format_vector(v)}"),
                              'state': None})
                steps.append({'step': 3, 'desc': Language.tr('step_decompose_result_numeric',
                                                             coeffs=self._format_vector(coeffs)), 'state': None})
            return coeffs, steps

    def change_of_basis_matrix(self, old_basis, new_basis, show_steps=False):
        """Compute transition matrix from old basis to new basis."""
        old = [self._as_vector(b) for b in old_basis]
        new = [self._as_vector(b) for b in new_basis]
        if self._symbolic_mode:
            M_old = sp.Matrix(old).T
            M_new = sp.Matrix(new).T
            # Transition: coordinates in new = P * coordinates in old
            P = M_new.inv() * M_old
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_transition_init'), 'state': None},
                         {'step': 1, 'desc': Language.tr('step_transition_formula'), 'state': None},
                         {'step': 2, 'desc': Language.tr('step_transition_matrix', matrix=str(P)), 'state': P}]
                return P, steps
            return P, None
        else:
            M_old = np.column_stack(old)
            M_new = np.column_stack(new)
            P = np.linalg.solve(M_new, M_old)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_transition_init'), 'state': None},
                         {'step': 1, 'desc': Language.tr('step_transition_formula'), 'state': None},
                         {'step': 2, 'desc': Language.tr('step_transition_matrix', matrix=str(P)), 'state': P}]
                return P, steps
            return P, None

    # ----------------------------------------------------------------------
    # Geometric primitives (points, area, volume)
    # ----------------------------------------------------------------------
    def vector_from_points(self, A, B):
        """Create vector from point A to point B."""
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        return B - A

    def points_collinear(self, A, B, C) -> bool:
        """Check if three points lie on a line."""
        AB = self.vector_from_points(A, B)
        AC = self.vector_from_points(A, C)
        return self.are_collinear(AB, AC)

    def points_coplanar(self, A, B, C, D) -> bool:
        """Check if four points lie in a plane."""
        AB = self.vector_from_points(A, B)
        AC = self.vector_from_points(A, C)
        AD = self.vector_from_points(A, D)
        return self.are_coplanar(AB, AC, AD)

    def triangle_area_points(self, A, B, C):
        """Area of triangle given three 3D points."""
        AB = self.vector_from_points(A, B)
        AC = self.vector_from_points(A, C)
        return self.triangle_area_vectors(AB, AC)

    def triangle_area_vectors(self, v1, v2):
        """Area of triangle from two side vectors."""
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        if len(v1) != 3 or len(v2) != 3:
            raise ValueError(Language.tr('err_3d_required'))
        cross = np.cross(v1, v2)
        return 0.5 * np.linalg.norm(cross)

    def tetrahedron_volume_points(self, A, B, C, D):
        """Volume of tetrahedron from four 3D points."""
        AB = self.vector_from_points(A, B)
        AC = self.vector_from_points(A, C)
        AD = self.vector_from_points(A, D)
        return abs(np.dot(AB, np.cross(AC, AD))) / 6.0

    def tetrahedron_volume_vectors(self, v1, v2, v3):
        """Volume from three edge vectors from one vertex."""
        return abs(np.dot(v1, np.cross(v2, v3))) / 6.0

    def characteristic_polynomial(self, A, var='λ', show_steps=False):
        """
        Return the characteristic polynomial det(A - λI).
        If symbolic mode, returns SymPy expression; else returns polynomial coefficients.
        """
        if self._symbolic_mode:
            A = sp.Matrix(A)
            λ = sp.Symbol(var)
            n = A.rows
            char_poly = (A - λ * sp.eye(n)).det()
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_charpoly_init', var=var), 'state': None},
                         {'step': 1, 'desc': Language.tr('step_charpoly_matrix',
                                                         matrix=str(A - λ * sp.eye(n))),
                          'state': A - λ * sp.eye(n)}, {'step': 2, 'desc': Language.tr('step_charpoly_det',
                                                                                       poly=str(char_poly)),
                                                        'state': None}]
                return char_poly, steps
            return char_poly, None
        else:
            A = np.array(A, dtype=self._current_dtype)
            n = A.shape[0]
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_charpoly_init', var=var), 'state': None},
                         {'step': 1, 'desc': Language.tr('step_charpoly_matrix_numeric',
                                                         matrix=str(A)), 'state': A}]
                coeffs = np.poly(A)
                steps.append({'step': 2, 'desc': Language.tr('step_charpoly_coeffs',
                                                             coeffs=str(coeffs)), 'state': None})
                return coeffs, steps
            return np.poly(A), None

    def eigenvalues(self, A, show_steps=False):
        """Return eigenvalues of matrix A."""
        if self._symbolic_mode:
            A = sp.Matrix(A)
            eigen = A.eigenvals()
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_eigenvals_init'), 'state': None}]
                char_poly = self.characteristic_polynomial(A, show_steps=False)[0]
                steps.append({'step': 1, 'desc': Language.tr('step_eigenvals_charpoly',
                                                             poly=str(char_poly)), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_eigenvals_roots',
                                                             vals=str(eigen)), 'state': None})
                return eigen, steps
            return eigen, None
        else:
            A = np.array(A, dtype=self._current_dtype)
            vals = np.linalg.eigvals(A)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_eigenvals_init'), 'state': None}]
                coeffs = np.poly(A)
                steps.append({'step': 1, 'desc': Language.tr('step_eigenvals_charpoly_numeric',
                                                             coeffs=str(coeffs)), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_eigenvals_result',
                                                             vals=str(vals)), 'state': None})
                return vals, steps
            return vals, None

    def eigenvectors(self, A, show_steps=False):
        """Return eigenvalues and eigenvectors. For symbolic, returns list of (eigenval, mult, [eigenvecs])."""
        if self._symbolic_mode:
            A = sp.Matrix(A)
            eigenvects = A.eigenvects()
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_eigenvecs_init'), 'state': None}]
                char_poly = self.characteristic_polynomial(A, show_steps=False)[0]
                steps.append({'step': 1, 'desc': Language.tr('step_eigenvecs_charpoly',
                                                             poly=str(char_poly)), 'state': None})
                # Show each eigenvalue and its eigenvectors
                step_num = 2
                for val, mult, vecs in eigenvects:
                    steps.append({'step': step_num, 'desc': Language.tr('step_eigenvecs_for_val',
                                                                        val=str(val), mult=mult), 'state': None})
                    step_num += 1
                    for i, vec in enumerate(vecs):
                        steps.append({'step': step_num, 'desc': Language.tr('step_eigenvecs_vec',
                                                                            idx=i + 1, vec=self._format_vector(vec)),
                                      'state': vec})
                        step_num += 1
                return eigenvects, steps
            return eigenvects, None
        else:
            A = np.array(A, dtype=self._current_dtype)
            vals, vecs = np.linalg.eig(A)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_eigenvecs_init'), 'state': None},
                         {'step': 1, 'desc': Language.tr('step_eigenvecs_vals',
                                                         vals=str(vals)), 'state': None},
                         {'step': 2, 'desc': Language.tr('step_eigenvecs_matrix',
                                                         matrix=str(vecs)), 'state': vecs}]
                return (vals, vecs), steps
            return (vals, vecs), None

    def diagonalize(self, A, show_steps=False):
        """
        Attempt to diagonalize matrix A.
        Returns (P, D) such that A = P * D * P^{-1}, or raises error if not diagonalizable.
        """
        if self._symbolic_mode:
            A = sp.Matrix(A)
            if not A.is_diagonalizable():
                raise ValueError(Language.tr('err_not_diagonalizable'))
            P, D = A.diagonalize()
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_diag_init'), 'state': None},
                         {'step': 1, 'desc': Language.tr('step_diag_eigenvals',
                                                         vals=str(D.diagonal())), 'state': D},
                         {'step': 2, 'desc': Language.tr('step_diag_eigenvecs',
                                                         matrix=str(P)), 'state': P},
                         {'step': 3, 'desc': Language.tr('step_diag_verify',
                                                         product=str(P * D * P.inv())), 'state': None}]
                return (P, D), steps
            return (P, D), None
        else:
            A = np.array(A, dtype=self._current_dtype)
            vals, vecs = np.linalg.eig(A)
            # Check if diagonalizable (all eigenvectors independent)
            if np.linalg.matrix_rank(vecs) < A.shape[0]:
                raise ValueError(Language.tr('err_not_diagonalizable'))
            D = np.diag(vals)
            P = vecs
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_diag_init'), 'state': None},
                         {'step': 1, 'desc': Language.tr('step_diag_eigenvals_numeric',
                                                         vals=str(vals)), 'state': D},
                         {'step': 2, 'desc': Language.tr('step_diag_eigenvecs_numeric',
                                                         matrix=str(P)), 'state': P},
                         {'step': 3, 'desc': Language.tr('step_diag_verify_numeric',
                                                         product=str(P @ D @ np.linalg.inv(P))), 'state': None}]
                return (P, D), steps
            return (P, D), None

    def gram_schmidt(self, vectors, normalize=True, show_steps=False):
        """
        Perform Gram–Schmidt orthogonalization on a list of vectors.
        Returns orthogonal (and optionally orthonormal) basis.
        """
        if self._symbolic_mode:
            vecs = [sp.Matrix(v) for v in vectors]
        else:
            vecs = [np.asarray(v, dtype=self._current_dtype) for v in vectors]

        dim = vecs[0].shape[0]
        for v in vecs:
            if v.shape[0] != dim:
                raise ValueError(Language.tr('err_vectors_same_dim'))

        basis = []
        steps = []
        step_num = 0

        if show_steps:
            steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_init',
                                                                vectors=str([self._format_vector(v) for v in vecs])),
                          'state': None})
            step_num += 1

        for i, v in enumerate(vecs):
            u = v
            if show_steps:
                steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_start',
                                                                    idx=i + 1, v=self._format_vector(v)),
                              'state': None})
                step_num += 1

            for j, b in enumerate(basis):
                if self._symbolic_mode:
                    proj = (v.dot(b) / b.dot(b)) * b
                else:
                    proj = (np.dot(v, b) / np.dot(b, b)) * b
                u = u - proj
                if show_steps:
                    steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_subtract',
                                                                        j=j + 1, proj=self._format_vector(proj),
                                                                        u=self._format_vector(u)), 'state': u})
                    step_num += 1

            # Check linear independence
            if self._symbolic_mode:
                if u.norm() == 0:
                    if show_steps:
                        steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_dependent',
                                                                            idx=i + 1), 'state': None})
                        step_num += 1
                    continue
            else:
                if np.linalg.norm(u) < 1e-12:
                    if show_steps:
                        steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_dependent',
                                                                            idx=i + 1), 'state': None})
                        step_num += 1
                    continue

            basis.append(u)
            if show_steps:
                steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_orthogonal',
                                                                    idx=i + 1, u=self._format_vector(u)), 'state': u})
                step_num += 1

        if normalize:
            normalized = []
            for i, u in enumerate(basis):
                if self._symbolic_mode:
                    norm_u = u.norm()
                    e = u / norm_u
                else:
                    norm_u = np.linalg.norm(u)
                    e = u / norm_u
                normalized.append(e)
                if show_steps:
                    steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_normalize',
                                                                        idx=i + 1, u=self._format_vector(u),
                                                                        e=self._format_vector(e)), 'state': e})
                    step_num += 1
            basis = normalized

        if show_steps:
            steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_result',
                                                                basis=str([self._format_vector(b) for b in basis])),
                          'state': None})

        return basis, steps

    def stats(self):
        return {
            "device": self.device,
            "precision": self.precision.value,
            "symbolic_mode": self._symbolic_mode,
            "cache_hits": 0,
            "cache_misses": 0
        }
