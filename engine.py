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
                                                         v3=self._format_vector(v3)), 'state': None}]
                # Step 1: Show cross product of v2 and v3
                steps.append({'step': 1, 'desc': Language.tr('step_triple_cross_start',
                                                             v2=self._format_vector(v2), v3=self._format_vector(v3)),
                              'state': None})
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

    def stats(self):
        return {
            "device": self.device,
            "precision": self.precision.value,
            "symbolic_mode": self._symbolic_mode,
            "cache_hits": 0,
            "cache_misses": 0
        }
