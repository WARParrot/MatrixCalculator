import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from localization import Language
import config


class Point:
    def __init__(self, coords, name=None, engine=None):
        self.coords = sp.Matrix(coords) if engine and engine.get_symbolic_mode() else np.array(coords, dtype=float)
        self.name = name or "P"
        self._engine = engine

    def _format(self):
        if isinstance(self.coords, sp.Matrix):
            return str(self.coords).replace('Matrix([[', '(').replace(']])', ')')
        return "(" + ", ".join(f"{x:.4g}" for x in self.coords) + ")"

    def __add__(self, vector):
        if isinstance(vector, (list, np.ndarray, sp.Matrix)):
            return Point(self.coords + vector, name=f"{self.name}+v", engine=self._engine)
        raise TypeError("Point can only be added to a vector")

    def __sub__(self, other):
        if isinstance(other, Point):
            return self.coords - other.coords
        elif isinstance(other, (list, np.ndarray, sp.Matrix)):
            return Point(self.coords - other, name=f"{self.name}-v", engine=self._engine)
        raise TypeError("Point subtraction requires another Point or a vector")


class AffineFrame:
    def __init__(self, origin, basis, engine=None):
        self.origin = origin
        self.basis = [sp.Matrix(v) if engine and engine.get_symbolic_mode() else np.array(v) for v in basis]
        self._engine = engine
        self._validate()

    def _validate(self):
        if len(self.basis) not in (2, 3):
            raise ValueError("Basis must have 2 or 3 vectors")
        M = sp.Matrix.hstack(*self.basis) if self._engine and self._engine.get_symbolic_mode() else np.column_stack(self.basis)
        det = M.det() if isinstance(M, sp.Matrix) else np.linalg.det(M)
        if det == 0 if isinstance(det, sp.Expr) else np.isclose(det, 0):
            raise ValueError("Basis vectors are linearly dependent")

    def point_to_coords(self, P, show_steps=False):
        v = P - self.origin
        if self._engine and self._engine.get_symbolic_mode():
            M = sp.Matrix.hstack(*self.basis)
            coords = M.LUsolve(sp.Matrix(v))
        else:
            M = np.column_stack(self.basis)
            coords = np.linalg.solve(M, v)
        if show_steps:
            steps = [
                {'step': 0, 'desc': f'Vector from origin: {P.name} - O = {v}', 'state': v},
                {'step': 1, 'desc': f'Basis matrix:\n{M}', 'state': M},
                {'step': 2, 'desc': f'Solve M * coords = v → coords = {coords}', 'state': coords},
            ]
            return coords, steps
        return coords

    def coords_to_point(self, coords, show_steps=False):
        if self._engine and self._engine.get_symbolic_mode():
            M = sp.Matrix.hstack(*self.basis)
            cart = self.origin.coords + M * sp.Matrix(coords)
        else:
            M = np.column_stack(self.basis)
            cart = self.origin.coords + M @ np.array(coords)
        if show_steps:
            steps = [
                {'step': 0, 'desc': f'Multiply basis by coords: {M} * {coords} = {cart - self.origin.coords}', 'state': None},
                {'step': 1, 'desc': f'Add origin: O + ... = {cart}', 'state': cart},
            ]
            return Point(cart, engine=self._engine), steps
        return Point(cart, engine=self._engine)


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
        self._symbolic_mode = False

    @property
    def gpu_available(self):
        return self._gpu_available

    def set_precision(self, prec):
        self.precision = prec

    @property
    def _current_dtype(self):
        if self.precision == config.ComputePrecision.FP32:
            return np.float32
        elif self.precision == config.ComputePrecision.FP64:
            return np.float64
        else:
            return np.float64

    def set_symbolic_mode(self, enabled):
        self._symbolic_mode = enabled

    def get_symbolic_mode(self):
        return self._symbolic_mode

    def _parse_expression(self, value):
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
                    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
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
        if isinstance(data, sp.Matrix):
            return data
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

    def _validate_vectors_same_length(self, v1, v2, op):
        if len(v1) != len(v2):
            raise ValueError(Language.tr('err_vector_same_len', op=op, len1=len(v1), len2=len(v2)))

    def _validate_square_matrix(self, A, operation="operation"):
        if A.shape[0] != A.shape[1]:
            raise ValueError(Language.tr('err_square_matrix', operation=operation, shape=A.shape))

    def _validate_shapes(self, A, B, operation):
        if A.shape != B.shape:
            raise ValueError(Language.tr('err_same_shape', operation=operation, shape1=A.shape, shape2=B.shape))

    # Matrix Operations
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

    def determinant_matrix(self, A, show_steps=False):
        if self._symbolic_mode:
            if not isinstance(A, sp.Matrix):
                A = self._to_sympy_matrix(A)
        else:
            A = np.array(A, dtype=self._current_dtype)
        self._validate_square_matrix(A, "Determinant calculation")
        if show_steps:
            return self._calculate_determinant_steps(A)
        else:
            if self._symbolic_mode:
                return A.det(), None
            else:
                return np.linalg.det(A.astype(self._current_dtype)), None

    def _calculate_determinant_steps(self, A_data):
        if self._symbolic_mode:
            A = A_data if isinstance(A_data, sp.Matrix) else self._to_sympy_matrix(A_data)
            n = A.rows
            steps_log = [{'step': 0, 'desc': Language.tr('step_initial'), 'state': A}]
            if n <= 3:
                det = A.det()
                if n == 1:
                    steps_log.append({'step': 1, 'desc': Language.tr('step_det_1x1', val=det), 'state': None})
                elif n == 2:
                    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
                    det = a*d - b*c
                    steps_log.append({'step': 1, 'desc': Language.tr('step_det_2x2', a=a, b=b, c=c, d=d, det=det), 'state': None})
                else:
                    steps_log.append({'step': 1, 'desc': Language.tr('step_det_3x3_expand'), 'state': None})
                    steps_log.append({'step': 2, 'desc': Language.tr('step_det_result', det=det), 'state': None})
            else:
                det = A.det()
                steps_log.append({'step': 1, 'desc': Language.tr('step_det_symbolic', det=det), 'state': None})
            return det, steps_log
        else:
            A = np.array(A_data, dtype=self._current_dtype)
            n = A.shape[0]
            steps_log = []
            current_step = 0
            sign = 1
            steps_log.append({'step': current_step, 'desc': Language.tr('step_initial'), 'state': A})
            current_step += 1
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

    def rank_matrix(self, A, show_steps=False):
        if self._symbolic_mode:
            if not isinstance(A, sp.Matrix):
                A = self._to_sympy_matrix(A)
            rank_val = A.rank()
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_rank_symbolic', rank=rank_val), 'state': A}]
                return rank_val, steps
            return rank_val, None
        else:
            A = np.array(A)
            if show_steps:
                return self._calculate_rank_steps(A)
            else:
                return np.linalg.matrix_rank(A), None

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

    def inverse_matrix(self, A, show_steps=False):
        if self._symbolic_mode:
            if not isinstance(A, sp.Matrix):
                A = self._to_sympy_matrix(A)
            self._validate_square_matrix(A, "Matrix inversion")
            if show_steps:
                return self._calculate_inverse_steps_symbolic(A)
            else:
                try:
                    return A.inv(), None
                except Exception as e:
                    raise ValueError(Language.tr('err_inversion_failed', msg=str(e)))
        else:
            A = np.array(A, dtype=self._current_dtype)
            self._validate_square_matrix(A, "Matrix inversion")
            if show_steps:
                return self._calculate_inverse_steps(A)
            else:
                try:
                    if self.gpu_available:
                        import cupy as cp
                        A_gpu = cp.asarray(A)
                        result = cp.linalg.inv(A_gpu).get()
                    else:
                        result = np.linalg.inv(A)
                    return result, None
                except Exception as e:
                    raise ValueError(Language.tr('err_inversion_failed', msg=str(e)))

    def _calculate_inverse_steps(self, A):
        n = A.shape[0]
        aug = np.hstack([A.astype(self._current_dtype), np.eye(n, dtype=self._current_dtype)])
        steps_log = []
        current_step = 0
        steps_log.append({'step': current_step, 'desc': Language.tr('step_initial_aug'), 'state': aug.copy()})
        current_step += 1
        for col in range(n):
            max_row_index = col + np.argmax(np.abs(aug[col:, col]))
            if not np.isclose(aug[max_row_index, col], 0):
                if max_row_index != col:
                    aug[[col, max_row_index]] = aug[[max_row_index, col]]
                    steps_log.append({'step': current_step, 'desc': Language.tr('step_swap', row1=col+1, row2=max_row_index+1), 'state': aug.copy()})
                    current_step += 1
            pivot_val = aug[col, col]
            if np.isclose(pivot_val, 0):
                raise np.linalg.LinAlgError(Language.tr('err_singular_matrix'))
            aug[col] /= pivot_val
            steps_log.append({'step': current_step, 'desc': Language.tr('step_normalize', row=col+1, value=pivot_val), 'state': aug.copy()})
            current_step += 1
            for i in range(n):
                if i != col:
                    factor = aug[i, col]
                    if not np.isclose(factor, 0):
                        aug[i] -= factor * aug[col]
                        steps_log.append({'step': current_step, 'desc': Language.tr('step_eliminate', col=col+1, row=i+1, factor=factor), 'state': aug.copy()})
                        current_step += 1
        inverse = aug[:, n:]
        return inverse, steps_log

    def _calculate_inverse_steps_symbolic(self, A):
        n = A.rows
        aug = A.row_join(sp.eye(n))
        steps_log = [{'step': 0, 'desc': Language.tr('step_initial_aug'), 'state': aug}]
        current_step = 1
        for col in range(n):
            pivot_val = aug[col, col]
            if pivot_val == 0:
                for r in range(col+1, n):
                    if aug[r, col] != 0:
                        aug.row_swap(col, r)
                        steps_log.append({'step': current_step, 'desc': Language.tr('step_swap', row1=col+1, row2=r+1), 'state': aug})
                        current_step += 1
                        pivot_val = aug[col, col]
                        break
            if pivot_val == 0:
                raise sp.MatrixError(Language.tr('err_singular_matrix'))
            aug[col, :] = aug[col, :] / pivot_val
            steps_log.append({'step': current_step, 'desc': Language.tr('step_normalize', row=col+1, value=pivot_val), 'state': aug})
            current_step += 1
            for i in range(n):
                if i != col:
                    factor = aug[i, col]
                    if factor != 0:
                        aug[i, :] = aug[i, :] - factor * aug[col, :]
                        steps_log.append({'step': current_step, 'desc': Language.tr('step_eliminate', col=col+1, row=i+1, factor=factor), 'state': aug})
                        current_step += 1
        inverse = aug[:, n:]
        steps_log.append({'step': current_step, 'desc': Language.tr('step_inverse_result_matrix'), 'state': inverse})
        return inverse, steps_log

    def solve_system(self, A, B, show_steps=False):
        if self._symbolic_mode:
            A = self._to_sympy_matrix(A) if not isinstance(A, sp.Matrix) else A
            B = sp.Matrix(B) if not isinstance(B, sp.Matrix) else B
            aug = A.row_join(B)
            if A.rank() < aug.rank():
                raise ValueError(Language.tr('err_inconsistent_system'))
            x = sp.symbols(f'x0:{A.cols}')
            sol = sp.linsolve((A, B), x)
            if not sol:
                raise ValueError(Language.tr('err_no_solution'))
            sol_list = list(sol)
            sol_tuple = sol_list[0] if sol_list else None
            rank_A = A.rank()
            nullity = A.cols - rank_A
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_slae_rank_symbolic',
                                                         rank=rank_A, nullity=nullity, free=nullity), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_slae_general_solution',
                                                             sol=sol_tuple), 'state': None})
                return sol_tuple, steps
            return sol_tuple, None
        else:
            A = np.array(A, dtype=self._current_dtype)
            B = np.array(B, dtype=self._current_dtype).flatten()
            if show_steps:
                return self._solve_system_steps(A, B)
            else:
                try:
                    return np.linalg.solve(A, B), None
                except np.linalg.LinAlgError:
                    x_part, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
                    u, s, vh = np.linalg.svd(A)
                    null_mask = s < 1e-12
                    null_space = vh[null_mask].T
                    return (x_part, null_space), None

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
        rank = np.sum(np.abs(np.diag(U)) > 1e-12)
        nullity = n - rank
        steps_log.append({'step': current_step, 'desc': Language.tr('step_slae_rank', rank=rank, nullity=nullity, free=nullity), 'state': None})
        current_step += 1
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

    def solve_cramer(self, A, B, show_steps=False):
        if self._symbolic_mode:
            A = sp.Matrix(A)
            B = sp.Matrix(B)
            if B.cols > 1:
                raise ValueError(Language.tr('err_cramer_single_rhs'))
            det_A = A.det()
            if det_A == 0:
                raise ValueError(Language.tr('err_singular_matrix'))
            n = A.rows
            x = sp.zeros(n, 1)
            steps = []
            if show_steps:
                steps.append({'step': 0, 'desc': Language.tr('step_cramer_init'), 'state': None})
                steps.append({'step': 1, 'desc': Language.tr('step_cramer_det_A', det=det_A), 'state': A})
                step_idx = 2
            for i in range(n):
                Ai = A.copy()
                Ai[:, i] = B
                det_Ai = Ai.det()
                xi = det_Ai / det_A
                x[i] = xi
                if show_steps:
                    steps.append({'step': step_idx, 'desc': Language.tr('step_cramer_replace', step=step_idx, col=i+1, det_i=det_Ai, xi=xi), 'state': Ai})
                    step_idx += 1
            if show_steps:
                steps.append({'step': step_idx, 'desc': Language.tr('step_cramer_result', x=self._format_vector(x)), 'state': x})
            return x, steps
        else:
            A = np.array(A, dtype=self._current_dtype)
            B = np.array(B, dtype=self._current_dtype).flatten()
            n = A.shape[0]
            det_A = np.linalg.det(A)
            if np.isclose(det_A, 0):
                raise ValueError(Language.tr('err_singular_matrix'))
            x = np.zeros(n, dtype=self._current_dtype)
            steps = []
            if show_steps:
                steps.append({'step': 0, 'desc': Language.tr('step_cramer_init'), 'state': None})
                steps.append({'step': 1, 'desc': Language.tr('step_cramer_det_A', det=det_A), 'state': A})
                step_idx = 2
            for i in range(n):
                Ai = A.copy()
                Ai[:, i] = B
                det_Ai = np.linalg.det(Ai)
                xi = det_Ai / det_A
                x[i] = xi
                if show_steps:
                    steps.append({'step': step_idx, 'desc': Language.tr('step_cramer_replace', step=step_idx, col=i+1, det_i=det_Ai, xi=xi), 'state': Ai})
                    step_idx += 1
            if show_steps:
                steps.append({'step': step_idx, 'desc': Language.tr('step_cramer_result', x=self._format_vector(x)), 'state': x})
            return x, steps

    def solve_inverse(self, A, B, show_steps=False):
        if self._symbolic_mode:
            A = sp.Matrix(A)
            B = sp.Matrix(B)
            inv_A = A.inv()
            x = inv_A * B
            steps = []
            if show_steps:
                steps.append({'step': 0, 'desc': Language.tr('step_inverse_init'), 'state': None})
                steps.append({'step': 1, 'desc': Language.tr('step_inverse_compute'), 'state': inv_A})
                steps.append({'step': 2, 'desc': Language.tr('step_inverse_multiply'), 'state': None})
                step_num = 3
                if B.cols == 1:
                    for i in range(A.rows):
                        expr = ' + '.join([f'({inv_A[i,j]})*({B[j]})' for j in range(A.cols)])
                        steps.append({'step': step_num, 'desc': Language.tr('step_inverse_component', i=i+1, expr=expr, val=x[i]), 'state': None})
                        step_num += 1
                    steps.append({'step': step_num, 'desc': Language.tr('step_inverse_result', x=self._format_vector(x)), 'state': x})
                else:
                    steps.append({'step': step_num, 'desc': Language.tr('step_inverse_result_matrix', x=str(x)), 'state': x})
            return x, steps
        else:
            A = np.array(A, dtype=self._current_dtype)
            B = np.array(B, dtype=self._current_dtype)
            inv_A = np.linalg.inv(A)
            x = inv_A @ B
            steps = []
            if show_steps:
                steps.append({'step': 0, 'desc': Language.tr('step_inverse_init'), 'state': None})
                steps.append({'step': 1, 'desc': Language.tr('step_inverse_compute'), 'state': inv_A})
                steps.append({'step': 2, 'desc': Language.tr('step_inverse_multiply'), 'state': None})
                step_num = 3
                if B.ndim == 1:
                    for i in range(A.shape[0]):
                        terms = [f'{inv_A[i,j]:.4f} * {B[j]:.4f}' for j in range(A.shape[1])]
                        expr = ' + '.join(terms)
                        val = float(x[i])
                        steps.append({'step': step_num, 'desc': Language.tr('step_inverse_component', i=i+1, expr=expr, val=val), 'state': None})
                        step_num += 1
                    steps.append({'step': step_num, 'desc': Language.tr('step_inverse_result', x=self._format_vector(x)), 'state': x})
                else:
                    steps.append({'step': step_num, 'desc': Language.tr('step_inverse_result_matrix', x=str(x)), 'state': x})
            return x, steps

    # Vector Operations
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
            steps = [{'step': 0, 'desc': Language.tr('step_vector_add_init', v1=self._format_vector(v1), v2=self._format_vector(v2)), 'state': None}]
            result = v1 + v2
            steps.append({'step': 1, 'desc': Language.tr('step_vector_add_result', res=self._format_vector(result)), 'state': result})
            return result, steps
        return v1 + v2, None

    def vector_subtract(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "subtraction")
        if show_steps:
            steps = [{'step': 0, 'desc': Language.tr('step_vector_sub_init', v1=self._format_vector(v1), v2=self._format_vector(v2)), 'state': None}]
            result = v1 - v2
            steps.append({'step': 1, 'desc': Language.tr('step_vector_sub_result', res=self._format_vector(result)), 'state': result})
            return result, steps
        return v1 - v2, None

    def vector_scalar_multiply(self, v, scalar, show_steps=False):
        v = self._as_vector(v)
        scalar = self._parse_expression(scalar)
        if show_steps:
            steps = [{'step': 0, 'desc': Language.tr('step_vector_scale_init', v=self._format_vector(v), scalar=scalar), 'state': None}]
            result = scalar * v
            steps.append({'step': 1, 'desc': Language.tr('step_vector_scale_result', res=self._format_vector(result)), 'state': result})
            return result, steps
        return scalar * v, None

    def vector_dot(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "dot product")
        if self._symbolic_mode:
            dot_val = v1.dot(v2)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_dot_init', v1=self._format_vector(v1), v2=self._format_vector(v2)), 'state': None}]
                products = [v1[i]*v2[i] for i in range(len(v1))]
                steps.append({'step': 1, 'desc': Language.tr('step_dot_products', prods=str(products)), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_dot_sum', sum=dot_val), 'state': None})
                return dot_val, steps
            return dot_val, None
        else:
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_dot_init', v1=self._format_vector(v1), v2=self._format_vector(v2)), 'state': None}]
                products = v1 * v2
                prod_desc = " + ".join([f"({v1[i]:.4g}·{v2[i]:.4g})" for i in range(len(v1))])
                steps.append({'step': 1, 'desc': Language.tr('step_dot_products_detail', detail=prod_desc, prods=self._format_vector(products)), 'state': products})
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
                steps = [{'step': 0, 'desc': Language.tr('step_cross_init', v1=self._format_vector(v1), v2=self._format_vector(v2)), 'state': None}]
                x = v1[1]*v2[2] - v1[2]*v2[1]
                y = v1[2]*v2[0] - v1[0]*v2[2]
                z = v1[0]*v2[1] - v1[1]*v2[0]
                steps.append({'step': 1, 'desc': Language.tr('step_cross_components', x=x, y=y, z=z), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_cross_result', res=self._format_vector(cross)), 'state': cross})
                return cross, steps
            return cross, None
        else:
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_cross_init', v1=self._format_vector(v1), v2=self._format_vector(v2)), 'state': None}]
                x = v1[1]*v2[2] - v1[2]*v2[1]
                y = v1[2]*v2[0] - v1[0]*v2[2]
                z = v1[0]*v2[1] - v1[1]*v2[0]
                steps.append({'step': 1, 'desc': Language.tr('step_cross_components_calc', x_expr=f"{v1[1]:.4g}·{v2[2]:.4g} - {v1[2]:.4g}·{v2[1]:.4g} = {x:.4f}", y_expr=f"{v1[2]:.4g}·{v2[0]:.4g} - {v1[0]:.4g}·{v2[2]:.4g} = {y:.4f}", z_expr=f"{v1[0]:.4g}·{v2[1]:.4g} - {v1[1]:.4g}·{v2[0]:.4g} = {z:.4f}"), 'state': None})
                result = np.array([x, y, z], dtype=self._current_dtype)
                steps.append({'step': 2, 'desc': Language.tr('step_cross_result', res=self._format_vector(result)), 'state': result})
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
                steps = [{'step': 0, 'desc': Language.tr('step_triple_init', v1=self._format_vector(v1), v2=self._format_vector(v2), v3=self._format_vector(v3)), 'state': None}]
                cx = v2[1]*v3[2] - v2[2]*v3[1]
                cy = v2[2]*v3[0] - v2[0]*v3[2]
                cz = v2[0]*v3[1] - v2[1]*v3[0]
                steps.append({'step': 1, 'desc': Language.tr('step_triple_cross_components', x=cx, y=cy, z=cz), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_triple_cross', cross=self._format_vector(cross)), 'state': cross})
                products = [v1[i] * cross[i] for i in range(3)]
                steps.append({'step': 3, 'desc': Language.tr('step_triple_dot_products', prods=str(products)), 'state': None})
                steps.append({'step': 4, 'desc': Language.tr('step_triple_dot_sum', sum=dot), 'state': None})
                steps.append({'step': 5, 'desc': Language.tr('step_triple_result', result=dot), 'state': None})
                return dot, steps
            return dot, None
        else:
            cross = np.cross(v2, v3)
            dot = float(np.dot(v1, cross))
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_triple_init', v1=self._format_vector(v1), v2=self._format_vector(v2), v3=self._format_vector(v3)), 'state': None}]
                x = v2[1]*v3[2] - v2[2]*v3[1]
                y = v2[2]*v3[0] - v2[0]*v3[2]
                z = v2[0]*v3[1] - v2[1]*v3[0]
                steps.append({'step': 1, 'desc': Language.tr('step_triple_cross_components_calc', x_expr=f"{v2[1]:.4g}·{v3[2]:.4g} - {v2[2]:.4g}·{v3[1]:.4g} = {x:.4f}", y_expr=f"{v2[2]:.4g}·{v3[0]:.4g} - {v2[0]:.4g}·{v3[2]:.4g} = {y:.4f}", z_expr=f"{v2[0]:.4g}·{v3[1]:.4g} - {v2[1]:.4g}·{v3[0]:.4g} = {z:.4f}"), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_triple_cross', cross=self._format_vector(cross)), 'state': cross})
                prod_expr = " + ".join([f"({v1[i]:.4g}·{cross[i]:.4g})" for i in range(3)])
                steps.append({'step': 3, 'desc': Language.tr('step_triple_dot_products_detail', detail=prod_expr, prods=self._format_vector(v1 * cross)), 'state': v1 * cross})
                steps.append({'step': 4, 'desc': Language.tr('step_triple_dot_sum', sum=dot), 'state': None})
                steps.append({'step': 5, 'desc': Language.tr('step_triple_result_volume', result=dot, volume=abs(dot)), 'state': None})
                return dot, steps
            return dot, None

    def are_collinear(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        steps = []
        if len(v1) != len(v2):
            if show_steps:
                steps.append({'step': 0, 'desc': Language.tr('step_collinear_false_dim'), 'state': None})
                return False, steps
            return False
        if self._symbolic_mode:
            if len(v1) == 2:
                det = v1[0]*v2[1] - v1[1]*v2[0]
                res = det == 0
                if show_steps:
                    steps.append({'step': 0, 'desc': Language.tr('step_collinear_2d_det', det=det), 'state': None})
                    steps.append({'step': 1, 'desc': Language.tr('step_collinear_result', result=res), 'state': None})
                    return res, steps
                return res
            else:
                cross = v1.cross(v2)
                res = cross == sp.Matrix([0,0,0])
                if show_steps:
                    steps.append({'step': 0, 'desc': Language.tr('step_collinear_3d_cross', cross=self._format_vector(cross)), 'state': cross})
                    steps.append({'step': 1, 'desc': Language.tr('step_collinear_result', result=res), 'state': None})
                    return res, steps
                return res
        else:
            if len(v1) == 2:
                det = v1[0]*v2[1] - v1[1]*v2[0]
                res = np.isclose(det, 0)
                if show_steps:
                    steps.append({'step': 0, 'desc': Language.tr('step_collinear_2d_det_numeric', det=det), 'state': None})
                    steps.append({'step': 1, 'desc': Language.tr('step_collinear_result', result=res), 'state': None})
                    return res, steps
                return res
            else:
                cross = np.cross(v1, v2)
                res = np.allclose(cross, 0)
                if show_steps:
                    steps.append({'step': 0, 'desc': Language.tr('step_collinear_3d_cross_numeric', cross=self._format_vector(cross)), 'state': cross})
                    steps.append({'step': 1, 'desc': Language.tr('step_collinear_result', result=res), 'state': None})
                    return res, steps
                return res

    def find_collinearity_parameter(self, v1_expr, v2_expr, param_name='λ', show_steps=False):
        v1 = [self._parse_expression(comp) for comp in v1_expr]
        v2 = [self._parse_expression(comp) for comp in v2_expr]
        param = sp.Symbol(param_name)
        steps = []
        if show_steps:
            steps.append({'step': 0, 'desc': Language.tr('step_collinear_init', v1=str(v1), v2=str(v2)), 'state': None})
        if len(v1) == 2:
            eq = v1[0]*v2[1] - v1[1]*v2[0]
            if show_steps:
                steps.append({'step': 1, 'desc': Language.tr('step_collinear_2d_eq', eq=f"{eq} = 0"), 'state': None})
        else:
            cross = sp.Matrix(v1).cross(sp.Matrix(v2))
            non_zero = [comp for comp in cross if comp != 0]
            if non_zero:
                eq = non_zero[0]
            else:
                eq = sp.S.Zero
            if show_steps:
                steps.append({'step': 1, 'desc': Language.tr('step_collinear_3d_cross', cross=str(cross)), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_collinear_3d_eq', eq=f"{eq} = 0"), 'state': None})
        solutions = sp.solve(eq, param)
        if show_steps:
            steps.append({'step': len(steps), 'desc': Language.tr('step_collinear_solve', param=param, solutions=str(solutions)), 'state': None})
        return solutions, steps

    def are_coplanar(self, v1, v2, v3, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        v3 = self._as_vector(v3)
        if not (len(v1) == len(v2) == len(v3) == 3):
            raise ValueError(Language.tr('err_coplanar_3d'))
        if self._symbolic_mode:
            cross = v2.cross(v3)
            triple = v1.dot(cross)
            res = triple == 0
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_coplanar_cross', cross=self._format_vector(cross)), 'state': cross}]
                steps.append({'step': 1, 'desc': Language.tr('step_coplanar_triple', triple=triple), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_coplanar_result', result=res), 'state': None})
                return res, steps
            return res
        else:
            cross = np.cross(v2, v3)
            triple = np.dot(v1, cross)
            res = np.isclose(triple, 0)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_coplanar_cross_numeric', cross=self._format_vector(cross)), 'state': cross}]
                steps.append({'step': 1, 'desc': Language.tr('step_coplanar_triple_numeric', triple=triple), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_coplanar_result', result=res), 'state': None})
                return res, steps
            return res

    def is_orthogonal(self, v1, v2, show_steps=False):
        v1 = self._as_vector(v1)
        v2 = self._as_vector(v2)
        self._validate_vectors_same_length(v1, v2, "orthogonality")
        if self._symbolic_mode:
            dot = v1.dot(v2)
            res = dot == 0
        else:
            dot = np.dot(v1, v2)
            res = np.isclose(dot, 0)
        if show_steps:
            steps = [{'step': 0, 'desc': Language.tr('step_orthogonal_dot', dot=dot), 'state': None}]
            steps.append({'step': 1, 'desc': Language.tr('step_orthogonal_result', result=res), 'state': None})
            return res, steps
        return res

    def is_basis(self, vectors, show_steps=False):
        if self._symbolic_mode:
            M = sp.Matrix.hstack(*[self._as_vector(v) for v in vectors])
            det = M.det()
            res = det != 0
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_basis_det', det=det), 'state': M}]
                steps.append({'step': 1, 'desc': Language.tr('step_basis_result', result=res), 'state': None})
                return res, steps
            return res
        else:
            vecs = [self._as_vector(v) for v in vectors]
            if len(vecs) != len(vecs[0]):
                if show_steps:
                    steps = [{'step': 0, 'desc': Language.tr('err_basis_count', got=len(vecs), expected=len(vecs[0])), 'state': None}]
                    return False, steps
                return False
            M = np.column_stack(vecs)
            det = np.linalg.det(M)
            res = not np.isclose(det, 0)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_basis_det_numeric', det=det), 'state': M}]
                steps.append({'step': 1, 'desc': Language.tr('step_basis_result', result=res), 'state': None})
                return res, steps
            return res

    def decompose_vector(self, v, basis, show_steps=False):
        v = self._as_vector(v)
        basis = [self._as_vector(b) for b in basis]
        dim = len(v)
        if any(len(b) != dim for b in basis):
            raise ValueError(Language.tr('err_basis_dimension'))
        if len(basis) != dim:
            raise ValueError(Language.tr('err_basis_count', got=len(basis), expected=dim))
        if self._symbolic_mode:
            M = sp.Matrix.hstack(*basis)
            v_vec = sp.Matrix(v)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_decompose_init', v=self._format_vector(v_vec), basis=str([self._format_vector(b) for b in basis])), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_decompose_matrix', matrix=str(M)), 'state': M})
                system_str = "\n".join([f"{M.row(i)} · x = {v[i]}" for i in range(dim)])
                steps.append({'step': 2, 'desc': Language.tr('step_decompose_system', system=system_str), 'state': None})
                try:
                    coeffs = M.LUsolve(v_vec)
                except Exception:
                    x_sym = sp.symbols(f'x0:{dim}')
                    eqs = [sum(M[i,j]*x_sym[j] for j in range(dim)) - v[i] for i in range(dim)]
                    sol = sp.linsolve(eqs, x_sym)
                    coeffs = sp.Matrix(list(sol)[0])
                steps.append({'step': 3, 'desc': Language.tr('step_decompose_result', coeffs=self._format_vector(coeffs)), 'state': coeffs})
                return list(coeffs), steps
            else:
                return list(M.LUsolve(v_vec)), None
        else:
            M = np.column_stack(basis)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_decompose_init', v=self._format_vector(v), basis=str([self._format_vector(b) for b in basis])), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_decompose_matrix_numeric', matrix=str(M)), 'state': M})
                system_str = "\n".join([f"{M[i]} · x = {v[i]:.4f}" for i in range(dim)])
                steps.append({'step': 2, 'desc': Language.tr('step_decompose_system', system=system_str), 'state': None})
            coeffs = np.linalg.solve(M, v)
            if show_steps:
                steps.append({'step': 3, 'desc': Language.tr('step_decompose_result_numeric', coeffs=self._format_vector(coeffs)), 'state': coeffs})
                return coeffs, steps
            return coeffs, None

    def change_of_basis_matrix(self, old_basis, new_basis, show_steps=False):
        old = [self._as_vector(b) for b in old_basis]
        new = [self._as_vector(b) for b in new_basis]
        if self._symbolic_mode:
            M_old = sp.Matrix.hstack(*old)
            M_new = sp.Matrix.hstack(*new)
            P = M_new.inv() * M_old
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_transition_init'), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_transition_formula'), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_transition_matrix', matrix=str(P)), 'state': P})
                return P, steps
            return P, None
        else:
            M_old = np.column_stack(old)
            M_new = np.column_stack(new)
            P = np.linalg.solve(M_new, M_old)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_transition_init'), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_transition_formula'), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_transition_matrix', matrix=str(P)), 'state': P})
                return P, steps
            return P, None

    def gram_schmidt(self, vectors, normalize=True, show_steps=False):
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
            steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_init', vectors=str([self._format_vector(v) for v in vecs])), 'state': None})
            step_num += 1
        for i, v in enumerate(vecs):
            u = v
            if show_steps:
                steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_start', idx=i+1, v=self._format_vector(v)), 'state': None})
                step_num += 1
            for j, b in enumerate(basis):
                if self._symbolic_mode:
                    proj = (v.dot(b) / b.dot(b)) * b
                else:
                    proj = (np.dot(v, b) / np.dot(b, b)) * b
                u = u - proj
                if show_steps:
                    steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_subtract', j=j+1, proj=self._format_vector(proj), u=self._format_vector(u)), 'state': u})
                    step_num += 1
            if self._symbolic_mode:
                if u.norm() == 0:
                    if show_steps:
                        steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_dependent', idx=i+1), 'state': None})
                        step_num += 1
                    continue
            else:
                if np.linalg.norm(u) < 1e-12:
                    if show_steps:
                        steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_dependent', idx=i+1), 'state': None})
                        step_num += 1
                    continue
            basis.append(u)
            if show_steps:
                steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_orthogonal', idx=i+1, u=self._format_vector(u)), 'state': u})
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
                    steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_normalize', idx=i+1, u=self._format_vector(u), e=self._format_vector(e)), 'state': e})
                    step_num += 1
            basis = normalized
        if show_steps:
            steps.append({'step': step_num, 'desc': Language.tr('step_gram_schmidt_result', basis=str([self._format_vector(b) for b in basis])), 'state': None})
        return basis, steps

    def characteristic_polynomial(self, A, var='λ', show_steps=False):
        if self._symbolic_mode:
            A = self._to_sympy_matrix(A) if not isinstance(A, sp.Matrix) else A
            var_str = str(var)
            λ = sp.Symbol(var_str)
            n = A.rows
            char_poly = (A - λ * sp.eye(n)).det()
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_charpoly_init', var=var_str), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_charpoly_matrix',
                                                             matrix=str(A - λ * sp.eye(n))),
                              'state': A - λ * sp.eye(n)})
                steps.append({'step': 2, 'desc': Language.tr('step_charpoly_det',
                                                             poly=str(char_poly)), 'state': None})
                return char_poly, steps
            return char_poly, None
        else:
            A = np.array(A, dtype=self._current_dtype)
            n = A.shape[0]
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_charpoly_init', var=var), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_charpoly_matrix_numeric',
                                                             matrix=str(A)), 'state': A})
                coeffs = np.poly(A)
                steps.append({'step': 2, 'desc': Language.tr('step_charpoly_coeffs',
                                                             coeffs=str(coeffs)), 'state': None})
                return coeffs, steps
            return np.poly(A), None

    def eigenvalues(self, A, show_steps=False):
        if self._symbolic_mode:
            A = sp.Matrix(A)
            eigen = A.eigenvals()
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_eigenvals_init'), 'state': None}]
                char_poly = self.characteristic_polynomial(A, show_steps=False)[0]
                steps.append({'step': 1, 'desc': Language.tr('step_eigenvals_charpoly', poly=str(char_poly)), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_eigenvals_roots', vals=str(eigen)), 'state': None})
                return eigen, steps
            return eigen, None
        else:
            A = np.array(A, dtype=self._current_dtype)
            vals = np.linalg.eigvals(A)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_eigenvals_init'), 'state': None}]
                coeffs = np.poly(A)
                steps.append({'step': 1, 'desc': Language.tr('step_eigenvals_charpoly_numeric', coeffs=str(coeffs)), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_eigenvals_result', vals=str(vals)), 'state': None})
                return vals, steps
            return vals, None

    def eigenvectors(self, A, show_steps=False):
        if self._symbolic_mode:
            A = self._to_sympy_matrix(A) if not isinstance(A, sp.Matrix) else A
            eigenvects = A.eigenvects()
            cleaned = []
            for val, mult, vecs in eigenvects:
                clean_vecs = []
                for v in vecs:
                    if v.free_symbols:
                        clean_vecs.append(v)
                    else:
                        try:
                            denoms = [sp.fraction(x)[1] for x in v if x != 0]
                            lcm_denom = sp.ilcm(*denoms) if denoms else 1
                            scaled = v * lcm_denom
                            nums = [sp.fraction(x)[0] for x in scaled if x != 0]
                            gcd_num = sp.igcd(*nums) if nums else 1
                            clean_v = scaled / gcd_num
                            clean_vecs.append(clean_v)
                        except (ValueError, TypeError):
                            clean_vecs.append(v)
                cleaned.append((val, mult, clean_vecs))
            eigenvects = cleaned
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_eigenvecs_init'), 'state': None}]
                char_poly = self.characteristic_polynomial(A, show_steps=False)[0]
                steps.append({'step': 1, 'desc': Language.tr('step_eigenvecs_charpoly',
                                                             poly=str(char_poly)), 'state': None})
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
            # numeric branch unchanged
            A = np.array(A, dtype=self._current_dtype)
            vals, vecs = np.linalg.eig(A)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_eigenvecs_init'), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_eigenvecs_vals',
                                                             vals=str(vals)), 'state': None})
                steps.append({'step': 2, 'desc': Language.tr('step_eigenvecs_matrix',
                                                             matrix=str(vecs)), 'state': vecs})
                return (vals, vecs), steps
            return (vals, vecs), None

    def diagonalize(self, A, show_steps=False):
        if self._symbolic_mode:
            A = self._to_sympy_matrix(A) if not isinstance(A, sp.Matrix) else A
            n = A.rows
            if n != A.cols:
                raise ValueError(Language.tr('err_square_matrix', operation='Diagonalization', shape=A.shape))

            eigen_info = A.eigenvects()
            for val, mult, vecs in eigen_info:
                if len(vecs) < mult:
                    raise ValueError(Language.tr('err_not_diagonalizable'))

            P_cols = []
            diag_entries = []
            for val, mult, vecs in eigen_info:
                for v in vecs:
                    P_cols.append(v)
                    diag_entries.append(val)
            P = sp.Matrix.hstack(*P_cols) if P_cols else sp.eye(n)
            D = sp.diag(*diag_entries)
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_diag_init'), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_diag_eigenvals',
                                                             vals=str(diag_entries)), 'state': D})
                steps.append({'step': 2, 'desc': Language.tr('step_diag_eigenvecs',
                                                             matrix=str(P)), 'state': P})
                steps.append({'step': 3, 'desc': Language.tr('step_diag_verify',
                                                             product=str(P * D * P.inv())), 'state': None})
                return (P, D), steps
            return (P, D), None
        else:
            # numeric branch unchanged
            A = np.array(A, dtype=self._current_dtype)
            n = A.shape[0]
            vals, vecs = np.linalg.eig(A)
            if np.linalg.matrix_rank(vecs) < n:
                raise ValueError(Language.tr('err_not_diagonalizable'))
            D = np.diag(vals)
            P = vecs
            if show_steps:
                steps = [{'step': 0, 'desc': Language.tr('step_diag_init'), 'state': None}]
                steps.append({'step': 1, 'desc': Language.tr('step_diag_eigenvals_numeric',
                                                             vals=str(vals)), 'state': D})
                steps.append({'step': 2, 'desc': Language.tr('step_diag_eigenvecs_numeric',
                                                             matrix=str(P)), 'state': P})
                steps.append({'step': 3, 'desc': Language.tr('step_diag_verify_numeric',
                                                             product=str(P @ D @ np.linalg.inv(P))), 'state': None})
                return (P, D), steps
            return (P, D), None

    def vector_from_points(self, A, B):
        A = self._parse_point(A)
        B = self._parse_point(B)
        return B - A

    def points_collinear(self, A, B, C, show_steps=False):
        AB = self.vector_from_points(A, B)
        AC = self.vector_from_points(A, C)
        if show_steps:
            res, steps = self.are_collinear(AB, AC, show_steps=True)
            steps.insert(0, {'step': 0, 'desc': Language.tr('step_points_collinear_init', A=A, B=B, C=C), 'state': None})
            for i, s in enumerate(steps):
                s['step'] = i
            return res, steps
        return self.are_collinear(AB, AC)

    def points_coplanar(self, A, B, C, D, show_steps=False):
        AB = self.vector_from_points(A, B)
        AC = self.vector_from_points(A, C)
        AD = self.vector_from_points(A, D)
        if show_steps:
            res, steps = self.are_coplanar(AB, AC, AD, show_steps=True)
            steps.insert(0, {'step': 0, 'desc': Language.tr('step_points_coplanar_init', A=A, B=B, C=C, D=D), 'state': None})
            for i, s in enumerate(steps):
                s['step'] = i
            return res, steps
        return self.are_coplanar(AB, AC, AD)

    def triangle_area_points(self, A, B, C, show_steps=False):
        A = self._parse_point(A)
        B = self._parse_point(B)
        C = self._parse_point(C)
        if len(A) == 2:
            A = np.append(A, 0)
            B = np.append(B, 0)
            C = np.append(C, 0)
        AB = B - A
        AC = C - A
        if self._symbolic_mode:
            cross = sp.Matrix(AB).cross(sp.Matrix(AC))
            area = sp.sqrt(cross.dot(cross)) / 2
        else:
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross)
        if show_steps:
            steps = [{'step': 0, 'desc': Language.tr('step_triangle_area_init', A=self._format_point(A), B=self._format_point(B), C=self._format_point(C)), 'state': None}]
            steps.append({'step': 1, 'desc': Language.tr('step_triangle_area_vectors', AB=self._format_vector(AB), AC=self._format_vector(AC)), 'state': None})
            steps.append({'step': 2, 'desc': Language.tr('step_triangle_area_cross', cross=self._format_vector(cross)), 'state': cross})
            steps.append({'step': 3, 'desc': Language.tr('step_triangle_area_result', area=area), 'state': None})
            return area, steps
        return area

    def tetrahedron_volume_points(self, A, B, C, D, show_steps=False):
        A = self._parse_point(A)
        B = self._parse_point(B)
        C = self._parse_point(C)
        D = self._parse_point(D)
        AB = B - A
        AC = C - A
        AD = D - A
        if self._symbolic_mode:
            cross = sp.Matrix(AC).cross(sp.Matrix(AD))
            triple = sp.Matrix(AB).dot(cross)
            volume = sp.Abs(triple) / 6
        else:
            cross = np.cross(AC, AD)
            triple = np.dot(AB, cross)
            volume = abs(triple) / 6.0
        if show_steps:
            steps = [{'step': 0, 'desc': Language.tr('step_tetrahedron_volume_init', A=self._format_point(A), B=self._format_point(B), C=self._format_point(C), D=self._format_point(D)), 'state': None}]
            steps.append({'step': 1, 'desc': Language.tr('step_tetrahedron_volume_vectors', AB=self._format_vector(AB), AC=self._format_vector(AC), AD=self._format_vector(AD)), 'state': None})
            steps.append({'step': 2, 'desc': Language.tr('step_tetrahedron_volume_cross', cross=self._format_vector(cross)), 'state': cross})
            steps.append({'step': 3, 'desc': Language.tr('step_tetrahedron_volume_triple', triple=triple), 'state': None})
            steps.append({'step': 4, 'desc': Language.tr('step_tetrahedron_volume_result', volume=volume), 'state': None})
            return volume, steps
        return volume

    def _parse_point(self, point):
        if self._symbolic_mode:
            return sp.Matrix([self._parse_expression(c) for c in point])
        else:
            arr = np.array([float(c) for c in point], dtype=self._current_dtype)
            return arr

    def _format_point(self, p):
        if self._symbolic_mode:
            return str(p)
        return "(" + ", ".join(f"{x:.4g}" for x in p) + ")"

    def _norm(self, v):
        if self._symbolic_mode:
            return sp.sqrt(v.dot(v))
        return np.linalg.norm(v)

    def _cross(self, a, b):
        if self._symbolic_mode:
            return a.cross(b)
        return np.cross(a, b)

    def _dot(self, a, b):
        if self._symbolic_mode:
            return a.dot(b)
        return np.dot(a, b)

    def create_point(self, coords, name=None):
        return Point(coords, name=name, engine=self)

    def midpoint(self, A, B):
        return Point((A.coords + B.coords) / 2, name=f"Mid({A.name},{B.name})", engine=self)

    def centroid(self, points):
        sum_coords = points[0].coords
        for p in points[1:]:
            sum_coords += p.coords
        return Point(sum_coords / len(points), name="Centroid", engine=self)

    def stats(self):
        return {
            "device": self.device,
            "precision": self.precision.value,
            "symbolic_mode": self._symbolic_mode,
            "cache_hits": 0,
            "cache_misses": 0
        }
