"""
Translation dictionary and language management.
"""

TRANSLATIONS = {
    'ru': {
        # Application
        'app_title': 'Матричный калькулятор',
        'file_menu': 'Файл',
        'help_menu': 'Справка',
        'exit': 'Выход',
        'about': 'О программе',

        # File menu items
        'load_a': 'Загрузить A...',
        'load_b': 'Загрузить B...',
        'save_a': 'Сохранить A...',
        'save_b': 'Сохранить B...',
        'save_result': 'Сохранить результат',

        # Toolbar
        'operation': 'Операция:',
        'scalar': 'Скаляр:',
        'compute': 'Вычислить',
        'cancel': 'Отмена',
        'swap': '⇄ Поменять A↔B',
        'precision': 'Точность:',
        'language': 'Язык:',

        # Matrix widget controls
        'rows': 'Строк:',
        'cols': 'Столбцов:',
        'resize': 'Изменить размер',

        # Status bar
        'ready': 'Готов',
        'computing': 'Вычисление...',
        'cancel_request': 'Отмена...',
        'swap_success': 'Матрицы обменяны',
        'precision_set': 'Точность: {}',

        # Matrix titles
        'matrix_a': 'Матрица A',
        'matrix_b': 'Матрица B',
        'matrix_a_coeff': 'A (коэффициенты)',
        'matrix_b_rhs': 'B (свободные члены)',

        # Step viewer header
        'step_solution': 'Пошаговое решение / Результат',
        'result': 'Результат:',
        'step_prefix': 'Шаг {number}:',
        'state': 'Состояние',

        # Operation names (internal IDs)
        'op_add': 'Сложение',
        'op_sub': 'Вычитание',
        'op_mul': 'Умножение',
        'op_scalar_mul': 'Умножение на число',
        'op_transpose': 'Транспонирование A',
        'op_det': 'Определитель A',
        'op_rank': 'Ранг A',
        'op_inv': 'Обращение A',
        'op_solve': 'Решение СЛАУ (A*X = B)',

        # Messages and errors
        'error': 'Ошибка',
        'warning': 'Предупреждение',
        'info': 'Информация',
        'save_error': 'Ошибка сохранения',
        'load_error': 'Ошибка загрузки',
        'file_empty': 'Файл пуст',
        'matrix_empty': 'Матрица пуста',
        'invalid_number': 'Неверное число',
        'operation_unknown': 'Неизвестная операция',
        'about_text': 'Матричный калькулятор\nИспользует NumPy{}\\nТекущая точность: {}',
        'gpu_available': ' и CuPy',
        'gpu_not_available': '',

        # Engine error messages
        'err_square_matrix': '{operation} требует квадратную матрицу. Получена форма {shape}',
        'err_same_shape': '{operation} требует матрицы одинаковой формы. Получены {shape1} и {shape2}',
        'err_incompatible_mul': 'Несовместимые размеры для умножения: {shape1} и {shape2}',
        'err_singular_matrix': 'Матрица вырождена (невозможно обратить)',
        'err_inversion_failed': 'Обращение не удалось: {msg}',
        'err_system_rows': 'Матрица A и правая часть B должны иметь одинаковое количество строк',
        'err_no_unique_solution': 'Система не имеет единственного решения: {msg}',

        # Step description keys (already in engine)
        'step_initial': 'Начальная матрица',
        'step_initial_aug': 'Начальная расширенная матрица [A | B]',
        'step_swap': 'Перестановка строк {row1} и {row2}',
        'step_normalize': 'Нормализация строки {row} (деление на {value:.2f})',
        'step_eliminate': 'Исключение столбца {col} из строки {row} (коэффициент = {factor:.4f})',
        'step_pivot_zero': 'Столбец {col}: ведущий элемент равен нулю → матрица вырождена',
        'step_singular': 'Матрица вырождена, определитель = 0',
        'step_back_subst': 'Обратная подстановка: x[{i}] = {value}',
        'step_diag_product': 'Произведение диагональных элементов = {product:.6f}, знак = {sign}',
        'step_det_result': 'Определитель = {det:.6f}',
        'step_rank_final': 'Ранг матрицы = {rank}',
    },
    'en': {
        # Application
        'app_title': 'Matrix Calculator',
        'file_menu': 'File',
        'help_menu': 'Help',
        'exit': 'Exit',
        'about': 'About',

        # File menu items
        'load_a': 'Load A...',
        'load_b': 'Load B...',
        'save_a': 'Save A...',
        'save_b': 'Save B...',
        'save_result': 'Save Result',

        # Toolbar
        'operation': 'Operation:',
        'scalar': 'Scalar:',
        'compute': 'Compute',
        'cancel': 'Cancel',
        'swap': '⇄ Swap A↔B',
        'precision': 'Precision:',
        'language': 'Language:',

        # Matrix widget controls
        'rows': 'Rows:',
        'cols': 'Cols:',
        'resize': 'Resize',

        # Status bar
        'ready': 'Ready',
        'computing': 'Computing...',
        'cancel_request': 'Canceling...',
        'swap_success': 'Matrices swapped',
        'precision_set': 'Precision: {}',

        # Matrix titles
        'matrix_a': 'Matrix A',
        'matrix_b': 'Matrix B',
        'matrix_a_coeff': 'A (coefficients)',
        'matrix_b_rhs': 'B (right-hand side)',

        # Step viewer header
        'step_solution': 'Step-by-step solution / Result',
        'result': 'Result:',
        'step_prefix': 'Step {number}:',
        'state': 'State',

        # Operation names (internal IDs)
        'op_add': 'Addition',
        'op_sub': 'Subtraction',
        'op_mul': 'Multiplication',
        'op_scalar_mul': 'Scalar Multiply',
        'op_transpose': 'Transpose A',
        'op_det': 'Determinant A',
        'op_rank': 'Rank A',
        'op_inv': 'Inverse A',
        'op_solve': 'Solve SLAE (A*X = B)',

        # Messages and errors
        'error': 'Error',
        'warning': 'Warning',
        'info': 'Information',
        'save_error': 'Save error',
        'load_error': 'Load error',
        'file_empty': 'File is empty',
        'matrix_empty': 'Matrix is empty',
        'invalid_number': 'Invalid number',
        'operation_unknown': 'Unknown operation',
        'about_text': 'Matrix Calculator\nUses NumPy{}\\nCurrent precision: {}',
        'gpu_available': ' and CuPy',
        'gpu_not_available': '',

        # Engine error messages
        'err_square_matrix': '{operation} requires a square matrix. Got shape {shape}',
        'err_same_shape': '{operation} requires matrices of the same shape. Got {shape1} and {shape2}',
        'err_incompatible_mul': 'Incompatible shapes for multiplication: {shape1} and {shape2}',
        'err_singular_matrix': 'Matrix is singular (cannot be inverted)',
        'err_inversion_failed': 'Inversion failed: {msg}',
        'err_system_rows': 'Matrix A and RHS B must have the same number of rows',
        'err_no_unique_solution': 'System has no unique solution: {msg}',

        # Step description keys
        'step_initial': 'Initial matrix',
        'step_initial_aug': 'Initial augmented matrix [A | B]',
        'step_swap': 'Swapped row {row1} and row {row2}',
        'step_normalize': 'Normalized row {row} by dividing by {value:.2f}',
        'step_eliminate': 'Eliminated column {col} from row {row} (factor = {factor:.4f})',
        'step_pivot_zero': 'Column {col}: pivot is zero → matrix is singular',
        'step_singular': 'Matrix is singular, determinant = 0',
        'step_back_subst': 'Back substitution: x[{i}] = {value}',
        'step_diag_product': 'Product of diagonal elements = {product:.6f}, sign = {sign}',
        'step_det_result': 'Determinant = {det:.6f}',
        'step_rank_final': 'Matrix rank = {rank}',
    }
}

class Language:
    """Simple language manager."""
    _current = 'ru'  # default

    @classmethod
    def get(cls):
        return cls._current

    @classmethod
    def set(cls, lang):
        if lang in TRANSLATIONS:
            cls._current = lang

    @classmethod
    def tr(cls, key, **kwargs):
        """Translate a key, optionally formatting with kwargs."""
        text = TRANSLATIONS[cls._current].get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text
