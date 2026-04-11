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

        # Vector widget controls
        'size': 'Размер:',

        # Notebook tabs
        'matrix_tab': 'Матрицы',
        'vector_tab': 'Векторы',

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

        # Vector titles
        'vector_a': 'Вектор A',
        'vector_b': 'Вектор B',

        # Step viewer header
        'step_solution': 'Пошаговое решение / Результат',
        'result': 'Результат:',
        'step_prefix': 'Шаг {number}:',
        'state': 'Состояние',

        # Operation names (matrix)
        'op_add': 'Сложение',
        'op_sub': 'Вычитание',
        'op_mul': 'Умножение',
        'op_scalar_mul': 'Умножение на число',
        'op_transpose': 'Транспонирование A',
        'op_det': 'Определитель A',
        'op_rank': 'Ранг A',
        'op_inv': 'Обращение A',
        'op_solve': 'Решение СЛАУ (A*X = B)',

        # Vector operation button labels
        'btn_add': 'A + B',
        'btn_subtract': 'A - B',
        'btn_dot': 'A · B',
        'btn_cross': 'A × B',
        'btn_norm_a': '||A||',
        'btn_norm_b': '||B||',
        'btn_normalize_a': 'Â',
        'btn_normalize_b': 'B̂',
        'btn_projection': 'proj_B A',
        'btn_angle': '∠(A,B)',
        'btn_triple': 'A·(B×C)',
        'btn_scalar_mul_a': 'k·A',
        'btn_scalar_mul_b': 'k·B',
        'show_steps': 'Показывать шаги',

        # Vector dialogs
        'angle_unit': 'Единицы измерения угла',
        'use_degrees': 'Использовать градусы? (Нет = радианы)',
        'triple_input': 'Смешанное произведение',
        'enter_vector_c': 'Введите вектор C (числа через пробел):',
        'scalar_input': 'Умножение на скаляр',
        'enter_scalar': 'Введите скалярное значение:',
        'operations': 'Операции с векторами',

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

        # Engine error messages (matrix)
        'err_square_matrix': '{operation} требует квадратную матрицу. Получена форма {shape}',
        'err_same_shape': '{operation} требует матрицы одинаковой формы. Получены {shape1} и {shape2}',
        'err_incompatible_mul': 'Несовместимые размеры для умножения: {shape1} и {shape2}',
        'err_singular_matrix': 'Матрица вырождена (невозможно обратить)',
        'err_inversion_failed': 'Обращение не удалось: {msg}',
        'err_system_rows': 'Матрица A и правая часть B должны иметь одинаковое количество строк',
        'err_no_unique_solution': 'Система не имеет единственного решения: {msg}',

        # Engine error messages (vector)
        'err_vector_1d': 'Вектор должен быть одномерным (получена форма {shape})',
        'err_vector_same_len': 'Векторы должны иметь одинаковую длину для {op} (получено {len1} и {len2})',
        'err_cross_3d': 'Векторное произведение требует трёхмерные векторы (получено {len1} и {len2})',
        'err_normalize_zero': 'Невозможно нормализовать нулевой вектор',
        'err_projection_zero_vec': 'Невозможно спроецировать на нулевой вектор',
        'err_angle_zero_vec': 'Угол не определён для нулевого вектора',
        'err_triple_3d': 'Смешанное произведение требует три трёхмерных вектора',

        # Matrix step descriptions
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

        # Vector step descriptions
        'step_vector_add_init': 'Сложение векторов: {v1} + {v2}',
        'step_vector_add_result': 'Результат: {res}',
        'step_vector_sub_init': 'Вычитание векторов: {v1} - {v2}',
        'step_vector_sub_result': 'Результат: {res}',
        'step_vector_scale_init': 'Умножение вектора {v} на {scalar}',
        'step_vector_scale_result': 'Результат: {res}',
        'step_dot_init': 'Скалярное произведение {v1} и {v2}',
        'step_dot_products': 'Поэлементные произведения: {prods}',
        'step_dot_sum': 'Сумма произведений = {sum}',
        'step_cross_init': 'Векторное произведение {v1} × {v2}',
        'step_cross_components': 'Компоненты: x={x}, y={y}, z={z}',
        'step_cross_result': 'Результирующий вектор: {res}',
        'step_norm_init': 'Вычисление нормы вектора {v}',
        'step_norm_squares': 'Квадраты элементов: {squares}',
        'step_norm_sum_sq': 'Сумма квадратов = {sum_sq}',
        'step_norm_result': 'Норма = {norm}',
        'step_normalize_init': 'Нормализация вектора {v}',
        'step_normalize_norm': 'Норма = {norm}',
        'step_normalize_result': 'Единичный вектор: {unit}',
        'step_proj_init': 'Проекция {v1} на {v2}',
        'step_proj_dot': 'Скалярное произведение (v1·v2) = {dot}',
        'step_proj_norm_sq': '||v2||² = {norm_sq}',
        'step_proj_scalar': 'Скалярный коэффициент = {scalar}',
        'step_proj_result': 'Вектор проекции: {proj}',
        'step_angle_init': 'Угол между {v1} и {v2}',
        'step_angle_dot': 'Скалярное произведение = {dot}',
        'step_angle_norms': '||v1|| = {norm1}, ||v2|| = {norm2}',
        'step_angle_cos': 'cos(θ) = {cos}',
        'step_angle_rad': 'θ = {rad} рад',
        'step_angle_deg': 'θ = {deg}°',
        'step_triple_init': 'Смешанное произведение: {v1} · ({v2} × {v3})',
        'step_triple_cross': 'Векторное произведение ({v2} × {v3}) = {cross}',
        'step_triple_dot': 'Скалярное произведение = {dot}',

        'step_dot_products_detail': 'Поэлементные произведения: {detail} = {prods}',
        'step_cross_determinant': 'Определитель: i·({i}) - j·({j}) + k·({k})',
        'step_cross_determinant_numeric': 'Определитель: |i  j  k|\n|{a1} {a2} {a3}|\n|{b1} {b2} {b3}|',
        'step_cross_components_calc': 'Компоненты:\n  x = {x_expr}\n  y = {y_expr}\n  z = {z_expr}',

        'step_triple_cross_start': '1. Вычисляем векторное произведение {v2} × {v3}:',
        'step_triple_cross_start_numeric': '1. Векторное произведение B × C:',
        'step_triple_cross_components_calc': '   x = {x_expr}\n   y = {y_expr}\n   z = {z_expr}',
        'step_triple_dot_start': '2. Скалярное произведение {v1} · ({v2} × {v3}):',
        'step_triple_dot_start_numeric': '2. Скалярное произведение A · (B × C):',
        'step_triple_dot_products_detail': '   Поэлементно: {detail} = {prods}',
        'step_triple_dot_products': '   Произведения: {prods}',
        'step_triple_dot_sum': '   Сумма = {sum:.6f}',
        'step_triple_result': 'Результат (смешанное произведение) = {result}',
        'step_triple_result_volume': 'Результат: A·(B×C) = {result:.6f}\nОбъём параллелепипеда = |{result:.6f}| = {'
                                     'volume:.6f}',

        'symbolic_mode': 'Символьный режим',

        # Special relations
        'input_vectors': 'Входные векторы',
        'comma_separated': '(через запятую/пробел)',
        'parameter_name': 'Параметр:',
        'btn_collinear_check': 'Коллинеарны?',
        'btn_collinear_param': 'Найти λ для коллинеарности',
        'btn_orthogonal': 'Ортогональны?',
        'btn_coplanar': 'Компланарны?',
        'collinearity_check': 'Проверка коллинеарности',
        'collinearity_param': 'Параметр коллинеарности',
        'orthogonality_check': 'Проверка ортогональности',
        'coplanarity_check': 'Проверка компланарности',

        # Step descriptions for collinearity
        'step_collinear_init': 'Векторы: a = {v1}, b = {v2}',
        'step_collinear_2d_eq': 'Условие коллинеарности (2D): {eq}',
        'step_collinear_3d_cross': 'Векторное произведение: a × b = {cross}',
        'step_collinear_3d_eq': 'Приравниваем компоненту к нулю: {eq}',
        'step_collinear_solve': 'Решаем относительно {param}: {solutions}',

        # Basis and decomposition
        'vector_to_decompose': 'Вектор для разложения',
        'basis_vectors': 'Базисные векторы',
        'btn_check_basis': 'Проверить базис',
        'btn_decompose': 'Разложить',
        'basis_check': 'Проверка базиса',
        'decomposition_result': 'Координаты в базисе',
        'coordinates': 'Координаты',

        'err_basis_dimension': 'Все векторы должны быть одной размерности',
        'err_basis_count': 'Требуется {expected} векторов, получено {got}',

        'step_decompose_init': 'Разложение вектора {v} по базису {basis}',
        'step_decompose_matrix': 'Матрица перехода (базис по столбцам):\n{matrix}',
        'step_decompose_solve': 'Решаем систему {system}',
        'step_decompose_result': 'Коэффициенты: {coeffs}',
        'step_decompose_matrix_numeric': 'Матрица перехода:\n{matrix}',
        'step_decompose_solve_numeric': 'Решаем линейную систему',
        'step_decompose_result_numeric': 'Координаты: {coeffs}',

        # Transition matrix
        'step_transition_init': 'Построение матрицы перехода',
        'step_transition_formula': 'P = (новый базис)⁻¹ · (старый базис)',
        'step_transition_matrix': 'Матрица перехода:\n{matrix}',

        # Geometry
        'points_coordinates': 'Координаты точек',
        'btn_collinear_points': 'Точки на одной прямой?',
        'btn_coplanar_points': 'Точки в одной плоскости?',
        'btn_triangle_area': 'Площадь треугольника (A,B,C)',
        'btn_tetrahedron_volume': 'Объём тетраэдра (A,B,C,D)',
        'points_collinear': 'Коллинеарность точек',
        'points_coplanar': 'Компланарность точек',
        'triangle_area': 'Площадь треугольника',
        'tetrahedron_volume': 'Объём тетраэдра',
        'err_3d_required': 'Требуются 3D векторы',
        'err_coplanar_3d': 'Компланарность определена только для 3D векторов',

        'special_tab': 'Специальные отношения',
        'basis_tab': 'Базис и разложение',
        'geometry_tab': 'Геометрия',

        # Eigenvalues
        'eigen_tab': 'Собственные значения',
        'input_matrix': 'Матрица',
        'btn_charpoly': 'Характеристический полином',
        'btn_eigenvalues': 'Собственные значения',
        'btn_eigenvectors': 'Собственные векторы',
        'btn_diagonalize': 'Диагонализация',
        'btn_copy_from_a': '← Копировать из A',
        'charpoly_result': 'Характеристический полином',
        'eigenvalues_result': 'Собственные значения',
        'eigenvectors_result': 'Собственные векторы',
        'diagonalization_result': 'Диагонализация',
        'err_not_diagonalizable': 'Матрица не диагонализируема',

        # Step descriptions
        'step_charpoly_init': 'Построение характеристического полинома det(A - {var}I)',
        'step_charpoly_matrix': 'Матрица A - {var}I:\n{matrix}',
        'step_charpoly_det': 'Определитель = {poly}',
        'step_charpoly_matrix_numeric': 'Матрица A:\n{matrix}',
        'step_charpoly_coeffs': 'Коэффициенты характеристического полинома: {coeffs}',

        'step_eigenvals_init': 'Поиск собственных значений',
        'step_eigenvals_charpoly': 'Характеристический полином: {poly}',
        'step_eigenvals_roots': 'Корни (собственные значения): {vals}',
        'step_eigenvals_charpoly_numeric': 'Коэффициенты: {coeffs}',
        'step_eigenvals_result': 'Собственные значения: {vals}',

        'step_eigenvecs_init': 'Поиск собственных векторов',
        'step_eigenvecs_charpoly': 'Характеристический полином: {poly}',
        'step_eigenvecs_for_val': 'Для λ = {val} (кратность {mult}):',
        'step_eigenvecs_vec': '  Вектор {idx}: {vec}',
        'step_eigenvecs_vals': 'Собственные значения: {vals}',
        'step_eigenvecs_matrix': 'Матрица собственных векторов (по столбцам):\n{matrix}',

        'step_diag_init': 'Диагонализация матрицы',
        'step_diag_eigenvals': 'Собственные значения (диагональ D):\n{vals}',
        'step_diag_eigenvecs': 'Матрица P (собственные векторы):\n{matrix}',
        'step_diag_verify': 'Проверка: P·D·P⁻¹ = {product}',
        'step_diag_eigenvals_numeric': 'Собственные значения: {vals}',
        'step_diag_eigenvecs_numeric': 'Матрица P:\n{matrix}',
        'step_diag_verify_numeric': 'Проверка: P·D·P⁻¹ =\n{product}',

        'gram_schmidt_tab': 'Ортогонализация',
        'visualization_tab': 'Визуализация',
        'vectors_to_plot': 'Векторы:',
        'comma_separated_vectors': '(x,y,z; ...)',
        'btn_plot': 'Построить',
        'btn_clear': 'Очистить',
        'vector_plot': 'Векторы в 3D',
        'vectors_format': 'Каждый вектор с новой строки: x y z (или x,y,z)',
        'normalize': 'Нормировать',
        'btn_gram_schmidt': 'Ортогонализовать',
        'gram_schmidt_result': 'Результат ортогонализации',
        'orthogonal_basis': 'Ортогональный базис',
        'export_latex': 'Экспорт в LaTeX...',
        'latex_export_success': 'Файл успешно сохранён',
        'success': 'Успех',

        # Gram-Schmidt steps
        'step_gram_schmidt_init': 'Начальные векторы: {vectors}',
        'step_gram_schmidt_start': 'Обработка вектора {idx}: v = {v}',
        'step_gram_schmidt_subtract': '  Вычитаем проекцию на e{j}: proj = {proj}, u = {u}',
        'step_gram_schmidt_dependent': '  Вектор {idx} линейно зависим, пропускаем',
        'step_gram_schmidt_orthogonal': '  Ортогональный вектор u{idx} = {u}',
        'step_gram_schmidt_normalize': '  Нормируем e{idx} = {e}',
        'step_gram_schmidt_result': 'Результирующий базис: {basis}',
        'err_vectors_same_dim': 'Все векторы должны быть одинаковой размерности',
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

        # Vector widget controls
        'size': 'Size:',

        # Notebook tabs
        'matrix_tab': 'Matrices',
        'vector_tab': 'Vectors',

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

        # Vector titles
        'vector_a': 'Vector A',
        'vector_b': 'Vector B',

        # Step viewer header
        'step_solution': 'Step-by-step solution / Result',
        'result': 'Result:',
        'step_prefix': 'Step {number}:',
        'state': 'State',

        # Operation names (matrix)
        'op_add': 'Addition',
        'op_sub': 'Subtraction',
        'op_mul': 'Multiplication',
        'op_scalar_mul': 'Scalar Multiply',
        'op_transpose': 'Transpose A',
        'op_det': 'Determinant A',
        'op_rank': 'Rank A',
        'op_inv': 'Inverse A',
        'op_solve': 'Solve SLAE (A*X = B)',

        # Vector operation button labels
        'btn_add': 'A + B',
        'btn_subtract': 'A - B',
        'btn_dot': 'A · B',
        'btn_cross': 'A × B',
        'btn_norm_a': '||A||',
        'btn_norm_b': '||B||',
        'btn_normalize_a': 'Â',
        'btn_normalize_b': 'B̂',
        'btn_projection': 'proj_B A',
        'btn_angle': '∠(A,B)',
        'btn_triple': 'A·(B×C)',
        'btn_scalar_mul_a': 'k·A',
        'btn_scalar_mul_b': 'k·B',
        'show_steps': 'Show steps',

        # Vector dialogs
        'angle_unit': 'Angle unit',
        'use_degrees': 'Use degrees? (No = radians)',
        'triple_input': 'Triple product',
        'enter_vector_c': 'Enter vector C (space-separated numbers):',
        'scalar_input': 'Scalar multiplication',
        'enter_scalar': 'Enter scalar value:',
        'operations': 'Vector Operations',

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

        # Engine error messages (matrix)
        'err_square_matrix': '{operation} requires a square matrix. Got shape {shape}',
        'err_same_shape': '{operation} requires matrices of the same shape. Got {shape1} and {shape2}',
        'err_incompatible_mul': 'Incompatible shapes for multiplication: {shape1} and {shape2}',
        'err_singular_matrix': 'Matrix is singular (cannot be inverted)',
        'err_inversion_failed': 'Inversion failed: {msg}',
        'err_system_rows': 'Matrix A and RHS B must have the same number of rows',
        'err_no_unique_solution': 'System has no unique solution: {msg}',

        # Engine error messages (vector)
        'err_vector_1d': 'Vector must be 1‑dimensional (got shape {shape})',
        'err_vector_same_len': 'Vectors must have same length for {op} (got {len1} and {len2})',
        'err_cross_3d': 'Cross product requires 3D vectors (got {len1} and {len2})',
        'err_normalize_zero': 'Cannot normalize zero vector',
        'err_projection_zero_vec': 'Cannot project onto zero vector',
        'err_angle_zero_vec': 'Angle undefined for zero vector',
        'err_triple_3d': 'Triple product requires three 3D vectors',

        # Matrix step descriptions
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

        # Vector step descriptions
        'step_vector_add_init': 'Adding vectors: {v1} + {v2}',
        'step_vector_add_result': 'Result: {res}',
        'step_vector_sub_init': 'Subtracting vectors: {v1} - {v2}',
        'step_vector_sub_result': 'Result: {res}',
        'step_vector_scale_init': 'Scaling vector {v} by {scalar}',
        'step_vector_scale_result': 'Result: {res}',
        'step_dot_init': 'Dot product of {v1} and {v2}',
        'step_dot_products': 'Element‑wise products: {prods}',
        'step_dot_sum': 'Sum of products = {sum}',
        'step_cross_init': 'Cross product of {v1} × {v2}',
        'step_cross_components': 'Components: x={x}, y={y}, z={z}',
        'step_cross_result': 'Result vector: {res}',
        'step_norm_init': 'Computing norm of {v}',
        'step_norm_squares': 'Squares: {squares}',
        'step_norm_sum_sq': 'Sum of squares = {sum_sq}',
        'step_norm_result': 'Norm = {norm}',
        'step_normalize_init': 'Normalizing vector {v}',
        'step_normalize_norm': 'Norm = {norm}',
        'step_normalize_result': 'Unit vector: {unit}',
        'step_proj_init': 'Projection of {v1} onto {v2}',
        'step_proj_dot': 'Dot product (v1·v2) = {dot}',
        'step_proj_norm_sq': '||v2||² = {norm_sq}',
        'step_proj_scalar': 'Scalar factor = {scalar}',
        'step_proj_result': 'Projection vector: {proj}',
        'step_angle_init': 'Angle between {v1} and {v2}',
        'step_angle_dot': 'Dot product = {dot}',
        'step_angle_norms': '||v1|| = {norm1}, ||v2|| = {norm2}',
        'step_angle_cos': 'cos(θ) = {cos}',
        'step_angle_rad': 'θ = {rad} rad',
        'step_angle_deg': 'θ = {deg}°',
        'step_triple_init': 'Scalar triple product: {v1} · ({v2} × {v3})',
        'step_triple_cross': 'Cross product ({v2} × {v3}) = {cross}',
        'step_triple_dot': 'Dot product = {dot}',

        'load': 'Load...',
        'save': 'Save...',
        'load_vector': 'Load {title}',
        'save_vector': 'Save {title}',

        'step_dot_products_detail': 'Element-wise products: {detail} = {prods}',
        'step_cross_determinant': 'Determinant: i·({i}) - j·({j}) + k·({k})',
        'step_cross_determinant_numeric': 'Determinant: |i  j  k|\n|{a1} {a2} {a3}|\n|{b1} {b2} {b3}|',
        'step_cross_components_calc': 'Components:\n  x = {x_expr}\n  y = {y_expr}\n  z = {z_expr}',

        'step_triple_cross_start': '1. Compute cross product {v2} × {v3}:',
        'step_triple_cross_start_numeric': '1. Cross product B × C:',
        'step_triple_cross_components_calc': '   x = {x_expr}\n   y = {y_expr}\n   z = {z_expr}',
        'step_triple_dot_start': '2. Dot product {v1} · ({v2} × {v3}):',
        'step_triple_dot_start_numeric': '2. Dot product A · (B × C):',
        'step_triple_dot_products_detail': '   Element-wise: {detail} = {prods}',
        'step_triple_dot_products': '   Products: {prods}',
        'step_triple_dot_sum': '   Sum = {sum:.6f}',
        'step_triple_result': 'Result (scalar triple product) = {result}',
        'step_triple_result_volume': 'Result: A·(B×C) = {result:.6f}\nParallelepiped volume = |{result:.6f}| = {'
                                     'volume:.6f}',

        'symbolic_mode': 'Symbolic Mode',

        # Special relations
        'input_vectors': 'Input Vectors',
        'comma_separated': '(comma/space separated)',
        'parameter_name': 'Parameter:',
        'btn_collinear_check': 'Collinear?',
        'btn_collinear_param': 'Find λ for collinearity',
        'btn_orthogonal': 'Orthogonal?',
        'btn_coplanar': 'Coplanar?',
        'collinearity_check': 'Collinearity Check',
        'collinearity_param': 'Collinearity Parameter',
        'orthogonality_check': 'Orthogonality Check',
        'coplanarity_check': 'Coplanarity Check',

        'step_collinear_init': 'Vectors: a = {v1}, b = {v2}',
        'step_collinear_2d_eq': 'Collinearity condition (2D): {eq}',
        'step_collinear_3d_cross': 'Cross product: a × b = {cross}',
        'step_collinear_3d_eq': 'Set component to zero: {eq}',
        'step_collinear_solve': 'Solve for {param}: {solutions}',

        'vector_to_decompose': 'Vector to decompose',
        'basis_vectors': 'Basis vectors',
        'btn_check_basis': 'Check basis',
        'btn_decompose': 'Decompose',
        'basis_check': 'Basis Check',
        'decomposition_result': 'Coordinates in basis',
        'coordinates': 'Coordinates',

        'err_basis_dimension': 'All vectors must have same dimension',
        'err_basis_count': 'Expected {expected} vectors, got {got}',

        'step_decompose_init': 'Decompose vector {v} in basis {basis}',
        'step_decompose_matrix': 'Transition matrix (basis columns):\n{matrix}',
        'step_decompose_solve': 'Solve system {system}',
        'step_decompose_result': 'Coefficients: {coeffs}',
        'step_decompose_matrix_numeric': 'Transition matrix:\n{matrix}',
        'step_decompose_solve_numeric': 'Solve linear system',
        'step_decompose_result_numeric': 'Coordinates: {coeffs}',

        'step_transition_init': 'Building transition matrix',
        'step_transition_formula': 'P = (new basis)⁻¹ · (old basis)',
        'step_transition_matrix': 'Transition matrix:\n{matrix}',

        'points_coordinates': 'Point Coordinates',
        'btn_collinear_points': 'Points collinear?',
        'btn_coplanar_points': 'Points coplanar?',
        'btn_triangle_area': 'Triangle Area (A,B,C)',
        'btn_tetrahedron_volume': 'Tetrahedron Volume (A,B,C,D)',
        'points_collinear': 'Points Collinearity',
        'points_coplanar': 'Points Coplanarity',
        'triangle_area': 'Triangle Area',
        'tetrahedron_volume': 'Tetrahedron Volume',
        'err_3d_required': '3D vectors required',
        'err_coplanar_3d': 'Coplanarity defined only for 3D vectors',

        'special_tab': 'Special Relations',
        'basis_tab': 'Basis and decomposition',
        'geometry_tab': 'Geometry',

        # Eigenvalues
        'eigen_tab': 'Eigenvalues',
        'input_matrix': 'Matrix',
        'btn_charpoly': 'Characteristic Polynomial',
        'btn_eigenvalues': 'Eigenvalues',
        'btn_eigenvectors': 'Eigenvectors',
        'btn_diagonalize': 'Diagonalize',
        'btn_copy_from_a': '← Copy from A',
        'charpoly_result': 'Characteristic Polynomial',
        'eigenvalues_result': 'Eigenvalues',
        'eigenvectors_result': 'Eigenvectors',
        'diagonalization_result': 'Diagonalization',
        'err_not_diagonalizable': 'Matrix is not diagonalizable',

        'step_charpoly_init': 'Build characteristic polynomial det(A - {var}I)',
        'step_charpoly_matrix': 'Matrix A - {var}I:\n{matrix}',
        'step_charpoly_det': 'Determinant = {poly}',
        'step_charpoly_matrix_numeric': 'Matrix A:\n{matrix}',
        'step_charpoly_coeffs': 'Characteristic polynomial coefficients: {coeffs}',

        'step_eigenvals_init': 'Find eigenvalues',
        'step_eigenvals_charpoly': 'Characteristic polynomial: {poly}',
        'step_eigenvals_roots': 'Roots (eigenvalues): {vals}',
        'step_eigenvals_charpoly_numeric': 'Coefficients: {coeffs}',
        'step_eigenvals_result': 'Eigenvalues: {vals}',

        'step_eigenvecs_init': 'Find eigenvectors',
        'step_eigenvecs_charpoly': 'Characteristic polynomial: {poly}',
        'step_eigenvecs_for_val': 'For λ = {val} (multiplicity {mult}):',
        'step_eigenvecs_vec': '  Vector {idx}: {vec}',
        'step_eigenvecs_vals': 'Eigenvalues: {vals}',
        'step_eigenvecs_matrix': 'Eigenvector matrix (columns):\n{matrix}',

        'step_diag_init': 'Diagonalize matrix',
        'step_diag_eigenvals': 'Eigenvalues (diagonal D):\n{vals}',
        'step_diag_eigenvecs': 'Matrix P (eigenvectors):\n{matrix}',
        'step_diag_verify': 'Verification: P·D·P⁻¹ = {product}',
        'step_diag_eigenvals_numeric': 'Eigenvalues: {vals}',
        'step_diag_eigenvecs_numeric': 'Matrix P:\n{matrix}',
        'step_diag_verify_numeric': 'Verification: P·D·P⁻¹ =\n{product}',

        'gram_schmidt_tab': 'Gram-Schmidt',
        'visualization_tab': 'Visualization',
        'vectors_to_plot': 'Vectors:',
        'comma_separated_vectors': '(x,y,z; ...)',
        'btn_plot': 'Plot',
        'btn_clear': 'Clear',
        'vector_plot': '3D Vectors',
        'vectors_format': 'One vector per line: x y z (or x,y,z)',
        'normalize': 'Normalize',
        'btn_gram_schmidt': 'Orthogonalize',
        'gram_schmidt_result': 'Gram-Schmidt Result',
        'orthogonal_basis': 'Orthogonal Basis',
        'export_latex': 'Export to LaTeX...',
        'latex_export_success': 'File saved successfully',
        'success': 'Success',

        'step_gram_schmidt_init': 'Initial vectors: {vectors}',
        'step_gram_schmidt_start': 'Processing vector {idx}: v = {v}',
        'step_gram_schmidt_subtract': '  Subtract projection onto e{j}: proj = {proj}, u = {u}',
        'step_gram_schmidt_dependent': '  Vector {idx} is linearly dependent, skipping',
        'step_gram_schmidt_orthogonal': '  Orthogonal vector u{idx} = {u}',
        'step_gram_schmidt_normalize': '  Normalize e{idx} = {e}',
        'step_gram_schmidt_result': 'Resulting basis: {basis}',
        'err_vectors_same_dim': 'All vectors must have the same dimension',
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
