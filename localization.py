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
        'export_latex': 'Экспорт в LaTeX...',
        'latex_export_success': 'Файл успешно сохранён',

        # Toolbar
        'operation': 'Операция:',
        'scalar': 'Скаляр:',
        'compute': 'Вычислить',
        'cancel': 'Отмена',
        'swap': '⇄ Поменять A↔B',
        'precision': 'Точность:',
        'language': 'Язык:',
        'symbolic_mode': 'Символьный режим',

        # Matrix widget controls
        'rows': 'Строк:',
        'cols': 'Столбцов:',
        'resize': 'Изменить размер',

        # Vector widget controls
        'size': 'Размер:',
        'load': 'Загрузить...',
        'save': 'Сохранить...',
        'load_vector': 'Загрузить {title}',
        'save_vector': 'Сохранить {title}',

        # Notebook tabs
        'matrix_tab': 'Матрицы',
        'vector_tab': 'Векторы',
        'special_tab': 'Специальные отношения',
        'basis_tab': 'Базис и разложение',
        'geometry_tab': 'Геометрия',
        'eigen_tab': 'Собственные значения',
        'gram_schmidt_tab': 'Ортогонализация',
        'visualization_tab': 'Визуализация',

        # Status bar
        'ready': 'Готов',
        'computing': 'Вычисление...',
        'cancel_request': 'Отмена...',
        'swap_success': 'Матрицы обменяны',
        'precision_set': 'Точность: {}',
        'success': 'Успех',
        'mode_status': 'Режим: {mode}',
        'symbolic_mode_short': 'Символьный',
        'numeric_mode_short': 'Числовой',

        # Matrix titles
        'matrix_a': 'Матрица A',
        'matrix_b': 'Матрица B',
        'matrix_a_coeff': 'A (коэффициенты)',
        'matrix_b_rhs': 'B (свободные члены)',

        # Vector titles
        'vector_a': 'Вектор A',
        'vector_b': 'Вектор B',

        # Step viewer
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
        'op_solve_gauss': 'Решение СЛАУ (Гаусс)',
        'op_solve_cramer': 'Решение СЛАУ (Крамер)',
        'op_solve_inverse': 'Решение СЛАУ (Обратная матрица)',

        # Vector operation buttons
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

        # Special relations UI
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

        # Basis UI
        'vector_to_decompose': 'Вектор для разложения',
        'basis_vectors': 'Базисные векторы',
        'btn_check_basis': 'Проверить базис',
        'btn_decompose': 'Разложить',
        'btn_transition': 'Матрица перехода',
        'basis_check': 'Проверка базиса',
        'decomposition_result': 'Координаты в базисе',
        'coordinates': 'Координаты',
        'coordinate_converter': 'Преобразователь координат',
        'vector_in_old_basis': 'Вектор в старом базисе:',
        'convert': 'Преобразовать',
        'converted_coords': 'Координаты в новом базисе',
        'err_no_transition': 'Сначала вычислите матрицу перехода',
        'new_basis': 'Новый базис',
        'enter_new_basis': 'Введите векторы нового базиса (по одному на строку):',

        # Geometry UI
        'points_coordinates': 'Координаты точек',
        'btn_collinear_points': 'Точки на одной прямой?',
        'btn_coplanar_points': 'Точки в одной плоскости?',
        'btn_triangle_area': 'Площадь треугольника (A,B,C)',
        'btn_tetrahedron_volume': 'Объём тетраэдра (A,B,C,D)',
        'points_collinear': 'Коллинеарность точек',
        'points_coplanar': 'Компланарность точек',
        'triangle_area': 'Площадь треугольника',
        'tetrahedron_volume': 'Объём тетраэдра',

        # Eigenvalues UI
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
        'eigenvecs_matrix': 'P (собственные векторы)',
        'eigenvals_diag': 'D (собственные значения)',

        # Gram-Schmidt UI
        'normalize': 'Нормировать',
        'btn_gram_schmidt': 'Ортогонализовать',
        'gram_schmidt_result': 'Результат ортогонализации',
        'orthogonal_basis': 'Ортогональный базис',

        # Visualization UI
        'vectors_to_plot': 'Векторы:',
        'comma_separated_vectors': '(x,y,z; ...)',
        'btn_plot': 'Построить',
        'btn_clear': 'Очистить',
        'vector_plot': 'Векторы в 3D',
        'vectors_format': 'Каждый вектор с новой строки: x y z (или x,y,z)',

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

        # Engine errors (matrix)
        'err_square_matrix': '{operation} требует квадратную матрицу. Получена форма {shape}',
        'err_same_shape': '{operation} требует матрицы одинаковой формы. Получены {shape1} и {shape2}',
        'err_incompatible_mul': 'Несовместимые размеры для умножения: {shape1} и {shape2}',
        'err_singular_matrix': 'Матрица вырождена (невозможно обратить)',
        'err_inversion_failed': 'Обращение не удалось: {msg}',
        'err_system_rows': 'Матрица A и правая часть B должны иметь одинаковое количество строк',
        'err_no_unique_solution': 'Система не имеет единственного решения: {msg}',
        'err_not_diagonalizable': 'Матрица не диагонализируема',
        'err_inconsistent_system': 'Система несовместна (нет решений)',
        'err_no_solution': 'Решение не найдено',

        # Engine errors (vector)
        'err_vector_1d': 'Вектор должен быть одномерным (получена форма {shape})',
        'err_vector_same_len': 'Векторы должны иметь одинаковую длину для {op} (получено {len1} и {len2})',
        'err_cross_3d': 'Векторное произведение требует трёхмерные векторы (получено {len1} и {len2})',
        'err_normalize_zero': 'Невозможно нормализовать нулевой вектор',
        'err_projection_zero_vec': 'Невозможно спроецировать на нулевой вектор',
        'err_angle_zero_vec': 'Угол не определён для нулевого вектора',
        'err_triple_3d': 'Смешанное произведение требует три трёхмерных вектора',
        'err_basis_dimension': 'Все векторы должны быть одной размерности',
        'err_basis_count': 'Требуется {expected} векторов, получено {got}',
        'err_3d_required': 'Требуются 3D векторы',
        'err_coplanar_3d': 'Компланарность определена только для 3D векторов',
        'err_vectors_same_dim': 'Все векторы должны быть одинаковой размерности',
        'err_cramer_single_rhs': 'Метод Крамера поддерживает только один столбец правой части',

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
        'step_rank_symbolic': 'Ранг матрицы (символьный): {rank}',
        'step_slae_rank': 'Ранг(A) = {rank}, дефект = {nullity}, свободных переменных = {free}',
        'step_slae_rank_symbolic': 'Ранг(A) = {rank}, дефект = {nullity}, свободных переменных = {free}',
        'step_slae_general_solution': 'Общее решение: x = {sol}',
        'step_slae_particular': 'Частное решение: xₚ = {x}',
        'step_slae_null_vector': 'Базисный вектор ядра {idx}: {vec}',

        # Determinant symbolic steps
        'step_det_1x1': 'Определитель 1×1: det = {val}',
        'step_det_2x2': 'Определитель 2×2: {a}·{d} - {b}·{c} = {det}',
        'step_det_3x3_expand': 'Вычисляем по правилу Саррюса или разложением...',
        'step_det_symbolic': 'Определитель (символьный): {det}',

        # Cramer's Rule steps
        'step_cramer_init': '=== Решение методом Крамера ===',
        'step_cramer_det_A': 'Шаг 1: определитель матрицы A: det(A) = {det}',
        'step_cramer_replace': 'Шаг {step}: заменяем столбец {col} на B, det(A{col}) = {det_i}, x{col} = {det_i} / det(A) = {xi}',
        'step_cramer_result': 'Решение системы: x = {x}',

        # Inverse matrix method steps
        'step_inverse_init': '=== Решение через обратную матрицу ===',
        'step_inverse_compute': 'Шаг 1: находим обратную матрицу A⁻¹:',
        'step_inverse_multiply': 'Шаг 2: умножаем A⁻¹ на B:',
        'step_inverse_component': '  x{i} = {expr} = {val}',
        'step_inverse_result': 'Решение: x = {x}',
        'step_inverse_result_matrix': 'Результирующая матрица X:\n{x}',

        # Vector basic steps
        'step_vector_add_init': 'Сложение векторов: {v1} + {v2}',
        'step_vector_add_result': 'Результат: {res}',
        'step_vector_sub_init': 'Вычитание векторов: {v1} - {v2}',
        'step_vector_sub_result': 'Результат: {res}',
        'step_vector_scale_init': 'Умножение вектора {v} на {scalar}',
        'step_vector_scale_result': 'Результат: {res}',
        'step_dot_init': 'Скалярное произведение {v1} и {v2}',
        'step_dot_products': 'Поэлементные произведения: {prods}',
        'step_dot_products_detail': 'Поэлементные произведения: {detail} = {prods}',
        'step_dot_sum': 'Сумма произведений = {sum:.6f}',
        'step_cross_init': 'Векторное произведение {v1} × {v2}',
        'step_cross_determinant': 'Определитель: i·({i}) - j·({j}) + k·({k})',
        'step_cross_determinant_numeric': 'Определитель: |i  j  k|\n|{a1} {a2} {a3}|\n|{b1} {b2} {b3}|',
        'step_cross_components': 'Компоненты: x={x}, y={y}, z={z}',
        'step_cross_components_calc': 'Компоненты:\n  x = {x_expr}\n  y = {y_expr}\n  z = {z_expr}',
        'step_cross_result': 'Результирующий вектор: {res}',
        'step_norm_init': 'Вычисление нормы вектора {v}',
        'step_norm_squares': 'Квадраты элементов: {squares}',
        'step_norm_sum_sq': 'Сумма квадратов = {sum_sq:.6f}',
        'step_norm_result': 'Норма = {norm:.6f}',
        'step_normalize_init': 'Нормализация вектора {v}',
        'step_normalize_norm': 'Норма = {norm:.6f}',
        'step_normalize_result': 'Единичный вектор: {unit}',
        'step_proj_init': 'Проекция {v1} на {v2}',
        'step_proj_dot': 'Скалярное произведение (v1·v2) = {dot:.6f}',
        'step_proj_norm_sq': '||v2||² = {norm_sq:.6f}',
        'step_proj_scalar': 'Скалярный коэффициент = {scalar:.6f}',
        'step_proj_result': 'Вектор проекции: {proj}',
        'step_angle_init': 'Угол между {v1} и {v2}',
        'step_angle_dot': 'Скалярное произведение = {dot:.6f}',
        'step_angle_norms': '||v1|| = {norm1:.6f}, ||v2|| = {norm2:.6f}',
        'step_angle_cos': 'cos(θ) = {cos:.6f}',
        'step_angle_rad': 'θ = {rad:.6f} рад',
        'step_angle_deg': 'θ = {deg:.6f}°',

        # Triple product steps
        'step_triple_init': 'Смешанное произведение: {v1} · ({v2} × {v3})',
        'step_triple_cross_start': '1. Вычисляем векторное произведение {v2} × {v3}:',
        'step_triple_cross_start_numeric': '1. Векторное произведение B × C:',
        'step_triple_cross_components': 'Компоненты векторного произведения: x={x}, y={y}, z={z}',
        'step_triple_cross_components_calc': '   x = {x_expr}\n   y = {y_expr}\n   z = {z_expr}',
        'step_triple_cross': 'Векторное произведение ({v2} × {v3}) = {cross}',
        'step_triple_dot_start': '2. Скалярное произведение {v1} · ({v2} × {v3}):',
        'step_triple_dot_start_numeric': '2. Скалярное произведение A · (B × C):',
        'step_triple_dot_products': '   Произведения: {prods}',
        'step_triple_dot_products_detail': '   Поэлементно: {detail} = {prods}',
        'step_triple_dot_sum': '   Сумма = {sum:.6f}',
        'step_triple_dot': 'Скалярное произведение = {dot:.6f}',
        'step_triple_result': 'Результат (смешанное произведение) = {result}',
        'step_triple_result_volume': 'Результат: A·(B×C) = {result:.6f}\nОбъём параллелепипеда = |{result:.6f}| = {volume:.6f}',

        # Collinearity steps
        'step_collinear_init': 'Векторы: a = {v1}, b = {v2}',
        'step_collinear_false_dim': 'Векторы разной размерности → не коллинеарны',
        'step_collinear_2d_det': '2D определитель = {det}',
        'step_collinear_2d_det_numeric': '2D определитель = {det:.6f}',
        'step_collinear_2d_eq': 'Условие коллинеарности (2D): {eq}',
        'step_collinear_3d_cross': 'Векторное произведение = {cross}',
        'step_collinear_3d_cross_numeric': 'Векторное произведение = {cross}',
        'step_collinear_3d_eq': 'Приравниваем компоненту к нулю: {eq}',
        'step_collinear_solve': 'Решаем относительно {param}: {solutions}',
        'step_collinear_result': 'Результат: {result}',

        # Orthogonality steps
        'step_orthogonal_dot': 'Скалярное произведение = {dot}',
        'step_orthogonal_result': 'Ортогональны? {result}',

        # Coplanarity steps
        'step_coplanar_cross': 'Векторное произведение v2×v3 = {cross}',
        'step_coplanar_cross_numeric': 'Векторное произведение = {cross}',
        'step_coplanar_triple': 'Смешанное произведение = {triple}',
        'step_coplanar_triple_numeric': 'Смешанное произведение = {triple:.6f}',
        'step_coplanar_result': 'Компланарны? {result}',

        # Basis steps
        'step_basis_det': 'Определитель матрицы из векторов = {det}',
        'step_basis_det_numeric': 'Определитель = {det:.6f}',
        'step_basis_result': 'Образуют базис? {result}',

        # Decomposition steps
        'step_decompose_init': 'Разложение вектора {v} по базису {basis}',
        'step_decompose_matrix': 'Матрица перехода (базис по столбцам):\n{matrix}',
        'step_decompose_matrix_numeric': 'Матрица перехода:\n{matrix}',
        'step_decompose_system': 'Система уравнений:\n{system}',
        'step_decompose_solve': 'Решаем систему',
        'step_decompose_solve_numeric': 'Решаем линейную систему',
        'step_decompose_result': 'Коэффициенты: {coeffs}',
        'step_decompose_result_numeric': 'Координаты: {coeffs}',

        # Transition matrix steps
        'step_transition_init': 'Построение матрицы перехода',
        'step_transition_formula': 'P = (новый базис)⁻¹ · (старый базис)',
        'step_transition_matrix': 'Матрица перехода:\n{matrix}',
        'transition_matrix': 'Матрица перехода',

        # Eigenvalue steps
        'step_charpoly_init': 'Построение характеристического полинома det(A - λI)',
        'step_charpoly_matrix': 'Матрица A - λI:\n{matrix}',
        'step_charpoly_matrix_numeric': 'Матрица A:\n{matrix}',
        'step_charpoly_det': 'Определитель = {poly}',
        'step_charpoly_coeffs': 'Коэффициенты характеристического полинома: {coeffs}',
        'step_eigenvals_init': 'Поиск собственных значений',
        'step_eigenvals_charpoly': 'Характеристический полином: {poly}',
        'step_eigenvals_charpoly_numeric': 'Коэффициенты: {coeffs}',
        'step_eigenvals_roots': 'Корни (собственные значения): {vals}',
        'step_eigenvals_result': 'Собственные значения: {vals}',
        'step_eigenvecs_init': 'Поиск собственных векторов',
        'step_eigenvecs_charpoly': 'Характеристический полином: {poly}',
        'step_eigenvecs_vals': 'Собственные значения: {vals}',
        'step_eigenvecs_for_val': 'Для λ = {val} (кратность {mult}):',
        'step_eigenvecs_vec': '  Вектор {idx}: {vec}',
        'step_eigenvecs_matrix': 'Матрица собственных векторов (по столбцам):\n{matrix}',
        'step_diag_init': 'Диагонализация матрицы',
        'step_diag_eigenvals': 'Собственные значения (диагональ D):\n{vals}',
        'step_diag_eigenvals_numeric': 'Собственные значения: {vals}',
        'step_diag_eigenvecs': 'Матрица P (собственные векторы):\n{matrix}',
        'step_diag_eigenvecs_numeric': 'Матрица P:\n{matrix}',
        'step_diag_verify': 'Проверка: P·D·P⁻¹ = {product}',
        'step_diag_verify_numeric': 'Проверка: P·D·P⁻¹ =\n{product}',

        # Gram-Schmidt steps
        'step_gram_schmidt_init': 'Начальные векторы: {vectors}',
        'step_gram_schmidt_start': 'Обработка вектора {idx}: v = {v}',
        'step_gram_schmidt_subtract': '  Вычитаем проекцию на e{j}: proj = {proj}, u = {u}',
        'step_gram_schmidt_dependent': '  Вектор {idx} линейно зависим, пропускаем',
        'step_gram_schmidt_orthogonal': '  Ортогональный вектор u{idx} = {u}',
        'step_gram_schmidt_normalize': '  Нормируем e{idx} = {e}',
        'step_gram_schmidt_result': 'Результирующий базис: {basis}',

        # Geometry steps
        'step_points_collinear_init': 'Проверка коллинеарности точек A={A}, B={B}, C={C}',
        'step_points_coplanar_init': 'Проверка компланарности точек A={A}, B={B}, C={C}, D={D}',
        'step_triangle_area_init': 'Вычисление площади треугольника с вершинами A={A}, B={B}, C={C}',
        'step_triangle_area_vectors': 'Векторы сторон: AB = {AB}, AC = {AC}',
        'step_triangle_area_cross': 'Векторное произведение AB × AC = {cross}',
        'step_triangle_area_result': 'Площадь S = 0.5 * |AB × AC| = {area:.6f}',
        'step_tetrahedron_volume_init': 'Вычисление объёма тетраэдра A={A}, B={B}, C={C}, D={D}',
        'step_tetrahedron_volume_vectors': 'Векторы рёбер: AB = {AB}, AC = {AC}, AD = {AD}',
        'step_tetrahedron_volume_cross': 'Векторное произведение AC × AD = {cross}',
        'step_tetrahedron_volume_triple': 'Смешанное произведение AB · (AC × AD) = {triple}',
        'step_tetrahedron_volume_result': 'Объём V = |triple| / 6 = {volume:.6f}',

        # Warnings
        'symbolic_mode_long': 'символьном',
        'numeric_mode': 'числовом',
        'warn_heavy_symbolic_op': 'Символьная операция «{op}» над матрицей {size}×{size} может занять много времени. Продолжить?',
        'warn_heavy_operation_detailed': 'Операция «{op}» над матрицей {size}×{size} в {mode} режиме может выполняться очень долго (возможно, бесконечно). Продолжить?',

        # Labels
        'v1_label': 'v1:',
        'v2_label': 'v2:',
        'v3_label': 'v3:',
        'v_label': 'v:',
        'b_label': 'b',
        'xyz_format': '(x,y,z)',

        'old_basis': 'Старый базис',
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
        'export_latex': 'Export to LaTeX...',
        'latex_export_success': 'File saved successfully',

        # Toolbar
        'operation': 'Operation:',
        'scalar': 'Scalar:',
        'compute': 'Compute',
        'cancel': 'Cancel',
        'swap': '⇄ Swap A↔B',
        'precision': 'Precision:',
        'language': 'Language:',
        'symbolic_mode': 'Symbolic Mode',

        # Matrix widget controls
        'rows': 'Rows:',
        'cols': 'Cols:',
        'resize': 'Resize',

        # Vector widget controls
        'size': 'Size:',
        'load': 'Load...',
        'save': 'Save...',
        'load_vector': 'Load {title}',
        'save_vector': 'Save {title}',

        # Notebook tabs
        'matrix_tab': 'Matrices',
        'vector_tab': 'Vectors',
        'special_tab': 'Special Relations',
        'basis_tab': 'Basis & Decomposition',
        'geometry_tab': 'Geometry',
        'eigen_tab': 'Eigenvalues',
        'gram_schmidt_tab': 'Gram–Schmidt',
        'visualization_tab': 'Visualization',

        # Status bar
        'ready': 'Ready',
        'computing': 'Computing...',
        'cancel_request': 'Canceling...',
        'swap_success': 'Matrices swapped',
        'precision_set': 'Precision: {}',
        'success': 'Success',
        'mode_status': 'Mode: {mode}',
        'symbolic_mode_short': 'Symbolic',
        'numeric_mode_short': 'Numeric',

        # Matrix titles
        'matrix_a': 'Matrix A',
        'matrix_b': 'Matrix B',
        'matrix_a_coeff': 'A (coefficients)',
        'matrix_b_rhs': 'B (right-hand side)',

        # Vector titles
        'vector_a': 'Vector A',
        'vector_b': 'Vector B',

        # Step viewer
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
        'op_solve_gauss': 'Solve SLAE (Gauss)',
        'op_solve_cramer': 'Solve SLAE (Cramer)',
        'op_solve_inverse': 'Solve SLAE (Inverse)',

        # Vector operation buttons
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

        # Special relations UI
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

        # Basis UI
        'vector_to_decompose': 'Vector to decompose',
        'basis_vectors': 'Basis vectors',
        'btn_check_basis': 'Check basis',
        'btn_decompose': 'Decompose',
        'btn_transition': 'Transition matrix',
        'basis_check': 'Basis Check',
        'decomposition_result': 'Coordinates in basis',
        'coordinates': 'Coordinates',
        'coordinate_converter': 'Coordinate Converter',
        'vector_in_old_basis': 'Vector in old basis:',
        'convert': 'Convert',
        'converted_coords': 'Coordinates in new basis',
        'err_no_transition': 'Compute transition matrix first',
        'new_basis': 'New basis',
        'enter_new_basis': 'Enter vectors of new basis (one per line):',

        # Geometry UI
        'points_coordinates': 'Point Coordinates',
        'btn_collinear_points': 'Points collinear?',
        'btn_coplanar_points': 'Points coplanar?',
        'btn_triangle_area': 'Triangle Area (A,B,C)',
        'btn_tetrahedron_volume': 'Tetrahedron Volume (A,B,C,D)',
        'points_collinear': 'Points Collinearity',
        'points_coplanar': 'Points Coplanarity',
        'triangle_area': 'Triangle Area',
        'tetrahedron_volume': 'Tetrahedron Volume',

        # Eigenvalues UI
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
        'eigenvecs_matrix': 'P (eigenvectors)',
        'eigenvals_diag': 'D (eigenvalues)',

        # Gram-Schmidt UI
        'normalize': 'Normalize',
        'btn_gram_schmidt': 'Orthogonalize',
        'gram_schmidt_result': 'Gram–Schmidt Result',
        'orthogonal_basis': 'Orthogonal Basis',

        # Visualization UI
        'vectors_to_plot': 'Vectors:',
        'comma_separated_vectors': '(x,y,z; ...)',
        'btn_plot': 'Plot',
        'btn_clear': 'Clear',
        'vector_plot': '3D Vectors',
        'vectors_format': 'One vector per line: x y z (or x,y,z)',

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

        # Engine errors (matrix)
        'err_square_matrix': '{operation} requires a square matrix. Got shape {shape}',
        'err_same_shape': '{operation} requires matrices of the same shape. Got {shape1} and {shape2}',
        'err_incompatible_mul': 'Incompatible shapes for multiplication: {shape1} and {shape2}',
        'err_singular_matrix': 'Matrix is singular (cannot be inverted)',
        'err_inversion_failed': 'Inversion failed: {msg}',
        'err_system_rows': 'Matrix A and RHS B must have the same number of rows',
        'err_no_unique_solution': 'System has no unique solution: {msg}',
        'err_not_diagonalizable': 'Matrix is not diagonalizable',
        'err_inconsistent_system': 'System is inconsistent (no solution)',
        'err_no_solution': 'No solution found',

        # Engine errors (vector)
        'err_vector_1d': 'Vector must be 1‑dimensional (got shape {shape})',
        'err_vector_same_len': 'Vectors must have same length for {op} (got {len1} and {len2})',
        'err_cross_3d': 'Cross product requires 3D vectors (got {len1} and {len2})',
        'err_normalize_zero': 'Cannot normalize zero vector',
        'err_projection_zero_vec': 'Cannot project onto zero vector',
        'err_angle_zero_vec': 'Angle undefined for zero vector',
        'err_triple_3d': 'Triple product requires three 3D vectors',
        'err_basis_dimension': 'All vectors must have same dimension',
        'err_basis_count': 'Expected {expected} vectors, got {got}',
        'err_3d_required': '3D vectors required',
        'err_coplanar_3d': 'Coplanarity defined only for 3D vectors',
        'err_vectors_same_dim': 'All vectors must have the same dimension',
        'err_cramer_single_rhs': 'Cramer\'s rule supports only a single right-hand side column',

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
        'step_rank_symbolic': 'Matrix rank (symbolic): {rank}',
        'step_slae_rank': 'Rank(A) = {rank}, Nullity = {nullity}, Free variables = {free}',
        'step_slae_rank_symbolic': 'Rank(A) = {rank}, Nullity = {nullity}, Free variables = {free}',
        'step_slae_general_solution': 'General solution: x = {sol}',
        'step_slae_particular': 'Particular solution: xₚ = {x}',
        'step_slae_null_vector': 'Null space basis vector {idx}: {vec}',

        # Determinant symbolic steps
        'step_det_1x1': 'Determinant 1×1: det = {val}',
        'step_det_2x2': 'Determinant 2×2: {a}·{d} - {b}·{c} = {det}',
        'step_det_3x3_expand': 'Computing via Sarrus rule or expansion...',
        'step_det_symbolic': 'Determinant (symbolic): {det}',

        # Cramer's Rule steps
        'step_cramer_init': '=== Solving with Cramer\'s Rule ===',
        'step_cramer_det_A': 'Step 1: determinant of A: det(A) = {det}',
        'step_cramer_replace': 'Step {step}: replace column {col} with B, det(A{col}) = {det_i}, x{col} = {det_i} / det(A) = {xi}',
        'step_cramer_result': 'Solution: x = {x}',

        # Inverse matrix method steps
        'step_inverse_init': '=== Solving with Inverse Matrix ===',
        'step_inverse_compute': 'Step 1: compute inverse matrix A⁻¹:',
        'step_inverse_multiply': 'Step 2: multiply A⁻¹ by B:',
        'step_inverse_component': '  x{i} = {expr} = {val}',
        'step_inverse_result': 'Solution: x = {x}',
        'step_inverse_result_matrix': 'Resulting matrix X:\n{x}',

        # Vector basic steps
        'step_vector_add_init': 'Adding vectors: {v1} + {v2}',
        'step_vector_add_result': 'Result: {res}',
        'step_vector_sub_init': 'Subtracting vectors: {v1} - {v2}',
        'step_vector_sub_result': 'Result: {res}',
        'step_vector_scale_init': 'Scaling vector {v} by {scalar}',
        'step_vector_scale_result': 'Result: {res}',
        'step_dot_init': 'Dot product of {v1} and {v2}',
        'step_dot_products': 'Element‑wise products: {prods}',
        'step_dot_products_detail': 'Element-wise products: {detail} = {prods}',
        'step_dot_sum': 'Sum of products = {sum:.6f}',
        'step_cross_init': 'Cross product of {v1} × {v2}',
        'step_cross_determinant': 'Determinant: i·({i}) - j·({j}) + k·({k})',
        'step_cross_determinant_numeric': 'Determinant: |i  j  k|\n|{a1} {a2} {a3}|\n|{b1} {b2} {b3}|',
        'step_cross_components': 'Components: x={x}, y={y}, z={z}',
        'step_cross_components_calc': 'Components:\n  x = {x_expr}\n  y = {y_expr}\n  z = {z_expr}',
        'step_cross_result': 'Result vector: {res}',
        'step_norm_init': 'Computing norm of {v}',
        'step_norm_squares': 'Squares: {squares}',
        'step_norm_sum_sq': 'Sum of squares = {sum_sq:.6f}',
        'step_norm_result': 'Norm = {norm:.6f}',
        'step_normalize_init': 'Normalizing vector {v}',
        'step_normalize_norm': 'Norm = {norm:.6f}',
        'step_normalize_result': 'Unit vector: {unit}',
        'step_proj_init': 'Projection of {v1} onto {v2}',
        'step_proj_dot': 'Dot product (v1·v2) = {dot:.6f}',
        'step_proj_norm_sq': '||v2||² = {norm_sq:.6f}',
        'step_proj_scalar': 'Scalar factor = {scalar:.6f}',
        'step_proj_result': 'Projection vector: {proj}',
        'step_angle_init': 'Angle between {v1} and {v2}',
        'step_angle_dot': 'Dot product = {dot:.6f}',
        'step_angle_norms': '||v1|| = {norm1:.6f}, ||v2|| = {norm2:.6f}',
        'step_angle_cos': 'cos(θ) = {cos:.6f}',
        'step_angle_rad': 'θ = {rad:.6f} rad',
        'step_angle_deg': 'θ = {deg:.6f}°',

        # Triple product steps
        'step_triple_init': 'Scalar triple product: {v1} · ({v2} × {v3})',
        'step_triple_cross_start': '1. Compute cross product {v2} × {v3}:',
        'step_triple_cross_start_numeric': '1. Cross product B × C:',
        'step_triple_cross_components': 'Cross product components: x={x}, y={y}, z={z}',
        'step_triple_cross_components_calc': '   x = {x_expr}\n   y = {y_expr}\n   z = {z_expr}',
        'step_triple_cross': 'Cross product ({v2} × {v3}) = {cross}',
        'step_triple_dot_start': '2. Dot product {v1} · ({v2} × {v3}):',
        'step_triple_dot_start_numeric': '2. Dot product A · (B × C):',
        'step_triple_dot_products': '   Products: {prods}',
        'step_triple_dot_products_detail': '   Element-wise: {detail} = {prods}',
        'step_triple_dot_sum': '   Sum = {sum:.6f}',
        'step_triple_dot': 'Dot product = {dot:.6f}',
        'step_triple_result': 'Result (scalar triple product) = {result}',
        'step_triple_result_volume': 'Result: A·(B×C) = {result:.6f}\nParallelepiped volume = |{result:.6f}| = {volume:.6f}',

        # Collinearity steps
        'step_collinear_init': 'Vectors: a = {v1}, b = {v2}',
        'step_collinear_false_dim': 'Vectors have different dimensions → not collinear',
        'step_collinear_2d_det': '2D determinant = {det}',
        'step_collinear_2d_det_numeric': '2D determinant = {det:.6f}',
        'step_collinear_2d_eq': 'Collinearity condition (2D): {eq}',
        'step_collinear_3d_cross': 'Cross product = {cross}',
        'step_collinear_3d_cross_numeric': 'Cross product = {cross}',
        'step_collinear_3d_eq': 'Set component to zero: {eq}',
        'step_collinear_solve': 'Solve for {param}: {solutions}',
        'step_collinear_result': 'Result: {result}',

        # Orthogonality steps
        'step_orthogonal_dot': 'Dot product = {dot}',
        'step_orthogonal_result': 'Orthogonal? {result}',

        # Coplanarity steps
        'step_coplanar_cross': 'Cross product v2×v3 = {cross}',
        'step_coplanar_cross_numeric': 'Cross product = {cross}',
        'step_coplanar_triple': 'Scalar triple product = {triple}',
        'step_coplanar_triple_numeric': 'Scalar triple product = {triple:.6f}',
        'step_coplanar_result': 'Coplanar? {result}',

        # Basis steps
        'step_basis_det': 'Determinant of matrix formed by vectors = {det}',
        'step_basis_det_numeric': 'Determinant = {det:.6f}',
        'step_basis_result': 'Form a basis? {result}',

        # Decomposition steps
        'step_decompose_init': 'Decompose vector {v} in basis {basis}',
        'step_decompose_matrix': 'Transition matrix (basis columns):\n{matrix}',
        'step_decompose_matrix_numeric': 'Transition matrix:\n{matrix}',
        'step_decompose_system': 'System of equations:\n{system}',
        'step_decompose_solve': 'Solve system',
        'step_decompose_solve_numeric': 'Solve linear system',
        'step_decompose_result': 'Coefficients: {coeffs}',
        'step_decompose_result_numeric': 'Coordinates: {coeffs}',

        # Transition matrix steps
        'step_transition_init': 'Building transition matrix',
        'step_transition_formula': 'P = (new basis)⁻¹ · (old basis)',
        'step_transition_matrix': 'Transition matrix:\n{matrix}',
        'transition_matrix': 'Transition matrix',

        # Eigenvalue steps
        'step_charpoly_init': 'Build characteristic polynomial det(A - λI)',
        'step_charpoly_matrix': 'Matrix A - λI:\n{matrix}',
        'step_charpoly_matrix_numeric': 'Matrix A:\n{matrix}',
        'step_charpoly_det': 'Determinant = {poly}',
        'step_charpoly_coeffs': 'Characteristic polynomial coefficients: {coeffs}',
        'step_eigenvals_init': 'Find eigenvalues',
        'step_eigenvals_charpoly': 'Characteristic polynomial: {poly}',
        'step_eigenvals_charpoly_numeric': 'Coefficients: {coeffs}',
        'step_eigenvals_roots': 'Roots (eigenvalues): {vals}',
        'step_eigenvals_result': 'Eigenvalues: {vals}',
        'step_eigenvecs_init': 'Find eigenvectors',
        'step_eigenvecs_charpoly': 'Characteristic polynomial: {poly}',
        'step_eigenvecs_vals': 'Eigenvalues: {vals}',
        'step_eigenvecs_for_val': 'For λ = {val} (multiplicity {mult}):',
        'step_eigenvecs_vec': '  Vector {idx}: {vec}',
        'step_eigenvecs_matrix': 'Eigenvector matrix (columns):\n{matrix}',
        'step_diag_init': 'Diagonalize matrix',
        'step_diag_eigenvals': 'Eigenvalues (diagonal D):\n{vals}',
        'step_diag_eigenvals_numeric': 'Eigenvalues: {vals}',
        'step_diag_eigenvecs': 'Matrix P (eigenvectors):\n{matrix}',
        'step_diag_eigenvecs_numeric': 'Matrix P:\n{matrix}',
        'step_diag_verify': 'Verification: P·D·P⁻¹ = {product}',
        'step_diag_verify_numeric': 'Verification: P·D·P⁻¹ =\n{product}',

        # Gram-Schmidt steps
        'step_gram_schmidt_init': 'Initial vectors: {vectors}',
        'step_gram_schmidt_start': 'Processing vector {idx}: v = {v}',
        'step_gram_schmidt_subtract': '  Subtract projection onto e{j}: proj = {proj}, u = {u}',
        'step_gram_schmidt_dependent': '  Vector {idx} is linearly dependent, skipping',
        'step_gram_schmidt_orthogonal': '  Orthogonal vector u{idx} = {u}',
        'step_gram_schmidt_normalize': '  Normalize e{idx} = {e}',
        'step_gram_schmidt_result': 'Resulting basis: {basis}',

        # Geometry steps
        'step_points_collinear_init': 'Checking collinearity of points A={A}, B={B}, C={C}',
        'step_points_coplanar_init': 'Checking coplanarity of points A={A}, B={B}, C={C}, D={D}',
        'step_triangle_area_init': 'Computing triangle area with vertices A={A}, B={B}, C={C}',
        'step_triangle_area_vectors': 'Side vectors: AB = {AB}, AC = {AC}',
        'step_triangle_area_cross': 'Cross product AB × AC = {cross}',
        'step_triangle_area_result': 'Area S = 0.5 * |AB × AC| = {area:.6f}',
        'step_tetrahedron_volume_init': 'Computing tetrahedron volume A={A}, B={B}, C={C}, D={D}',
        'step_tetrahedron_volume_vectors': 'Edge vectors: AB = {AB}, AC = {AC}, AD = {AD}',
        'step_tetrahedron_volume_cross': 'Cross product AC × AD = {cross}',
        'step_tetrahedron_volume_triple': 'Scalar triple product AB · (AC × AD) = {triple}',
        'step_tetrahedron_volume_result': 'Volume V = |triple| / 6 = {volume:.6f}',

        # Warnings
        'symbolic_mode_long': 'symbolic',
        'numeric_mode': 'numeric',
        'warn_heavy_symbolic_op': 'Symbolic operation "{op}" on a {size}×{size} matrix may take a long time. Continue?',
        'warn_heavy_operation_detailed': 'Operation "{op}" on a {size}×{size} matrix in {mode} mode may take a very long time (potentially indefinitely). Continue?',

        # Labels
        'v1_label': 'v1:',
        'v2_label': 'v2:',
        'v3_label': 'v3:',
        'v_label': 'v:',
        'b_label': 'b',
        'xyz_format': '(x,y,z)',

        'old_basis': 'Old basis',
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
            try:
                return text.format(**kwargs)
            except KeyError as e:
                return f"{text} [missing key: {e}]"
        return text
