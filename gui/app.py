import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from ui.widgets import MatrixWidget, StepViewer

class MatrixCalculatorApp(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.config(padx=20, pady=20)
        
        # --- Toolbar Section ---
        toolbar = ttk.Frame(self)
        toolbar.pack(fill='x', pady=5)
        
        ttk.Button(toolbar, text="Открыть A...", 
                   command=lambda: self._load_matrix_a()).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Сохранить A...", 
                   command=lambda: self._save_matrix_a()).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="Открыть B...", 
                   command=lambda: self._load_matrix_b()).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Сохранить B...", 
                   command=lambda: self._save_matrix_b()).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="Экспорт результата", 
                   command=self._export_result).pack(side=tk.RIGHT)

        # --- Main Content Section ---
        content_frame = ttk.Frame(self)
        content_frame.pack(fill='both', expand=True)
        
        self.matrix_a_widget = MatrixWidget(content_frame, title="Матрица A")
        self.matrix_a_widget.pack(side=tk.LEFT, padx=10, pady=10, 
                                   fill='y', expand=False)
        
        ttk.Button(self.matrix_a_widget, text="Изменить размер", 
                   command=self._resize_matrix).pack(pady=5)
        
        self.matrix_b_widget = MatrixWidget(content_frame, title="Матрица B")
        self.matrix_b_widget.pack(side=tk.LEFT, padx=10, pady=10, 
                                   fill='y', expand=False)

        # Operation Panel
        panel = ttk.Frame(self)
        panel.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(panel, text="Операция:").pack(side=tk.LEFT)
        self.operation_combo = ttk.Combobox(panel, values=[
            "Сложение", "Вычитание", "Умножение на число", 
            "Перемножение матриц", "Транспонирование", "Определитель",
            "Ранг", "Обращение", "Решение СЛАУ (A * X = B)"])
        self.operation_combo.pack(side=tk.LEFT, padx=5)
        
        self.scalar_entry = ttk.Entry(panel, width=5)
        self.scalar_entry.insert(0, "1")
        self.scalar_entry.pack(side=tk.LEFT, padx=5)
        
        self.calculate_btn = ttk.Button(panel, text="Вычислить", 
                                        command=self._start_calculation)
        self.calculate_btn.pack(side=tk.LEFT, padx=5)

        # --- Result Section ---
        result_frame = ttk.Frame(self)
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        ttk.Label(result_frame, text="Результат:").pack(anchor=tk.W)
        
        self.result_viewer = MatrixResultViewer(result_frame)
        self.step_viewer = StepViewer(result_frame) # NEW: Step viewer
        
        # Pack both viewers side by side (optional layout)
        self.result_viewer.pack(side=tk.TOP, fill='both', expand=True)
        ttk.Label(result_frame, text="Пошаговое решение:", 
                   font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
        self.step_viewer.pack(fill='x', expand=True)

    def _load_matrix_a(self):
        path = filedialog.askopenfilename(filetypes=[("Text/CSV files", "*.txt *.csv")])
        if path:
            self.matrix_a_widget.load_from_file(path)
            arr = self.matrix_a_widget.get_matrix()
            if len(arr.shape) > 0:
                self.matrix_a_widget.set_size(arr.shape[0], arr.shape[1])

    def _save_matrix_a(self):
        path = filedialog.asksaveasfilename(filetypes=[("CSV/Text files", "*.csv *.txt")])
        if path:
            self.matrix_a_widget.save_to_file(path)

    def _load_matrix_b(self):
        path = filedialog.askopenfilename(filetypes=[("Text/CSV files", "*.txt *.csv")])
        if path:
            self.matrix_b_widget.load_from_file(path)
            arr = self.matrix_b_widget.get_matrix()
            if len(arr.shape) > 0:
                self.matrix_b_widget.set_size(arr.shape[0], arr.shape[1])

    def _save_matrix_b(self):
        path = filedialog.asksaveasfilename(filetypes=[("CSV/Text files", "*.csv *.txt")])
        if path:
            self.matrix_b_widget.save_to_file(path)
        
    def _resize_matrix(self):
        messagebox.showinfo("Инфо", "Функция изменения размера скоро появится!")

    def _export_result(self):
        content = self.result_viewer.text_widget.get(1.0, tk.END)
        path = filedialog.asksaveasfilename(filetypes=[("Text files", "*.txt")])
        if path and content:
            try:
                with open(path, 'w') as f:
                    f.write(content)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def _start_calculation(self):
        """Perform selected operation with step-by-step solving for applicable operations."""
        
        operation = self.operation_combo.get()
        
        if operation == "Решение СЛАУ (A * X = B)":
            # Solve system Ax = B with step viewer
            A = self.matrix_a_widget.get_matrix()
            B = self.matrix_b_widget.get_matrix()
            
            if A is None or B is None:
                self.step_viewer.add_error("Введите обе матрицы для решения СЛАУ")
                return
            
            try:
                self._solve_slau_step_by_step(A, B)
            except Exception as e:
                self.step_viewer.add_error(f"Ошибка при решении: {str(e)}")
            
        elif operation in ["Сложение", "Вычитание"]:
            # Regular operation (no steps needed)
            result = self._perform_operation(operation)
            self.result_viewer.display_result(result)
            
        else:
            # Other operations without step viewer
            result = self._perform_operation(operation)
            self.result_viewer.display_result(result)

    def _solve_slau_step_by_step(self, A, B):
        """Solve system Ax = B with detailed steps captured in StepViewer."""
        
        self.step_viewer.clear()
        self.step_viewer.add_header("Решение системы линейных уравнений")
        
        try:
            # Create augmented matrix [A|B]
            n_rows, n_cols_A = A.shape
            n_rows_B = B.shape[0]
            
            if n_rows != n_rows_B:
                raise ValueError("Количество строк в A и B должно совпадать")
            
            n_cols = n_cols_A
            
            # Create augmented matrix [A|B]
            aug_matrix = np.hstack([A, B])
            
            self.step_viewer.add_step(1, 
                                     f"Создание расширенной матрицы [A|B]:\n{self._format_matrix(aug_matrix)}")
            
            # Gaussian elimination with partial pivoting
            for col in range(n_cols):
                if n_rows > col:
                    max_row = np.argmax(np.abs(aug_matrix[col:n_rows, col])) + col
                    
                    if aug_matrix[max_row, col] == 0:
                        raise ValueError("Матрица вырождена или система несовместима")
                    
                    # Swap rows
                    augmented = np.concatenate([aug_matrix[:col], [aug_matrix[max_row, col:], 
                                                                      aug_matrix[col:max_row+1]]], axis=0)
                    self.step_viewer.add_step(col + 1, 
                                             f"Приведение {max_row + 1} строки в ведущую позицию:\n{self._format_matrix(augmented)}")
                    
                    # Eliminate column
                    for row in range(col + 1, n_rows):
                        factor = aug_matrix[row, col] / aug_matrix[col, col]
                        augmented = augmented - np.outer(factor, aug_matrix[col, col:]) \
                                                  * aug_matrix[row, col:]
                        
            # Back substitution (if solving)
            if n_cols_A == n_rows:
                # Compute inverse or solve directly
                try:
                    X = np.linalg.solve(A, B)
                    self.step_viewer.add_step(n_rows + 1, 
                                             f"Решение системы найдено:\n{self._format_matrix(X)}")
                    self.step_viewer.add_result(X)
                except np.linalg.LinAlgError:
                    raise ValueError("Система не имеет единственного решения")
            
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _perform_operation(self, operation):
        """Perform basic matrix operations."""
        A = self.matrix_a_widget.get_matrix()
        B = self.matrix_b_widget.get_matrix()
        
        if operation == "Сложение":
            if A.shape != B.shape:
                raise ValueError("Матрицы должны иметь одинаковые размеры")
            return A + B
            
        elif operation == "Вычитание":
            return A - B
            
        elif operation == "Умножение на число":
            scalar = float(self.scalar_entry.get())
            return scalar * A
            
        elif operation == "Перемножение матриц":
            if A.shape[1] != B.shape[0]:
                raise ValueError("Количество столбцов в A должно совпадать со строками в B")
            return np.dot(A, B)
            
        elif operation == "Транспонирование":
            return A.T
            
        else:
            raise ValueError(f"Неизвестная операция: {operation}")

    def _format_matrix(self, matrix):
        """Format numpy array as readable string."""
        if matrix is None or len(matrix) == 0:
            return "Пустая матрица"
        
        lines = []
        for i, row in enumerate(matrix):
            line = str(i + 1) + ": [" + ", ".join(f"{x:.2f}" for x in row) + "]"
            lines.append(line)
        return "\n".join(lines)

# Helper class for Result Viewer
class MatrixResultViewer(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.text_widget = tk.Text(self, wrap='none', font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, 
                                  command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        self.text_widget.pack(side=tk.LEFT, fill='both', expand=True)
        scrollbar.pack(side=tk.RIGHT, fill='y')
    
    def display_result(self, result):
        """Display a computed result."""
        if hasattr(result, 'shape'):  # Numpy array
            content = f"Результат:\n{self._format_matrix(result)}\n\n"
        else:
            content = f"Результат:\n{result}\n\n"
        
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, content)

    def _format_matrix(self, matrix):
        """Format numpy array as readable string."""
        if matrix is None or len(matrix) == 0:
            return "Пустая матрица"
        
        lines = []
        for i, row in enumerate(matrix):
            line = str(i + 1) + ": [" + ", ".join(f"{x:.2f}" for x in row) + "]"
            lines.append(line)
        return "\n".join(lines)