import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import threading
import time
import psutil
import logging
from engine import MatrixEngine
from ui.widgets import MatrixWidget, StepViewer
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MatrixCalc')

class MatrixCalculatorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Матричный калькулятор")
        self.root.geometry("1300x900")

        self.engine = MatrixEngine()
        self.config = config.Config()
        self.cancel_event = threading.Event()

        # Operation definitions: (num_matrices, needs_scalar, method_name)
        self.operations = {
            "Сложение":                (2, False, "add_matrices"),
            "Вычитание":               (2, False, "subtract_matrices"),
            "Умножение":               (2, False, "multiply_matrices"),
            "Умножение на число":      (1, True,  "scalar_multiply"),
            "Транспонирование A":      (1, False, "transpose_matrix"),
            "Определитель A":          (1, False, "determinant_matrix"),
            "Ранг A":                  (1, False, "rank_matrix"),
            "Обращение A":             (1, False, "inverse_matrix"),
            "Решение СЛАУ (A*X = B)": (2, False, "solve_system"),
        }

        self.operation_var = tk.StringVar(value="Сложение")
        self.scalar_var = tk.StringVar(value="1")
        self.precision_var = tk.StringVar(value=config.ComputePrecision.AUTO.value)
        self.status_var = tk.StringVar(value="Готов")

        self._create_menus()
        self._create_toolbar()
        self._create_main_area()
        self._create_status_bar()

        self._start_monitoring()

    def _create_menus(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Загрузить A...", command=lambda: self._load_matrix(self.matrix_a))
        file_menu.add_command(label="Загрузить B...", command=lambda: self._load_matrix(self.matrix_b))
        file_menu.add_command(label="Сохранить A...", command=lambda: self._save_matrix(self.matrix_a))
        file_menu.add_command(label="Сохранить B...", command=lambda: self._save_matrix(self.matrix_b))
        file_menu.add_command(label="Сохранить результат", command=self._save_result)
        file_menu.add_separator()
        file_menu.add_command(label="Выйти", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="О программе", command=self._show_about)
        menubar.add_cascade(label="Справка", menu=help_menu)

        self.root.config(menu=menubar)

    def _create_toolbar(self):
        toolbar = ttk.Frame(self.root, padding="5")
        toolbar.pack(fill="x")

        ttk.Label(toolbar, text="Операция:").pack(side="left", padx=5)
        op_combo = ttk.Combobox(toolbar, textvariable=self.operation_var,
                                 values=list(self.operations.keys()),
                                 state="readonly", width=20)
        op_combo.pack(side="left", padx=5)
        op_combo.bind("<<ComboboxSelected>>", self._on_operation_change)

        ttk.Label(toolbar, text="Скаляр:").pack(side="left", padx=(20, 5))
        self.scalar_entry = ttk.Entry(toolbar, textvariable=self.scalar_var, width=8)
        self.scalar_entry.pack(side="left", padx=5)

        ttk.Button(toolbar, text="Вычислить", command=self._compute).pack(side="left", padx=10)
        ttk.Button(toolbar, text="Отмена", command=self._cancel_compute).pack(side="left", padx=2)
        ttk.Button(toolbar, text="⇄ Поменять A↔B", command=self._swap_matrices).pack(side="left", padx=2)

        ttk.Label(toolbar, text="Точность:").pack(side="left", padx=(20, 5))
        prec_combo = ttk.Combobox(toolbar, textvariable=self.precision_var,
                                   values=config.Config.PRECISION_OPTIONS,
                                   state="readonly", width=8)
        prec_combo.pack(side="left", padx=5)
        prec_combo.bind("<<ComboboxSelected>>", self._on_precision_change)

        self.toolbar = toolbar

    def _create_main_area(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # Left side: matrices with optional separator for SLAE
        self.left_frame = ttk.Frame(main_paned)
        main_paned.add(self.left_frame, weight=1)

        self.matrix_a = MatrixWidget(self.left_frame, "Матрица A", rows=3, cols=3)
        self.matrix_a.pack(fill="both", expand=True, padx=5, pady=5)

        self.slau_separator = ttk.Separator(self.left_frame, orient='vertical')
        # Not packed initially

        self.matrix_b = MatrixWidget(self.left_frame, "Матрица B", rows=3, cols=3)
        self.matrix_b.pack(fill="both", expand=True, padx=5, pady=5)

        # Right side: step viewer
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        ttk.Label(right_frame, text="Пошаговое решение / Результат",
                  font=('Arial', 10, 'bold')).pack(anchor='w', padx=5)
        self.step_viewer = StepViewer(right_frame)
        self.step_viewer.pack(fill="both", expand=True, padx=5, pady=5)

    def _create_status_bar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=2)

        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=150)
        self.progress.pack(side="left", padx=5)
        self.progress_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.progress_label.pack(side="left")

    def _on_operation_change(self, event=None):
        op_name = self.operation_var.get()
        if op_name not in self.operations:
            return
        num_matrices, needs_scalar, _ = self.operations[op_name]
        # Enable/disable scalar entry
        self.scalar_entry.config(state='normal' if needs_scalar else 'disabled')
        # Enable/disable matrix B inputs
        self.matrix_b.set_state('normal' if num_matrices == 2 else 'disabled')
        # Update UI for SLAE
        self._update_slau_ui()

    def _update_slau_ui(self):
        """Show/hide vertical separator and adjust titles for SLAE."""
        if self.operation_var.get() == "Решение СЛАУ (A*X = B)":
            if not self.slau_separator.winfo_ismapped():
                self.slau_separator.pack(side='left', fill='y', padx=2, pady=5, before=self.matrix_b)
            self.matrix_a.set_title("A (коэффициенты)")
            self.matrix_b.set_title("B (свободные члены)")
        else:
            if self.slau_separator.winfo_ismapped():
                self.slau_separator.pack_forget()
            self.matrix_a.set_title("Матрица A")
            self.matrix_b.set_title("Матрица B")

    def _on_precision_change(self, event=None):
        prec_str = self.precision_var.get()
        try:
            prec = config.ComputePrecision(prec_str)
            self.engine.set_precision(prec)
            self.status_var.set(f"Точность: {prec.value}")
        except ValueError:
            pass

    def _swap_matrices(self):
        try:
            data_a = np.array(self.matrix_a.get_matrix_data())
            data_b = np.array(self.matrix_b.get_matrix_data())
            self.matrix_a.set_matrix_data(data_b)
            self.matrix_b.set_matrix_data(data_a)
            self.status_var.set("Матрицы обменяны")
        except Exception as e:
            logger.error(f"Swap error: {e}")
            messagebox.showerror("Ошибка", str(e))

    def _load_matrix(self, widget):
        filename = filedialog.askopenfilename(
            title=f"Загрузить {widget.title_text}",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")]
        )
        if filename:
            widget.load_from_file(filename)

    def _save_matrix(self, widget):
        filename = filedialog.asksaveasfilename(
            title=f"Сохранить {widget.title_text}",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if filename:
            widget.save_to_file(filename)

    def _save_result(self):
        text = self.step_viewer.text.get('1.0', tk.END).strip()
        if not text:
            return
        filename = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def _show_about(self):
        info = (f"Матричный калькулятор\n"
                f"Использует NumPy{' и CuPy' if self.engine.gpu_available else ''}\n"
                f"Текущая точность: {self.engine.precision.value}")
        messagebox.showinfo("О программе", info)

    def _cancel_compute(self):
        self.cancel_event.set()
        self.status_var.set("Отмена...")

    def _compute(self):
        self.step_viewer.clear()
        self.cancel_event.clear()
        self._set_buttons_state('disabled')
        self.progress.start()
        self.status_var.set("Вычисление...")

        thread = threading.Thread(target=self._compute_thread, daemon=True)
        thread.start()

    def _compute_thread(self):
        try:
            op_name = self.operation_var.get()
            if op_name not in self.operations:
                raise ValueError("Неизвестная операция")

            num_matrices, needs_scalar, method_name = self.operations[op_name]

            A = np.array(self.matrix_a.get_matrix_data())
            B = np.array(self.matrix_b.get_matrix_data())

            if A.size == 0 and num_matrices >= 1:
                raise ValueError("Матрица A пуста")
            if num_matrices == 2 and B.size == 0:
                raise ValueError("Матрица B пуста")

            scalar = float(self.scalar_var.get()) if needs_scalar else 1.0

            # Build argument list correctly
            if num_matrices == 1:
                if needs_scalar:
                    args = [A, scalar]
                else:
                    args = [A]
            else:  # num_matrices == 2
                args = [A, B]
                if needs_scalar:
                    args.append(scalar)

            method = getattr(self.engine, method_name)
            show_steps = method_name in ("inverse_matrix", "determinant_matrix", "solve_system", "rank_matrix")
            if show_steps:
                result, steps = method(*args, show_steps=True)
            else:
                result, steps = method(*args)

            self.root.after(0, lambda: self._display_result(result, steps, op_name))

        except Exception as e:
            logger.exception("Compute error")
            self.root.after(0, lambda: self.step_viewer.add_error(str(e)))
        finally:
            self.root.after(0, self._compute_finished)

    def _display_result(self, result, steps, op_name):
        self.step_viewer.clear()
        self.step_viewer.add_header(f"Операция: {op_name}")
        if steps:
            for step in steps:
                self.step_viewer.add_step(step['step'], step['desc'])
                if 'state' in step:
                    self.step_viewer.add_matrix(step['state'], "Состояние")
        self.step_viewer.add_header("Результат")
        if isinstance(result, np.ndarray):
            self.step_viewer.add_matrix(result)
        else:
            self.step_viewer.add_result(str(result))
        self.step_viewer.scroll_to_bottom()

    def _compute_finished(self):
        self.progress.stop()
        self._set_buttons_state('normal')
        self.status_var.set("Готов")

    def _set_buttons_state(self, state):
        for child in self.toolbar.winfo_children():
            if isinstance(child, ttk.Button):
                try:
                    child.config(state=state)
                except:
                    pass

    def _start_monitoring(self):
        def monitor():
            while True:
                try:
                    mem = psutil.Process().memory_info().rss / 1024 / 1024
                    if mem > 1500:
                        logger.warning(f"High memory usage: {mem:.2f} MB")
                except:
                    pass
                time.sleep(5)
        threading.Thread(target=monitor, daemon=True).start()


if __name__ == "__main__":
    app = MatrixCalculatorApp()
    app.root.mainloop()