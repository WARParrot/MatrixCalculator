import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import threading
import time
import psutil
import logging
from engine import MatrixEngine
from ui.widgets import MatrixWidget, StepViewer
from localization import Language
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MatrixCalc')

class MatrixCalculatorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(Language.tr('app_title'))
        self.root.geometry("1300x900")

        self.engine = MatrixEngine()
        self.config = config.Config()
        self.cancel_event = threading.Event()

        # Operation configuration: internal id -> (method_name, num_matrices, needs_scalar)
        self.op_config = {
            'add':        ('add_matrices',       2, False),
            'sub':        ('subtract_matrices',  2, False),
            'mul':        ('multiply_matrices',  2, False),
            'scalar_mul': ('scalar_multiply',    1, True),
            'transpose':  ('transpose_matrix',   1, False),
            'det':        ('determinant_matrix', 1, False),
            'rank':       ('rank_matrix',        1, False),
            'inv':        ('inverse_matrix',     1, False),
            'solve':      ('solve_system',       2, False),
        }
        self.operation_ids = list(self.op_config.keys())
        self._build_operations_dict()

        self.operation_var = tk.StringVar()
        self.scalar_var = tk.StringVar(value="1")
        self.precision_var = tk.StringVar(value=config.ComputePrecision.AUTO.value)
        self.status_var = tk.StringVar(value=Language.tr('ready'))

        self._create_menus()
        self._create_toolbar()
        self._create_main_area()
        self._create_status_bar()

        self._start_monitoring()
        # Set initial operation
        first_op = list(self.operations.keys())[0]
        self.operation_var.set(first_op)
        self._on_operation_change()

    def _build_operations_dict(self):
        """Rebuild self.operations from self.op_config using current language."""
        self.operations = {}
        for op_id in self.operation_ids:
            display = Language.tr(f'op_{op_id}')
            self.operations[display] = self.op_config[op_id]

    def _create_menus(self):
        self.menubar = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label=Language.tr('load_a'), command=lambda: self._load_matrix(self.matrix_a))
        self.file_menu.add_command(label=Language.tr('load_b'), command=lambda: self._load_matrix(self.matrix_b))
        self.file_menu.add_command(label=Language.tr('save_a'), command=lambda: self._save_matrix(self.matrix_a))
        self.file_menu.add_command(label=Language.tr('save_b'), command=lambda: self._save_matrix(self.matrix_b))
        self.file_menu.add_command(label=Language.tr('save_result'), command=self._save_result)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=Language.tr('exit'), command=self.root.quit)
        self.menubar.add_cascade(label=Language.tr('file_menu'), menu=self.file_menu)

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.help_menu.add_command(label=Language.tr('about'), command=self._show_about)
        self.menubar.add_cascade(label=Language.tr('help_menu'), menu=self.help_menu)

        self.root.config(menu=self.menubar)

    def _create_toolbar(self):
        toolbar = ttk.Frame(self.root, padding="5")
        toolbar.pack(fill="x")

        # Operation
        self.op_label = ttk.Label(toolbar, text=Language.tr('operation'))
        self.op_label.pack(side="left", padx=5)

        self.op_combo = ttk.Combobox(toolbar, textvariable=self.operation_var,
                                      values=list(self.operations.keys()),
                                      state="readonly", width=20)
        self.op_combo.pack(side="left", padx=5)
        self.op_combo.bind("<<ComboboxSelected>>", self._on_operation_change)

        # Scalar
        self.scalar_label = ttk.Label(toolbar, text=Language.tr('scalar'))
        self.scalar_label.pack(side="left", padx=(20, 5))

        self.scalar_entry = ttk.Entry(toolbar, textvariable=self.scalar_var, width=8)
        self.scalar_entry.pack(side="left", padx=5)

        # Buttons
        self.compute_btn = ttk.Button(toolbar, text=Language.tr('compute'), command=self._compute)
        self.compute_btn.pack(side="left", padx=10)

        self.cancel_btn = ttk.Button(toolbar, text=Language.tr('cancel'), command=self._cancel_compute)
        self.cancel_btn.pack(side="left", padx=2)

        self.swap_btn = ttk.Button(toolbar, text=Language.tr('swap'), command=self._swap_matrices)
        self.swap_btn.pack(side="left", padx=2)

        # Precision
        self.prec_label = ttk.Label(toolbar, text=Language.tr('precision'))
        self.prec_label.pack(side="left", padx=(20, 5))

        self.prec_combo = ttk.Combobox(toolbar, textvariable=self.precision_var,
                                        values=config.Config.PRECISION_OPTIONS,
                                        state="readonly", width=8)
        self.prec_combo.pack(side="left", padx=5)
        self.prec_combo.bind("<<ComboboxSelected>>", self._on_precision_change)

        # Language selector
        self.lang_label = ttk.Label(toolbar, text=Language.tr('language'))
        self.lang_label.pack(side="left", padx=(20, 5))

        self.lang_combo = ttk.Combobox(toolbar, values=['ru', 'en'], state='readonly', width=5)
        self.lang_combo.set(Language.get())
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_language_change)
        self.lang_combo.pack(side="left", padx=5)

        self.toolbar = toolbar

    def _create_main_area(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # Left side: matrices
        self.left_frame = ttk.Frame(main_paned)
        main_paned.add(self.left_frame, weight=1)

        self.matrix_a = MatrixWidget(self.left_frame, Language.tr('matrix_a'), rows=3, cols=3)
        self.matrix_a.pack(fill="both", expand=True, padx=5, pady=5)

        self.slau_separator = ttk.Separator(self.left_frame, orient='vertical')
        # Not packed initially

        self.matrix_b = MatrixWidget(self.left_frame, Language.tr('matrix_b'), rows=3, cols=3)
        self.matrix_b.pack(fill="both", expand=True, padx=5, pady=5)

        # Right side: step viewer
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        self.step_header = ttk.Label(right_frame, text=Language.tr('step_solution'),
                                     font=('Arial', 10, 'bold'))
        self.step_header.pack(anchor='w', padx=5)

        self.step_viewer = StepViewer(right_frame)
        self.step_viewer.pack(fill="both", expand=True, padx=5, pady=5)

    def _create_status_bar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=2)

        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=150)
        self.progress.pack(side="left", padx=5)

        self.progress_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.progress_label.pack(side="left")

    def _on_language_change(self, event=None):
        new_lang = self.lang_combo.get()
        Language.set(new_lang)
        self._refresh_ui_language()

    def _refresh_ui_language(self):
        """Update all UI texts after language change by rebuilding menus and refreshing other elements."""
        # Find current operation ID (if any) to preserve selection
        current_op_display = self.operation_var.get()
        current_op_id = None
        for op_id, (method, num, scalar) in self.op_config.items():
            if Language.tr(f'op_{op_id}') == current_op_display:
                current_op_id = op_id
                break

        # Update window title
        self.root.title(Language.tr('app_title'))

        # --- Rebuild menus ---
        # Delete all existing menu entries
        self.menubar.delete(0, 'end')
        # Recreate menus (this also recreates self.file_menu and self.help_menu)
        self._create_menus()

        # --- Update operation combobox ---
        self._build_operations_dict()
        self.op_combo['values'] = list(self.operations.keys())
        # Restore previous operation if possible
        if current_op_id and Language.tr(f'op_{current_op_id}') in self.operations:
            self.operation_var.set(Language.tr(f'op_{current_op_id}'))
        elif self.operations:
            self.operation_var.set(list(self.operations.keys())[0])
        else:
            self.operation_var.set('')
        self._on_operation_change()  # updates scalar state and SLAE UI

        # --- Update toolbar labels (static labels and buttons) ---
        self.op_label.config(text=Language.tr('operation'))
        self.scalar_label.config(text=Language.tr('scalar'))
        self.compute_btn.config(text=Language.tr('compute'))
        self.cancel_btn.config(text=Language.tr('cancel'))
        self.swap_btn.config(text=Language.tr('swap'))
        self.prec_label.config(text=Language.tr('precision'))
        self.lang_label.config(text=Language.tr('language'))

        # --- Update matrix widgets ---
        self.matrix_a.update_language()
        self.matrix_b.update_language()
        # Update titles (they may have been changed for SLAE)
        self._update_slau_ui()

        # --- Update step viewer header ---
        self.step_header.config(text=Language.tr('step_solution'))

        # --- Update status bar (if in ready state) ---
        current_status = self.status_var.get()
        if current_status not in ('Вычисление...', 'Computing...', 'Отмена...', 'Canceling...'):
            self.status_var.set(Language.tr('ready'))

    def _on_operation_change(self, event=None):
        op_display = self.operation_var.get()
        if op_display not in self.operations:
            return
        _, num_matrices, needs_scalar = self.operations[op_display]

        # Scalar entry state
        self.scalar_entry.config(state='normal' if needs_scalar else 'disabled')

        # Matrix B state
        self.matrix_b.set_state('normal' if num_matrices == 2 else 'disabled')

        # SLAE UI
        self._update_slau_ui()

    def _update_slau_ui(self):
        """Show/hide vertical separator and adjust titles for SLAE."""
        op_display = self.operation_var.get()
        if op_display == Language.tr('op_solve'):
            if not self.slau_separator.winfo_ismapped():
                self.slau_separator.pack(side='left', fill='y', padx=2, pady=5, before=self.matrix_b)
            self.matrix_a.set_title(Language.tr('matrix_a_coeff'))
            self.matrix_b.set_title(Language.tr('matrix_b_rhs'))
        else:
            if self.slau_separator.winfo_ismapped():
                self.slau_separator.pack_forget()
            self.matrix_a.set_title(Language.tr('matrix_a'))
            self.matrix_b.set_title(Language.tr('matrix_b'))

    def _on_precision_change(self, event=None):
        prec_str = self.precision_var.get()
        try:
            prec = config.ComputePrecision(prec_str)
            self.engine.set_precision(prec)
            self.status_var.set(Language.tr('precision_set', prec.value))
        except ValueError:
            pass

    def _swap_matrices(self):
        try:
            data_a = np.array(self.matrix_a.get_matrix_data())
            data_b = np.array(self.matrix_b.get_matrix_data())
            self.matrix_a.set_matrix_data(data_b)
            self.matrix_b.set_matrix_data(data_a)
            self.status_var.set(Language.tr('swap_success'))
        except Exception as e:
            logger.error(f"Swap error: {e}")
            messagebox.showerror(Language.tr('error'), str(e))

    def _load_matrix(self, widget):
        filename = filedialog.askopenfilename(
            title=f"{Language.tr('load_a') if widget is self.matrix_a else Language.tr('load_b')}",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")]
        )
        if filename:
            try:
                widget.load_from_file(filename)
            except Exception as e:
                messagebox.showerror(Language.tr('load_error'), str(e))

    def _save_matrix(self, widget):
        filename = filedialog.asksaveasfilename(
            title=f"{Language.tr('save_a') if widget is self.matrix_a else Language.tr('save_b')}",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if filename:
            try:
                widget.save_to_file(filename)
            except Exception as e:
                messagebox.showerror(Language.tr('save_error'), str(e))

    def _save_result(self):
        text = self.step_viewer.text.get('1.0', tk.END).strip()
        if not text:
            return
        filename = filedialog.asksaveasfilename(
            title=Language.tr('save_result'),
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
            except Exception as e:
                messagebox.showerror(Language.tr('save_error'), str(e))

    def _show_about(self):
        gpu_text = Language.tr('gpu_available') if self.engine.gpu_available else Language.tr('gpu_not_available')
        info = Language.tr('about_text', gpu_text, self.engine.precision.value)
        messagebox.showinfo(Language.tr('about'), info)

    def _cancel_compute(self):
        self.cancel_event.set()
        self.status_var.set(Language.tr('cancel_request'))

    def _compute(self):
        self.step_viewer.clear()
        self.cancel_event.clear()
        self._set_buttons_state('disabled')
        self.progress.start()
        self.status_var.set(Language.tr('computing'))

        thread = threading.Thread(target=self._compute_thread, daemon=True)
        thread.start()

    def _compute_thread(self):
        try:
            op_display = self.operation_var.get()
            if op_display not in self.operations:
                raise ValueError(Language.tr('operation_unknown'))

            method_name, num_matrices, needs_scalar = self.operations[op_display]

            A = np.array(self.matrix_a.get_matrix_data())
            B = np.array(self.matrix_b.get_matrix_data())

            if A.size == 0 and num_matrices >= 1:
                raise ValueError(Language.tr('matrix_empty'))
            if num_matrices == 2 and B.size == 0:
                raise ValueError(Language.tr('matrix_empty'))

            scalar = float(self.scalar_var.get()) if needs_scalar else 1.0

            # Build argument list
            if num_matrices == 1:
                args = [A, scalar] if needs_scalar else [A]
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

            self.root.after(0, lambda: self._display_result(result, steps, op_display))

        except Exception as e:
            logger.exception("Compute error")
            self.root.after(0, lambda: self.step_viewer.add_error(f"{Language.tr('error')}: {str(e)}"))
        finally:
            self.root.after(0, self._compute_finished)

    def _display_result(self, result, steps, op_name):
        self.step_viewer.clear()
        self.step_viewer.add_header(f"{Language.tr('operation')} {op_name}")
        if steps:
            for step in steps:
                self.step_viewer.add_step(step['step'] + 1, step['desc'])
                if 'state' in step and step['state'] is not None:
                    self.step_viewer.add_matrix(step['state'], Language.tr('state'))
        self.step_viewer.add_header(Language.tr('result'))
        if isinstance(result, np.ndarray):
            self.step_viewer.add_matrix(result)
        else:
            self.step_viewer.add_result(str(result))
        self.step_viewer.scroll_to_bottom()

    def _compute_finished(self):
        self.progress.stop()
        self._set_buttons_state('normal')
        self.status_var.set(Language.tr('ready'))

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