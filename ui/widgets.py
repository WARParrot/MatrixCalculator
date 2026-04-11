import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import sympy as sp
from localization import Language


class MatrixWidget(ttk.Frame):
    MAX_ROWS = 20
    MAX_COLS = 20

    def __init__(self, master=None, title="Matrix", rows=3, cols=3):
        super().__init__(master)
        self.master = master
        self.title_text = title
        self.rows = rows
        self.cols = cols
        self.entry_width = 8
        self.widgets = {}
        self.active_key = None

        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(fill='x', padx=5, pady=5)

        self.title_label = ttk.Label(self.control_frame, text=f"{title}: ")
        self.title_label.pack(side='left')

        self.rows_label = ttk.Label(self.control_frame, text=Language.tr('rows'))
        self.rows_label.pack(side='left')
        self.rows_var = tk.IntVar(value=rows)
        self.rows_spin = ttk.Spinbox(self.control_frame, from_=1, to=self.MAX_ROWS,
                                      textvariable=self.rows_var, width=5)
        self.rows_spin.pack(side='left', padx=2)

        self.cols_label = ttk.Label(self.control_frame, text=Language.tr('cols'))
        self.cols_label.pack(side='left')
        self.cols_var = tk.IntVar(value=cols)
        self.cols_spin = ttk.Spinbox(self.control_frame, from_=1, to=self.MAX_COLS,
                                      textvariable=self.cols_var, width=5)
        self.cols_spin.pack(side='left', padx=2)

        self.resize_btn = ttk.Button(self.control_frame, text=Language.tr('resize'),
                                      command=self._resize_from_spin)
        self.resize_btn.pack(side='left', padx=5)

        self.grid_frame = ttk.Frame(self)
        self.grid_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self._create_grid()

    def update_language(self):
        self.title_label.config(text=f"{self.title_text}: ")
        self.rows_label.config(text=Language.tr('rows'))
        self.cols_label.config(text=Language.tr('cols'))
        self.resize_btn.config(text=Language.tr('resize'))

    def set_title(self, title):
        self.title_text = title
        self.title_label.config(text=f"{title}: ")

    def _create_grid(self):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.widgets.clear()

        for r in range(self.rows):
            for c in range(self.cols):
                entry = ttk.Entry(self.grid_frame, width=self.entry_width, justify='right')
                self._bind_keys(entry)
                entry.grid(row=r, column=c, padx=2, pady=2, sticky='ew')
                self.widgets[(r, c)] = entry

        for c in range(self.cols):
            self.grid_frame.columnconfigure(c, weight=1)

    def _resize_from_spin(self):
        new_rows = self.rows_var.get()
        new_cols = self.cols_var.get()
        self.set_size(new_rows, new_cols)

    def _bind_keys(self, entry):
        entry.bind('<Return>', self._on_return)
        entry.bind('<Tab>', self._on_tab)
        entry.bind('<Shift-Tab>', self._on_shift_tab)
        entry.bind('<FocusIn>', lambda e, k=entry: self._on_focus_in(entry))

    def _on_focus_in(self, entry):
        for key, w in self.widgets.items():
            if w is entry:
                self.active_key = key
                break

    def _on_return(self, event):
        if self.active_key:
            r, c = self.active_key
            next_r = (r + 1) % self.rows
            next_c = c
            if next_r == 0 and c < self.cols - 1:
                next_c = c + 1
                next_r = 0
            next_key = (next_r, next_c)
            if next_key in self.widgets:
                self.widgets[next_key].focus_set()
        return 'break'

    def _on_tab(self, event):
        if self.active_key:
            r, c = self.active_key
            if c + 1 < self.cols:
                next_key = (r, c + 1)
            elif r + 1 < self.rows:
                next_key = (r + 1, 0)
            else:
                next_key = (0, 0)
            self.widgets[next_key].focus_set()
        return 'break'

    def _on_shift_tab(self, event):
        if self.active_key:
            r, c = self.active_key
            if c - 1 >= 0:
                next_key = (r, c - 1)
            elif r - 1 >= 0:
                next_key = (r - 1, self.cols - 1)
            else:
                next_key = (self.rows - 1, self.cols - 1)
            self.widgets[next_key].focus_set()
        return 'break'

    def set_size(self, rows, cols):
        old_data = {}
        for (r, c), entry in self.widgets.items():
            old_data[(r, c)] = entry.get()

        self.rows = rows
        self.cols = cols
        self.rows_var.set(rows)
        self.cols_var.set(cols)

        self._create_grid()

        for (r, c), val in old_data.items():
            if r < rows and c < cols:
                self.widgets[(r, c)].insert(0, val)

        if self.widgets:
            self.widgets[(0, 0)].focus_set()

    def get_matrix_data(self, symbolic=False):
        result = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                val_str = self.widgets[(r, c)].get().strip()
                if symbolic:
                    row.append(val_str if val_str != "" else "0")
                else:
                    if val_str == "":
                        row.append(0.0)
                    else:
                        try:
                            row.append(float(val_str))
                        except ValueError:
                            row.append(val_str)
            result.append(row)
        return result

    def set_matrix_data(self, matrix, symbolic=False):
        matrix = np.asarray(matrix)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        rows, cols = matrix.shape
        if rows != self.rows or cols != self.cols:
            self.set_size(rows, cols)

        for r in range(rows):
            for c in range(cols):
                val = matrix[r, c]
                if symbolic and isinstance(val, (sp.Expr, sp.Number)):
                    text = str(val)
                else:
                    if isinstance(val, float) and val.is_integer():
                        text = str(int(val))
                    else:
                        text = str(val)
                self.widgets[(r, c)].delete(0, tk.END)
                self.widgets[(r, c)].insert(0, text)

    def clear(self):
        for entry in self.widgets.values():
            entry.delete(0, tk.END)

    def set_state(self, state):
        for entry in self.widgets.values():
            entry.config(state=state)

    def save_to_file(self, filename):
        try:
            with open(filename, 'w') as f:
                for r in range(self.rows):
                    row_vals = []
                    for c in range(self.cols):
                        val_str = self.widgets[(r, c)].get().strip()
                        if val_str == "":
                            row_vals.append("0")
                        else:
                            row_vals.append(val_str)
                    f.write(" ".join(row_vals) + "\n")
            return True
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
            return False

    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                raise ValueError("File is empty")

            data_rows = []
            for line in lines:
                parts = line.replace(',', ' ').split()
                data_rows.append(parts)

            rows = len(data_rows)
            cols = max(len(row) for row in data_rows)
            for row in data_rows:
                while len(row) < cols:
                    row.append("0")

            self.set_size(rows, cols)

            for r in range(rows):
                for c in range(cols):
                    val = data_rows[r][c]
                    self.widgets[(r, c)].delete(0, tk.END)
                    self.widgets[(r, c)].insert(0, val)
            return True
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return False


class StepViewer(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.text = tk.Text(self, wrap='none', font=('Consolas', 10))
        v_scroll = ttk.Scrollbar(self, orient='vertical', command=self.text.yview)
        h_scroll = ttk.Scrollbar(self, orient='horizontal', command=self.text.xview)
        self.text.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.text.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.text.tag_configure('header', foreground='blue', font=('Consolas', 10, 'bold'))
        self.text.tag_configure('step', foreground='black')
        self.text.tag_configure('matrix', foreground='darkgreen', font=('Consolas', 9))
        self.text.tag_configure('result', foreground='red', font=('Consolas', 10, 'bold'))
        self.text.tag_configure('error', foreground='red')

    def clear(self):
        self.text.delete('1.0', tk.END)

    def add_header(self, text):
        self.text.insert(tk.END, text + '\n', 'header')

    def add_step(self, number, description):
        prefix = Language.tr('step_prefix', number=number)
        self.text.insert(tk.END, f"\n{prefix} {description}\n", 'step')
        self.scroll_to_bottom()

    def add_matrix(self, matrix, title=""):
        if title:
            self.text.insert(tk.END, f"{title}:\n", 'step')
        if isinstance(matrix, sp.Matrix):
            lines = []
            for i in range(matrix.rows):
                row = matrix.row(i)
                line = f"  {i+1}: " + " ".join(str(x) for x in row)
                lines.append(line)
            self.text.insert(tk.END, "\n".join(lines) + "\n", 'matrix')
        elif isinstance(matrix, np.ndarray):
            if matrix.ndim == 0:
                val = matrix.item()
                self.text.insert(tk.END, f"  {val:.6g}\n", 'matrix')
            elif matrix.ndim == 1:
                line = " ".join(f"{x:8.4f}" for x in matrix)
                self.text.insert(tk.END, f"  {line}\n", 'matrix')
            else:
                lines = []
                for i, row in enumerate(matrix):
                    line = f"  {i+1}: " + " ".join(f"{x:8.4f}" for x in row)
                    lines.append(line)
                self.text.insert(tk.END, "\n".join(lines) + "\n", 'matrix')
        else:
            self.text.insert(tk.END, str(matrix) + "\n", 'matrix')

    def add_result(self, text):
        self.text.insert(tk.END, f"\n{Language.tr('result')} {text}\n", 'result')

    def add_error(self, text):
        self.text.insert(tk.END, f"\n{Language.tr('error')}: {text}\n", 'error')

    def scroll_to_bottom(self):
        self.text.see(tk.END)


class VectorWidget(ttk.Frame):
    MAX_SIZE = 20

    def __init__(self, master=None, title="Vector", size=3):
        super().__init__(master)
        self.title_text = title
        self.size = size
        self.entry_width = 8
        self.widgets = []
        self.active_index = None

        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(fill='x', padx=5, pady=5)

        self.title_label = ttk.Label(self.control_frame, text=f"{title}: ")
        self.title_label.pack(side='left')

        self.size_label = ttk.Label(self.control_frame, text=Language.tr('size'))
        self.size_label.pack(side='left', padx=(10, 2))
        self.size_var = tk.IntVar(value=size)
        self.size_spin = ttk.Spinbox(self.control_frame, from_=1, to=self.MAX_SIZE,
                                     textvariable=self.size_var, width=5)
        self.size_spin.pack(side='left', padx=2)

        self.resize_btn = ttk.Button(self.control_frame, text=Language.tr('resize'),
                                     command=self._resize_from_spin)
        self.resize_btn.pack(side='left', padx=5)

        self.load_btn = ttk.Button(self.control_frame, text=Language.tr('load'),
                                   command=self._load_vector)
        self.load_btn.pack(side='right', padx=2)
        self.save_btn = ttk.Button(self.control_frame, text=Language.tr('save'),
                                   command=self._save_vector)
        self.save_btn.pack(side='right', padx=2)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar.pack(side='right', fill='y')

        self._create_entries()

    def update_language(self):
        self.title_label.config(text=f"{self.title_text}: ")
        self.size_label.config(text=Language.tr('size'))
        self.resize_btn.config(text=Language.tr('resize'))
        self.load_btn.config(text=Language.tr('load'))
        self.save_btn.config(text=Language.tr('save'))

    def set_title(self, title):
        self.title_text = title
        self.title_label.config(text=f"{title}: ")

    def _create_entries(self):
        for w in self.scrollable_frame.winfo_children():
            w.destroy()
        self.widgets.clear()

        entries_frame = ttk.Frame(self.scrollable_frame)
        entries_frame.pack(fill='both', expand=True)

        for i in range(self.size):
            frame = ttk.Frame(entries_frame)
            frame.pack(fill='x', pady=1)
            lbl = ttk.Label(frame, text=f"{i+1}:", width=3)
            lbl.pack(side='left')
            entry = ttk.Entry(frame, width=self.entry_width, justify='right')
            self._bind_keys(entry, i)
            entry.pack(side='left', fill='x', expand=True)
            self.widgets.append(entry)

    def _bind_keys(self, entry, index):
        entry.bind('<Return>', lambda e, idx=index: self._on_return(idx))
        entry.bind('<Tab>', lambda e, idx=index: self._on_tab(idx))
        entry.bind('<Shift-Tab>', lambda e, idx=index: self._on_shift_tab(idx))
        entry.bind('<FocusIn>', lambda e, idx=index: self._on_focus_in(idx))

    def _on_focus_in(self, index):
        self.active_index = index

    def _on_return(self, index):
        next_idx = (index + 1) % self.size
        self.widgets[next_idx].focus_set()
        return 'break'

    def _on_tab(self, index):
        next_idx = (index + 1) % self.size
        self.widgets[next_idx].focus_set()
        return 'break'

    def _on_shift_tab(self, index):
        next_idx = (index - 1) % self.size
        self.widgets[next_idx].focus_set()
        return 'break'

    def _resize_from_spin(self):
        new_size = self.size_var.get()
        self.set_size(new_size)

    def set_size(self, size):
        old_data = [entry.get() for entry in self.widgets]
        self.size = size
        self.size_var.set(size)
        self._create_entries()
        for i, val in enumerate(old_data):
            if i < size:
                self.widgets[i].insert(0, val)
        if self.widgets:
            self.widgets[0].focus_set()

    def get_vector_data(self, symbolic=False):
        result = []
        for entry in self.widgets:
            val_str = entry.get().strip()
            if symbolic:
                result.append(val_str if val_str != "" else "0")
            else:
                if val_str == "":
                    result.append(0.0)
                else:
                    try:
                        result.append(float(val_str))
                    except ValueError:
                        result.append(val_str)
        if symbolic:
            return result
        return np.array(result, dtype=np.float64)

    def set_vector_data(self, vector, symbolic=False):
        vector = np.asarray(vector).flatten()
        if len(vector) != self.size:
            self.set_size(len(vector))
        for i, val in enumerate(vector):
            if symbolic and isinstance(val, (sp.Expr, sp.Number)):
                text = str(val)
            else:
                if isinstance(val, float) and val.is_integer():
                    text = str(int(val))
                else:
                    text = f"{val:.4g}"
            self.widgets[i].delete(0, tk.END)
            self.widgets[i].insert(0, text)

    def clear(self):
        for entry in self.widgets:
            entry.delete(0, tk.END)

    def set_state(self, state):
        for entry in self.widgets:
            entry.config(state=state)

    def _load_vector(self):
        filename = filedialog.askopenfilename(
            title=Language.tr('load_vector', title=self.title_text),
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    content = f.read().strip()
                parts = content.replace(',', ' ').split()
                data = [float(x) for x in parts]
                self.set_vector_data(data)
            except Exception as e:
                messagebox.showerror(Language.tr('load_error'), str(e))

    def _save_vector(self):
        filename = filedialog.asksaveasfilename(
            title=Language.tr('save_vector', title=self.title_text),
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if filename:
            try:
                data = self.get_vector_data()
                with open(filename, 'w') as f:
                    f.write(" ".join(str(x) for x in data))
            except Exception as e:
                messagebox.showerror(Language.tr('save_error'), str(e))


class VectorOperationsPanel(ttk.Frame):
    def __init__(self, parent, engine, step_viewer):
        super().__init__(parent)
        self.engine = engine
        self.step_viewer = step_viewer

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        top_frame = ttk.Frame(self)
        top_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=1)
        top_frame.grid_rowconfigure(0, weight=1)

        self.vec_a = VectorWidget(top_frame, title=Language.tr('vector_a'), size=3)
        self.vec_a.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        self.vec_b = VectorWidget(top_frame, title=Language.tr('vector_b'), size=3)
        self.vec_b.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        btn_frame = ttk.LabelFrame(self, text=Language.tr('operations'))
        btn_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        for i in range(4):
            btn_frame.columnconfigure(i, weight=1)

        buttons = [
            ('add', self._on_add),
            ('subtract', self._on_subtract),
            ('dot', self._on_dot),
            ('cross', self._on_cross),
            ('norm_a', lambda: self._on_norm(self.vec_a)),
            ('norm_b', lambda: self._on_norm(self.vec_b)),
            ('normalize_a', lambda: self._on_normalize(self.vec_a)),
            ('normalize_b', lambda: self._on_normalize(self.vec_b)),
            ('projection', self._on_projection),
            ('angle', self._on_angle),
            ('triple', self._on_triple),
            ('scalar_mul_a', lambda: self._on_scalar_mul(self.vec_a)),
            ('scalar_mul_b', lambda: self._on_scalar_mul(self.vec_b)),
        ]

        self._create_buttons(btn_frame, buttons)

        self.show_steps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text=Language.tr('show_steps'),
                        variable=self.show_steps_var).grid(row=2, column=0, sticky='w', padx=5)

    def _create_buttons(self, parent, button_specs):
        row, col = 0, 0
        for key, cmd in button_specs:
            text = Language.tr(f'btn_{key}')
            btn = ttk.Button(parent, text=text, command=cmd)
            btn.grid(row=row, column=col, padx=3, pady=3, sticky='ew')
            col += 1
            if col > 3:
                col = 0
                row += 1

    def update_language(self):
        self.vec_a.update_language()
        self.vec_b.update_language()
        btn_frame = self.grid_slaves(row=1, column=0)[0]
        for child in btn_frame.winfo_children():
            child.destroy()
        buttons = [
            ('add', self._on_add),
            ('subtract', self._on_subtract),
            ('dot', self._on_dot),
            ('cross', self._on_cross),
            ('norm_a', lambda: self._on_norm(self.vec_a)),
            ('norm_b', lambda: self._on_norm(self.vec_b)),
            ('normalize_a', lambda: self._on_normalize(self.vec_a)),
            ('normalize_b', lambda: self._on_normalize(self.vec_b)),
            ('projection', self._on_projection),
            ('angle', self._on_angle),
            ('triple', self._on_triple),
            ('scalar_mul_a', lambda: self._on_scalar_mul(self.vec_a)),
            ('scalar_mul_b', lambda: self._on_scalar_mul(self.vec_b)),
        ]
        self._create_buttons(btn_frame, buttons)

    def _show_result(self, result, steps):
        self.step_viewer.clear()
        if steps:
            for step in steps:
                self.step_viewer.add_step(step['step'] + 1, step['desc'])
                state = step.get('state')
                if state is not None:
                    self.step_viewer.add_matrix(state, title=Language.tr('state'))
        if isinstance(result, (np.ndarray, sp.Matrix)):
            if hasattr(result, 'ndim') and result.ndim == 0:
                self.step_viewer.add_result(str(result.item()))
            else:
                self.step_viewer.add_matrix(result, title=Language.tr('result'))
        else:
            self.step_viewer.add_result(str(result))

    def _on_add(self):
        a = self.vec_a.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        b = self.vec_b.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        try:
            res, steps = self.engine.vector_add(a, b, show_steps=self.show_steps_var.get())
            self._show_result(res, steps)
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _on_subtract(self):
        a = self.vec_a.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        b = self.vec_b.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        try:
            res, steps = self.engine.vector_subtract(a, b, show_steps=self.show_steps_var.get())
            self._show_result(res, steps)
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _on_dot(self):
        a = self.vec_a.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        b = self.vec_b.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        try:
            res, steps = self.engine.vector_dot(a, b, show_steps=self.show_steps_var.get())
            self._show_result(res, steps)
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _on_cross(self):
        a = self.vec_a.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        b = self.vec_b.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        try:
            res, steps = self.engine.vector_cross(a, b, show_steps=self.show_steps_var.get())
            self._show_result(res, steps)
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _on_norm(self, vec_widget):
        v = vec_widget.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        try:
            res, steps = self.engine.vector_norm(v, show_steps=self.show_steps_var.get())
            self._show_result(res, steps)
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _on_normalize(self, vec_widget):
        v = vec_widget.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        try:
            res, steps = self.engine.vector_normalize(v, show_steps=self.show_steps_var.get())
            self._show_result(res, steps)
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _on_projection(self):
        a = self.vec_a.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        b = self.vec_b.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        try:
            res, steps = self.engine.vector_projection(a, b, show_steps=self.show_steps_var.get())
            self._show_result(res, steps)
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _on_angle(self):
        a = self.vec_a.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        b = self.vec_b.get_vector_data(symbolic=self.engine.get_symbolic_mode())
        degrees = messagebox.askyesno(Language.tr('angle_unit'), Language.tr('use_degrees'))
        try:
            res, steps = self.engine.vector_angle(a, b, show_steps=self.show_steps_var.get(), degrees=degrees)
            self._show_result(res, steps)
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _on_triple(self):
        from tkinter.simpledialog import askstring
        third_str = askstring(Language.tr('triple_input'), Language.tr('enter_vector_c'))
        if not third_str:
            return
        try:
            a = self.vec_a.get_vector_data(symbolic=self.engine.get_symbolic_mode())
            b = self.vec_b.get_vector_data(symbolic=self.engine.get_symbolic_mode())
            c = [float(x.strip()) for x in third_str.split()] if not self.engine.get_symbolic_mode() else third_str.split()
            res, steps = self.engine.vector_triple_scalar(a, b, c, show_steps=self.show_steps_var.get())
            self._show_result(res, steps)
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _on_scalar_mul(self, vec_widget):
        from tkinter.simpledialog import askstring
        scalar = askstring(Language.tr('scalar_input'), Language.tr('enter_scalar'))
        if scalar is None:
            return
        try:
            v = vec_widget.get_vector_data(symbolic=self.engine.get_symbolic_mode())
            res, steps = self.engine.vector_scalar_multiply(v, scalar, show_steps=self.show_steps_var.get())
            self._show_result(res, steps)
        except Exception as e:
            self.step_viewer.add_error(str(e))


class SpecialRelationsPanel(ttk.Frame):
    """Panel for collinearity, orthogonality, coplanarity checks."""
    def __init__(self, parent, engine, step_viewer):
        super().__init__(parent)
        self.engine = engine
        self.step_viewer = step_viewer

        # Vector input frames
        input_frame = ttk.LabelFrame(self, text=Language.tr('input_vectors'))
        input_frame.pack(fill='x', padx=5, pady=5)

        # Vector 1
        ttk.Label(input_frame, text="v1:").grid(row=0, column=0, padx=5, pady=2)
        self.v1_entry = ttk.Entry(input_frame, width=30)
        self.v1_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(input_frame, text=Language.tr('comma_separated')).grid(row=0, column=2, padx=5)

        # Vector 2
        ttk.Label(input_frame, text="v2:").grid(row=1, column=0, padx=5, pady=2)
        self.v2_entry = ttk.Entry(input_frame, width=30)
        self.v2_entry.grid(row=1, column=1, padx=5, pady=2)

        # Vector 3 (for coplanarity)
        ttk.Label(input_frame, text="v3:").grid(row=2, column=0, padx=5, pady=2)
        self.v3_entry = ttk.Entry(input_frame, width=30)
        self.v3_entry.grid(row=2, column=1, padx=5, pady=2)

        # Parameter for collinearity
        param_frame = ttk.Frame(self)
        param_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(param_frame, text=Language.tr('parameter_name')).pack(side='left', padx=5)
        self.param_var = tk.StringVar(value='λ')
        ttk.Entry(param_frame, textvariable=self.param_var, width=5).pack(side='left')

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(btn_frame, text=Language.tr('btn_collinear_check'),
                   command=self._check_collinear).pack(side='left', padx=2)
        ttk.Button(btn_frame, text=Language.tr('btn_collinear_param'),
                   command=self._find_collinear_param).pack(side='left', padx=2)
        ttk.Button(btn_frame, text=Language.tr('btn_orthogonal'),
                   command=self._check_orthogonal).pack(side='left', padx=2)
        ttk.Button(btn_frame, text=Language.tr('btn_coplanar'),
                   command=self._check_coplanar).pack(side='left', padx=2)

        self.show_steps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text=Language.tr('show_steps'),
                        variable=self.show_steps_var).pack(anchor='w', padx=5)

    def _parse_vector(self, entry_str):
        parts = entry_str.replace(',', ' ').split()
        return [part.strip() for part in parts if part.strip()]

    def _check_collinear(self):
        v1 = self._parse_vector(self.v1_entry.get())
        v2 = self._parse_vector(self.v2_entry.get())
        try:
            res = self.engine.are_collinear(v1, v2)
            self.step_viewer.clear()
            self.step_viewer.add_header(Language.tr('collinearity_check'))
            self.step_viewer.add_result(str(res))
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _find_collinear_param(self):
        v1 = self._parse_vector(self.v1_entry.get())
        v2 = self._parse_vector(self.v2_entry.get())
        param = self.param_var.get()
        try:
            sols, steps = self.engine.find_collinearity_parameter(
                v1, v2, param, show_steps=self.show_steps_var.get())
            self._show_result(sols, steps, Language.tr('collinearity_param'))
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _check_orthogonal(self):
        v1 = self._parse_vector(self.v1_entry.get())
        v2 = self._parse_vector(self.v2_entry.get())
        try:
            res = self.engine.is_orthogonal(v1, v2)
            self.step_viewer.clear()
            self.step_viewer.add_header(Language.tr('orthogonality_check'))
            self.step_viewer.add_result(str(res))
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _check_coplanar(self):
        v1 = self._parse_vector(self.v1_entry.get())
        v2 = self._parse_vector(self.v2_entry.get())
        v3 = self._parse_vector(self.v3_entry.get())
        try:
            res = self.engine.are_coplanar(v1, v2, v3)
            self.step_viewer.clear()
            self.step_viewer.add_header(Language.tr('coplanarity_check'))
            self.step_viewer.add_result(str(res))
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _show_result(self, result, steps, title):
        self.step_viewer.clear()
        self.step_viewer.add_header(title)
        if steps:
            for step in steps:
                self.step_viewer.add_step(step['step'] + 1, step['desc'])
                if step.get('state') is not None:
                    self.step_viewer.add_matrix(step['state'], title=Language.tr('state'))
        self.step_viewer.add_result(str(result))


class BasisPanel(ttk.Frame):
    def __init__(self, parent, engine, step_viewer):
        super().__init__(parent)
        self.engine = engine
        self.step_viewer = step_viewer

        # Vector to decompose
        f1 = ttk.LabelFrame(self, text=Language.tr('vector_to_decompose'))
        f1.pack(fill='x', padx=5, pady=5)
        ttk.Label(f1, text="v:").pack(side='left', padx=5)
        self.v_entry = ttk.Entry(f1, width=30)
        self.v_entry.pack(side='left', padx=5)

        # Basis vectors (3 inputs)
        f2 = ttk.LabelFrame(self, text=Language.tr('basis_vectors'))
        f2.pack(fill='x', padx=5, pady=5)
        self.basis_entries = []
        for i in range(3):
            ttk.Label(f2, text=f"b{i+1}:").grid(row=i, column=0, padx=5, pady=2)
            entry = ttk.Entry(f2, width=30)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.basis_entries.append(entry)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=5, pady=5)
        ttk.Button(btn_frame, text=Language.tr('btn_check_basis'),
                   command=self._check_basis).pack(side='left', padx=2)
        ttk.Button(btn_frame, text=Language.tr('btn_decompose'),
                   command=self._decompose).pack(side='left', padx=2)

        self.show_steps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text=Language.tr('show_steps'),
                        variable=self.show_steps_var).pack(anchor='w', padx=5)

    def _parse_vector(self, entry_str):
        parts = entry_str.replace(',', ' ').split()
        return [part.strip() for part in parts if part.strip()]

    def _check_basis(self):
        basis = [self._parse_vector(e.get()) for e in self.basis_entries]
        try:
            res = self.engine.is_basis(basis)
            self.step_viewer.clear()
            self.step_viewer.add_header(Language.tr('basis_check'))
            self.step_viewer.add_result(str(res))
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _decompose(self):
        v = self._parse_vector(self.v_entry.get())
        basis = [self._parse_vector(e.get()) for e in self.basis_entries]
        try:
            coeffs, steps = self.engine.decompose_vector(
                v, basis, show_steps=self.show_steps_var.get())
            self._show_result(coeffs, steps, Language.tr('decomposition_result'))
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _show_result(self, result, steps, title):
        self.step_viewer.clear()
        self.step_viewer.add_header(title)
        if steps:
            for step in steps:
                self.step_viewer.add_step(step['step'] + 1, step['desc'])
                if step.get('state') is not None:
                    self.step_viewer.add_matrix(step['state'], title=Language.tr('state'))
        if isinstance(result, (list, np.ndarray)):
            self.step_viewer.add_matrix(result, title=Language.tr('coordinates'))
        else:
            self.step_viewer.add_result(str(result))


class GeometryPanel(ttk.Frame):
    def __init__(self, parent, engine, step_viewer):
        super().__init__(parent)
        self.engine = engine
        self.step_viewer = step_viewer

        # Point inputs (4 points for tetrahedron)
        f = ttk.LabelFrame(self, text=Language.tr('points_coordinates'))
        f.pack(fill='x', padx=5, pady=5)
        self.point_entries = []
        for i, label in enumerate(['A', 'B', 'C', 'D']):
            ttk.Label(f, text=f"{label}:").grid(row=i, column=0, padx=5, pady=2)
            entry = ttk.Entry(f, width=30)
            entry.grid(row=i, column=1, padx=5, pady=2)
            ttk.Label(f, text="(x,y,z)").grid(row=i, column=2, padx=5)
            self.point_entries.append(entry)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=5, pady=5)
        ttk.Button(btn_frame, text=Language.tr('btn_collinear_points'),
                   command=self._check_points_collinear).pack(side='left', padx=2)
        ttk.Button(btn_frame, text=Language.tr('btn_coplanar_points'),
                   command=self._check_points_coplanar).pack(side='left', padx=2)
        ttk.Button(btn_frame, text=Language.tr('btn_triangle_area'),
                   command=self._triangle_area).pack(side='left', padx=2)
        ttk.Button(btn_frame, text=Language.tr('btn_tetrahedron_volume'),
                   command=self._tetrahedron_volume).pack(side='left', padx=2)

    def _parse_point(self, entry_str):
        parts = entry_str.replace(',', ' ').split()
        return [float(p) for p in parts[:3]]

    def _check_points_collinear(self):
        A = self._parse_point(self.point_entries[0].get())
        B = self._parse_point(self.point_entries[1].get())
        C = self._parse_point(self.point_entries[2].get())
        res = self.engine.points_collinear(A, B, C)
        self.step_viewer.clear()
        self.step_viewer.add_header(Language.tr('points_collinear'))
        self.step_viewer.add_result(str(res))

    def _check_points_coplanar(self):
        A = self._parse_point(self.point_entries[0].get())
        B = self._parse_point(self.point_entries[1].get())
        C = self._parse_point(self.point_entries[2].get())
        D = self._parse_point(self.point_entries[3].get())
        res = self.engine.points_coplanar(A, B, C, D)
        self.step_viewer.clear()
        self.step_viewer.add_header(Language.tr('points_coplanar'))
        self.step_viewer.add_result(str(res))

    def _triangle_area(self):
        A = self._parse_point(self.point_entries[0].get())
        B = self._parse_point(self.point_entries[1].get())
        C = self._parse_point(self.point_entries[2].get())
        area = self.engine.triangle_area_points(A, B, C)
        self.step_viewer.clear()
        self.step_viewer.add_header(Language.tr('triangle_area'))
        self.step_viewer.add_result(f"{area:.6f}")

    def _tetrahedron_volume(self):
        A = self._parse_point(self.point_entries[0].get())
        B = self._parse_point(self.point_entries[1].get())
        C = self._parse_point(self.point_entries[2].get())
        D = self._parse_point(self.point_entries[3].get())
        vol = self.engine.tetrahedron_volume_points(A, B, C, D)
        self.step_viewer.clear()
        self.step_viewer.add_header(Language.tr('tetrahedron_volume'))
        self.step_viewer.add_result(f"{vol:.6f}")


class EigenPanel(ttk.Frame):
    """Panel for eigenvalues, eigenvectors, and diagonalization."""
    def __init__(self, parent, engine, step_viewer, matrix_widget_getter):
        super().__init__(parent)
        self.engine = engine
        self.step_viewer = step_viewer
        self.get_matrix = matrix_widget_getter  # function that returns current matrix data

        # Matrix input section
        input_frame = ttk.LabelFrame(self, text=Language.tr('input_matrix'))
        input_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Use a MatrixWidget inside for convenience
        self.matrix_widget = MatrixWidget(input_frame, title=Language.tr('matrix_a'), rows=3, cols=3)
        self.matrix_widget.pack(fill='both', expand=True, padx=5, pady=5)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(btn_frame, text=Language.tr('btn_charpoly'),
                   command=self._compute_charpoly).pack(side='left', padx=2)
        ttk.Button(btn_frame, text=Language.tr('btn_eigenvalues'),
                   command=self._compute_eigenvalues).pack(side='left', padx=2)
        ttk.Button(btn_frame, text=Language.tr('btn_eigenvectors'),
                   command=self._compute_eigenvectors).pack(side='left', padx=2)
        ttk.Button(btn_frame, text=Language.tr('btn_diagonalize'),
                   command=self._diagonalize).pack(side='left', padx=2)

        # Copy from main matrix A button
        ttk.Button(btn_frame, text=Language.tr('btn_copy_from_a'),
                   command=self._copy_from_a).pack(side='left', padx=10)

        self.show_steps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text=Language.tr('show_steps'),
                        variable=self.show_steps_var).pack(anchor='w', padx=5)

    def _get_matrix_data(self):
        return self.matrix_widget.get_matrix_data(symbolic=self.engine.get_symbolic_mode())

    def _copy_from_a(self):
        matrix = self.get_matrix()
        if matrix is not None:
            self.matrix_widget.set_matrix_data(matrix, symbolic=self.engine.get_symbolic_mode())

    def _compute_charpoly(self):
        A = self._get_matrix_data()
        try:
            poly, steps = self.engine.characteristic_polynomial(
                A, show_steps=self.show_steps_var.get())
            self._show_result(poly, steps, Language.tr('charpoly_result'))
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _compute_eigenvalues(self):
        A = self._get_matrix_data()
        try:
            vals, steps = self.engine.eigenvalues(A, show_steps=self.show_steps_var.get())
            self._show_result(vals, steps, Language.tr('eigenvalues_result'))
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _compute_eigenvectors(self):
        A = self._get_matrix_data()
        try:
            vecs, steps = self.engine.eigenvectors(A, show_steps=self.show_steps_var.get())
            self._show_result(vecs, steps, Language.tr('eigenvectors_result'))
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _diagonalize(self):
        A = self._get_matrix_data()
        try:
            (P, D), steps = self.engine.diagonalize(A, show_steps=self.show_steps_var.get())
            self._show_result((P, D), steps, Language.tr('diagonalization_result'))
        except Exception as e:
            self.step_viewer.add_error(str(e))

    def _show_result(self, result, steps, title):
        self.step_viewer.clear()
        self.step_viewer.add_header(title)
        if steps:
            for step in steps:
                self.step_viewer.add_step(step['step'] + 1, step['desc'])
                if step.get('state') is not None:
                    self.step_viewer.add_matrix(step['state'], title=Language.tr('state'))
        if isinstance(result, tuple) and len(result) == 2:  # (P, D)
            P, D = result
            self.step_viewer.add_matrix(P, title="P (eigenvectors)")
            self.step_viewer.add_matrix(D, title="D (eigenvalues)")
        elif isinstance(result, (list, np.ndarray, sp.Matrix)):
            self.step_viewer.add_matrix(result)
        else:
            self.step_viewer.add_result(str(result))
