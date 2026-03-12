import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os
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

        # Control frame
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(fill='x', padx=5, pady=5)

        self.title_label = ttk.Label(self.control_frame, text=f"{title}: ")
        self.title_label.pack(side='left')

        # These labels will be updated dynamically
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

        # Grid frame
        self.grid_frame = ttk.Frame(self)
        self.grid_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self._create_grid()

    def update_language(self):
        """Update all translatable texts in this widget."""
        self.title_label.config(text=f"{self.title_text}: ")
        self.rows_label.config(text=Language.tr('rows'))
        self.cols_label.config(text=Language.tr('cols'))
        self.resize_btn.config(text=Language.tr('resize'))

    def set_title(self, title):
        self.title_text = title
        self.title_label.config(text=f"{title}: ")

    def _create_grid(self):
        """Destroy old grid and create new one according to self.rows, self.cols."""
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
        """Change matrix dimensions, preserving existing data."""
        # Capture current data from widgets
        old_data = {}
        for (r, c), entry in self.widgets.items():
            old_data[(r, c)] = entry.get()

        self.rows = rows
        self.cols = cols
        self.rows_var.set(rows)
        self.cols_var.set(cols)

        self._create_grid()

        # Restore data that fits
        for (r, c), val in old_data.items():
            if r < rows and c < cols:
                self.widgets[(r, c)].insert(0, val)

        if self.widgets:
            self.widgets[(0, 0)].focus_set()

    def get_matrix_data(self):
        """Return a 2D list of floats (reading from entry widgets)."""
        result = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                val_str = self.widgets[(r, c)].get().strip()
                if val_str == "":
                    row.append(0.0)
                else:
                    try:
                        row.append(float(val_str))
                    except ValueError:
                        row.append(float('nan'))
            result.append(row)
        return result

    def set_matrix_data(self, matrix):
        """Fill the widget from a 2D list or numpy array."""
        matrix = np.asarray(matrix)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        rows, cols = matrix.shape
        if rows != self.rows or cols != self.cols:
            self.set_size(rows, cols)

        for r in range(rows):
            for c in range(cols):
                val = matrix[r, c]
                if isinstance(val, float) and val.is_integer():
                    text = str(int(val))
                else:
                    text = str(val)
                self.widgets[(r, c)].delete(0, tk.END)
                self.widgets[(r, c)].insert(0, text)

    def clear(self):
        """Clear all entries."""
        for entry in self.widgets.values():
            entry.delete(0, tk.END)

    def set_state(self, state):
        """Enable or disable all entries."""
        for entry in self.widgets.values():
            entry.config(state=state)

    def save_to_file(self, filename):
        """Save matrix to a text file (space/comma separated)."""
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
        """Load matrix from a text file (space/comma separated)."""
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
    """A scrollable text widget for displaying step‑by‑step solutions."""
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
        """Add a step with localized prefix."""
        prefix = Language.tr('step_prefix', number=number)
        self.text.insert(tk.END, f"\n{prefix} {description}\n", 'step')
        self.scroll_to_bottom()

    def add_matrix(self, matrix, title=""):
        if title:
            self.text.insert(tk.END, f"{title}:\n", 'step')
        if isinstance(matrix, np.ndarray):
            lines = []
            for i, row in enumerate(matrix):
                line = f"  {i+1}: " + " ".join(f"{x:8.4f}" for x in row)
                lines.append(line)
            self.text.insert(tk.END, "\n".join(lines) + "\n", 'matrix')
        else:
            self.text.insert(tk.END, str(matrix) + "\n", 'matrix')

    def add_result(self, text):
        self.text.insert(tk.END, f"\nРезультат:\n{text}\n", 'result')

    def add_error(self, text):
        self.text.insert(tk.END, f"\nОШИБКА: {text}\n", 'error')

    def scroll_to_bottom(self):
        self.text.see(tk.END)