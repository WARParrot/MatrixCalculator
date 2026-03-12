# Matrix Calculator

A feature‑rich desktop application for performing matrix operations with a clean bilingual interface and step‑by‑step solutions.

## Features
- **Interactive matrix widgets** – Resizable entry grids (up to 20×20) with keyboard navigation.
- **Basic operations** – Addition, subtraction, multiplication, scalar multiplication, transpose.
- **Advanced operations** – Determinant, rank, inverse, solving linear systems (A·X = B).
- **Step‑by‑step solving** – For determinant, rank, inverse, and SLAE. Displays each pivot, row swap, elimination, and back‑substitution.
- **Bilingual UI** – Switch between **Russian** and **English** on the fly; all texts update immediately.
- **File I/O** – Load/save matrices from/to plain text files (space/comma separated).
- **GPU acceleration** – Uses CuPy automatically if installed for fast computation.
- **Cancellation support** – Stop long‑running operations.
- **Memory monitoring** – Background warning for high memory usage.

## Installation
1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/matrix-calculator.git
   cd matrix-calculator
   ```
2. **Install dependencies**  
   ```bash
   pip install numpy psutil
   # optional for GPU:
   pip install cupy-cuda11x   # adjust to your CUDA version
   ```
3. **Run the application**  
   ```bash
   python main.py
   ```

## Usage
- Enter numbers directly into the matrices. Empty cells are treated as `0`.
- Use the **Rows/Cols** spinboxes and **Resize** button to change dimensions (existing data is preserved).
- Select an operation from the dropdown.
- For operations that require a scalar, enter it in the **Scalar** field.
- Click **Compute** – step‑by‑step log appears in the right panel.
- To swap matrices A and B, click **⇄ Swap A↔B**.
- Load/save matrices via the **File** menu.

### Language Switching
Choose **Russian** or **English** from the language combobox in the toolbar. The entire interface, including step descriptions, updates immediately.

## Step‑by‑Step Output
For supported operations, the viewer shows:
- The initial matrix (or augmented matrix for SLAE).
- Each row swap (with 1‑based indices).
- Each elimination step, including the factor used.
- Intermediate matrix states.
- The final result.

## File Format
Matrices are saved as plain text, one row per line, columns separated by spaces or commas:
```
1 2 3
4 5 6
7 8 9
```
When loading, the widget automatically resizes to match the file.

## Dependencies
- Python 3.8+
- [NumPy](https://numpy.org/)
- [psutil](https://github.com/giampaolo/psutil) (for CPU thread config and memory monitoring)
- (Optional) [CuPy](https://cupy.dev/) for GPU acceleration

## Project Structure
```
matrix-calculator/
├── main.py          # Application entry point, GUI logic
├── engine.py        # Core matrix operations (fast + step logging)
├── widgets.py       # Custom Tkinter widgets (MatrixWidget, StepViewer)
├── localization.py  # Translation dictionary and language manager
├── config.py        # Constants and configuration
├── utils.py         # Helper functions
└── app.py           # Alternative entry point
```

## License
MIT License – feel free to use and modify.
