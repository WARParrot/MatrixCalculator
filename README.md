# Matrix Calculator

A feature‑rich desktop application for linear algebra and analytic geometry, combining **numeric** (NumPy/CuPy) and **symbolic** (SymPy) computation with detailed step‑by‑step solutions.

## Features

### Core Matrix Operations
- Addition, subtraction, multiplication, scalar multiplication, transpose.
- Determinant, rank, inverse, solving linear systems (A·X = B).
- **Three SLAE solvers**: Gauss elimination, Cramer's rule, and inverse matrix method – each with distinct step logging.

### Vector Operations (2D/3D)
- Basic: addition, subtraction, scalar multiplication.
- Products: dot product, cross product, scalar triple product.
- Norm, normalization, projection, angle between vectors.
- **Special relations**: collinearity check (with parameter solving), orthogonality, coplanarity.
- **Basis & decomposition**: validate basis, decompose vector, compute transition matrix.
- **Gram–Schmidt orthogonalization** (normalization optional).

### Geometric Primitives
- Create vectors from two points.
- Check collinearity / coplanarity of three or four points.
- Triangle area (from vertices or side vectors).
- Tetrahedron volume and surface area (from four vertices).

### Eigenvalues & Diagonalization
- Characteristic polynomial (numeric coefficients or symbolic expression).
- Eigenvalues and eigenvectors (numeric or symbolic).
- Diagonalization (P, D matrices) when possible.

### Symbolic Mode (SymPy)
- Toggle between numeric (float) and symbolic (expressions) computation.
- Enter expressions like `2*λ + 1`, `sqrt(3)`, `pi/2` directly into matrix/vector cells.
- Works for all operations: determinant, rank, inverse, SLAE, eigenvalues, Gram–Schmidt, etc.
- **Heavy operation warnings** prevent accidental freezes on large symbolic matrices.

### Step‑by‑Step Logging
Detailed intermediate steps are displayed for most operations:
- Matrix inversion (Gauss‑Jordan with row swaps and elimination).
- Determinant (Gauss elimination or Sarrus/Laplace for small symbolic matrices).
- Rank (row echelon form).
- SLAE solvers (Gauss, Cramer, Inverse).
- Vector products (cross product components, dot product element‑wise).
- Scalar triple product (cross + dot steps).
- Gram–Schmidt (projection subtractions and normalization).
- Eigenvalues / eigenvectors / diagonalization.
- All conditionals (collinearity, orthogonality, coplanarity, basis check).

### Visualization & Export
- **3D vector plot** (Matplotlib) – quiver arrows from origin.
- **LaTeX export** – save the entire step log as a compilable `.tex` file.

### User Interface
- **Interactive matrix/vector widgets** – resizable grids with keyboard navigation (Tab, Enter, Shift+Tab).
- **Notebook tabs** for different functional areas:
  - Matrices
  - Vectors (basic operations)
  - Special Relations (collinearity, orthogonality, coplanarity)
  - Basis & Decomposition
  - Geometry (points, area, volume)
  - Eigenvalues
  - Gram–Schmidt
  - Visualization
- **Bilingual UI** – Russian and English; switch on the fly.
- **File I/O** – load/save matrices and vectors from plain text files.
- **Precision control** – Auto, Float32, Float64.
- **GPU acceleration** – CuPy automatically used if available.
- **Cancellation support** – stop long‑running computations.
- **Memory monitoring** – background warning for high usage.

## Installation
1. **Clone the repository**  
   ```bash
   git clone https://github.com/WARParrot/MatrixCalculator
   cd MatrixCalculator
   ```
2. **Install dependencies**  
   ```bash
   pip install numpy sympy psutil matplotlib
   # optional for GPU:
   pip install cupy-cuda13x   # adjust to your CUDA version
   ```
3. **Run the application**  
   ```bash
   python main.py
   ```

## Usage

### Numeric Mode (default)
- Enter numbers directly. Empty cells are treated as `0`.
- Resize matrices/vectors with the spinboxes and **Resize** button.

### Symbolic Mode
- Check **Symbolic Mode** in the toolbar.
- Enter expressions like `2*alpha + 1`, `sqrt(3)`, `pi/2`.
- Supported symbols: any valid SymPy identifier (e.g., `λ`, `α`, `beta`).
- **Note:** Large symbolic matrices (>2×2) may be slow; a warning will appear.

### Performing Operations
- Select an operation from the dropdown (or navigate to the appropriate tab).
- For scalar multiplication, enter the scalar in the toolbar.
- Click **Compute** – the step‑by‑step log appears in the right panel.
- For vector‑specific operations, use the dedicated tabs (e.g., **Vectors**, **Special Relations**).

### Step Viewer
- Steps are numbered and colour‑coded.
- Intermediate matrix/vector states are shown.
- Final result is highlighted.
- Right‑click to copy, or use **File → Save Result** to export as text.
- **File → Export to LaTeX** saves the log as a compilable `.tex` document.

### Language Switching
Choose **Russian** or **English** from the language combobox in the toolbar. All UI texts and step descriptions update immediately.

## File Format
Matrices and vectors are saved as plain text, one row/vector per line, columns separated by spaces or commas:
```
1 2 3
4 5 6
7 8 9
```
When loading, the widget automatically resizes.

## Dependencies
- Python 3.8+
- [NumPy](https://numpy.org/)
- [SymPy](https://www.sympy.org/) (symbolic computation)
- [Matplotlib](https://matplotlib.org/) (3D visualization)
- [psutil](https://github.com/giampaolo/psutil) (memory monitoring)
- (Optional) [CuPy](https://cupy.dev/) for GPU acceleration

## Project Structure
```
matrix-calculator/
├── main.py          # Application entry point, GUI logic
├── engine.py        # Core operations (numeric + symbolic, step logging)
├── ui/
│   └── widgets.py   # Custom Tkinter widgets (Matrix, Vector, StepViewer, all panels)
├── localization.py  # Translation dictionary and language manager
├── config.py        # Constants, precision options, limits
└── README.md
```

## License
MIT License – feel free to use and modify.
