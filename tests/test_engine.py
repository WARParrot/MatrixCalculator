import pytest
import numpy as np
from models.engine import MatrixEngine

@pytest.fixture
def engine():
    return MatrixEngine()

def test_add_matrices(engine):
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = engine.add_matrices(A, B)
    expected = np.array([[6, 8], [10, 12]])
    np.testing.assert_array_equal(C, expected)

def test_determinant(engine):
    A = [[1, 2], [3, 4]]
    det = engine.determinant_matrix(A)
    assert abs(det - (-2.0)) < 1e-6

def test_rank(engine):
    # Singular matrix
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rank = engine.rank_matrix(A)
    assert rank == 2 
