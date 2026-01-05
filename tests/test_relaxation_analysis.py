import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.relaxation_analysis import fit_power_law


def test_fit_power_law_recovers_exponent():
    x = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
    A_true = 3.0
    beta_true = -2.0
    y = A_true * x**beta_true
    A, beta, r2 = fit_power_law(x, y)
    assert np.isclose(A, A_true, rtol=1e-12, atol=0.0)
    assert np.isclose(beta, beta_true, rtol=1e-12, atol=0.0)
    assert np.isclose(r2, 1.0, rtol=0.0, atol=1e-12)

