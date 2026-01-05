import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.master_equation import generator_from_rates
from src.master_equation import mean_energy
from src.master_equation import relaxation_time_from_energy
from src.master_equation import solve_master_equation
from src.master_equation import stationary_distribution


def test_generator_from_rates_conserves_probability():
    W = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 0.0, 4.0],
            [5.0, 6.0, 0.0],
        ],
        dtype=float,
    )
    Q = generator_from_rates(W)
    assert np.allclose(np.sum(Q, axis=1), 0.0)


def test_stationary_distribution_two_state_matches_analytic():
    a = 2.0
    b = 5.0
    W = np.array([[0.0, a], [b, 0.0]], dtype=float)
    P = stationary_distribution(W)
    expected = np.array([b / (a + b), a / (a + b)], dtype=float)
    assert np.allclose(P, expected, rtol=1e-10, atol=0.0)


def test_solve_master_equation_two_state_matches_analytic():
    a = 2.0
    b = 5.0
    W = np.array([[0.0, a], [b, 0.0]], dtype=float)
    P0 = np.array([1.0, 0.0], dtype=float)
    t_eval = np.array([0.0, 1.0], dtype=float)

    t, P = solve_master_equation(W=W, P0=P0, t_span=(0.0, 1.0), t_eval=t_eval, method="expm")
    assert np.allclose(t, t_eval)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)

    p_star = b / (a + b)
    p_t = p_star + (1.0 - p_star) * np.exp(-(a + b) * 1.0)
    expected = np.array([p_t, 1.0 - p_t], dtype=float)
    assert np.allclose(P[-1], expected, rtol=1e-10, atol=0.0)


def test_mean_energy_and_relaxation_time_helpers():
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    E_mean = np.array([10.0, 6.0, 5.1], dtype=float)
    tau = relaxation_time_from_energy(t=t, E_mean=E_mean, E_target=5.0, threshold=0.02)
    assert tau == 2.0

    P = np.array([[0.25, 0.75], [0.5, 0.5]], dtype=float)
    E = np.array([0.0, 2.0], dtype=float)
    out = mean_energy(P=P, E_grid=E)
    assert np.allclose(out, np.array([1.5, 1.0], dtype=float))

