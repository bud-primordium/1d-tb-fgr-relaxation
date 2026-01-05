import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.electron_phonon import g_monatomic
from src.fermi_golden_rule import _g_monatomic_bloch
from src.fermi_golden_rule import build_rate_matrix
from src.lattice import k_grid, q_grid
from src.phonon_1atom import dispersion_monatomic
from src.tb_electron_1band import bloch_state


def test_g_monatomic_bloch_formula_matches_numeric():
    n_cells = 32
    a = 1.0
    alpha = 0.5
    mass = 1.0

    k_vals = k_grid(n_cells=n_cells, a=a)
    q_vals = q_grid(n_cells=n_cells, a=a)

    i = 3
    q_idx = 5
    j = (i + q_idx) % n_cells

    psi_i = bloch_state(n_cells=n_cells, k=float(k_vals[i]), a=a)
    psi_j = bloch_state(n_cells=n_cells, k=float(k_vals[j]), a=a)
    q = float(q_vals[q_idx])

    g_numeric = g_monatomic(
        psi_i=psi_j,
        psi_j=psi_i,
        n_cells=n_cells,
        q=q,
        alpha=alpha,
        a=a,
        mass=mass,
        pbc=True,
    )
    g_analytic = _g_monatomic_bloch(
        k_i=float(k_vals[i]),
        k_f=np.array([float(k_vals[j])], dtype=float),
        q_vals=np.array([q], dtype=float),
        alpha=alpha,
        mass=mass,
        a=a,
    )[0]
    assert np.allclose(g_numeric, g_analytic, rtol=1e-12, atol=1e-12)


def test_rate_matrix_approximately_satisfies_detailed_balance_on_resonant_pairs():
    n_cells = 48
    t0 = 1.0
    k_spring = 1.0
    mass = 1.0
    alpha = 0.5
    kT = 0.4
    sigma = 0.05
    a = 1.0
    delta_mode = "gaussian"

    W, k_vals, e_vals = build_rate_matrix(
        n_cells=n_cells,
        t0=t0,
        k_spring=k_spring,
        mass=mass,
        alpha=alpha,
        kT=kT,
        sigma=sigma,
        a=a,
        delta_mode=delta_mode,
    )

    q_vals = q_grid(n_cells=n_cells, a=a)
    omega_vals = dispersion_monatomic(q_vals=q_vals, k_spring=k_spring, mass=mass)

    atol_log = 0.2
    close = 1.5 * sigma
    far = 5.0 * sigma
    checked = 0
    for i in range(n_cells):
        for j in range(n_cells):
            if i == j:
                continue
            if W[i, j] <= 0.0 or W[j, i] <= 0.0:
                continue
            q_idx = (j - i) % n_cells
            dE = float(e_vals[j] - e_vals[i])
            omega = float(omega_vals[q_idx])
            x_plus = abs(dE + omega)
            x_minus = abs(dE - omega)
            if not (min(x_plus, x_minus) < close and max(x_plus, x_minus) > far):
                continue

            log_ratio = float(np.log(W[i, j] / W[j, i]))
            target = -dE / kT
            assert abs(log_ratio - target) < atol_log
            checked += 1

    assert checked >= 10


def test_transition_rate_has_explicit_1_over_n_cells_normalization():
    t0 = 1.0
    k_spring = 1.0
    mass = 1.0
    alpha = 0.5
    kT = 0.025
    sigma = 2.0
    a = 1.0

    n1 = 20
    n2 = 40
    i1, j1 = 0, n1 // 4
    i2, j2 = 0, n2 // 4

    W1, _, _ = build_rate_matrix(
        n_cells=n1,
        t0=t0,
        k_spring=k_spring,
        mass=mass,
        alpha=alpha,
        kT=kT,
        sigma=sigma,
        a=a,
        delta_mode="gaussian",
    )
    W2, _, _ = build_rate_matrix(
        n_cells=n2,
        t0=t0,
        k_spring=k_spring,
        mass=mass,
        alpha=alpha,
        kT=kT,
        sigma=sigma,
        a=a,
        delta_mode="gaussian",
    )

    scaled_1 = float(W1[i1, j1] * n1)
    scaled_2 = float(W2[i2, j2] * n2)
    assert np.isfinite(scaled_1) and np.isfinite(scaled_2)
    assert np.isclose(scaled_1, scaled_2, rtol=1e-10, atol=0.0)
