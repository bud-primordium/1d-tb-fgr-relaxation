import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kinetic_monte_carlo import gillespie_step, run_trajectory


def test_gillespie_step_is_deterministic_with_fixed_rng():
    W = np.array(
        [
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(0)
    next_state, dt = gillespie_step(current_state=0, W=W, rng=rng)
    assert next_state == 2
    assert np.isclose(dt, 0.11276144267775219, rtol=0.0, atol=1e-15)


def test_run_trajectory_stops_on_terminal_condition():
    W = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    E = np.array([0.0, 1.0], dtype=float)
    rng = np.random.default_rng(0)

    traj = run_trajectory(
        W=W,
        E_grid=E,
        initial_state=1,
        terminal_condition=lambda state, energy, time: state == 0,
        max_steps=10,
        rng=rng,
    )

    assert traj.states.tolist() == [1, 0]
    assert traj.n_hops == 1
    assert traj.energies.tolist() == [1.0, 0.0]


def test_run_trajectory_no_hops_if_already_terminal():
    W = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    E = np.array([0.0, 1.0], dtype=float)
    traj = run_trajectory(
        W=W,
        E_grid=E,
        initial_state=0,
        terminal_condition=lambda state, energy, time: state == 0,
        max_steps=10,
        rng=np.random.default_rng(0),
    )
    assert traj.states.tolist() == [0]
    assert traj.n_hops == 0

