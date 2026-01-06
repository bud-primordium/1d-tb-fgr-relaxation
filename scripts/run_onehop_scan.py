"""扫描 ω_max，观察“一步跳”(n_hop=1) 何时出现。

目的：
- 定量回答“什么时候可能出现 A→D 一步跳？”
- 在保持电子带宽不变 (t0 固定) 的前提下，通过改变声子 ω_max（等价于改 K/M）
  观察 P(n_hop=1) 的阈值行为。

输出：
- results/onehop_scan_data.npz
"""

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.fermi_golden_rule import build_rate_matrix
from src.kinetic_monte_carlo import run_ensemble


def main():
    params = {
        "t0": 1.0,
        "mass": 1.0,
        "alpha": 0.5,
        "kT": 0.025,
        "sigma": 0.1,
        "n_cells": 80,
        "initial_energy_fraction": 0.9,
        "terminal_energy_fraction": 0.1,
        "n_trajectories": 2000,
        "max_steps": 20000,
        "seed": 123,
    }

    # ω_max = sqrt(2K/M) -> K = (ω_max^2 * M)/2
    omega_max_values = np.array([1.0, 1.4, 2.0, 2.6, 3.2, 3.6, 4.0, 4.5, 5.0], dtype=float)
    k_spring_values = 0.5 * (omega_max_values**2) * float(params["mass"])

    p_onehop = []
    p_twohop = []
    mean_nhop = []

    # 电子能量阈值（与声子参数无关，但依赖离散网格的 initial_state）
    E_init_state = []
    E_terminal = []
    deltaE_to_terminal = []

    for omega_max, k_spring in zip(omega_max_values, k_spring_values):
        W, k_grid, E_grid = build_rate_matrix(
            n_cells=int(params["n_cells"]),
            t0=float(params["t0"]),
            k_spring=float(k_spring),
            mass=float(params["mass"]),
            alpha=float(params["alpha"]),
            kT=float(params["kT"]),
            sigma=float(params["sigma"]),
            a=1.0,
            delta_mode="gaussian",
        )

        E = np.asarray(E_grid, dtype=float)
        Emin, Emax = float(np.min(E)), float(np.max(E))
        E_init_target = Emin + float(params["initial_energy_fraction"]) * (Emax - Emin)
        initial_state = int(np.argmin(np.abs(E - E_init_target)))

        E_term = Emin + float(params["terminal_energy_fraction"]) * (Emax - Emin)

        def terminal_condition(_state: int, energy: float, _time: float) -> bool:
            return energy <= E_term

        trajectories = run_ensemble(
            W=W,
            E_grid=E,
            initial_state=initial_state,
            terminal_condition=terminal_condition,
            n_trajectories=int(params["n_trajectories"]),
            max_steps=int(params["max_steps"]),
            seed=int(params["seed"]),
        )
        n_hops = np.array([tr.n_hops for tr in trajectories], dtype=int)

        p_onehop.append(float(np.mean(n_hops == 1)))
        p_twohop.append(float(np.mean(n_hops == 2)))
        mean_nhop.append(float(np.mean(n_hops)))

        Ei = float(E[initial_state])
        E_init_state.append(Ei)
        E_terminal.append(float(E_term))
        deltaE_to_terminal.append(max(0.0, Ei - float(E_term)))

        print(
            f"ω_max={omega_max:4.1f} (K={k_spring:5.2f})  "
            f"P(n=1)={p_onehop[-1]:.4f}  P(n=2)={p_twohop[-1]:.4f}  <n>={mean_nhop[-1]:.3f}"
        )

    out = repo_root / "results" / "onehop_scan_data.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        omega_max_values=omega_max_values,
        k_spring_values=k_spring_values,
        p_onehop=np.asarray(p_onehop, dtype=float),
        p_twohop=np.asarray(p_twohop, dtype=float),
        mean_nhop=np.asarray(mean_nhop, dtype=float),
        E_init_state=np.asarray(E_init_state, dtype=float),
        E_terminal=np.asarray(E_terminal, dtype=float),
        deltaE_to_terminal=np.asarray(deltaE_to_terminal, dtype=float),
        **params,
    )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

