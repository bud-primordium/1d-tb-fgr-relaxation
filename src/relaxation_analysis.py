"""弛豫分析工具（无量纲单位）。"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from .fermi_golden_rule import build_rate_matrix, characteristic_step, total_rate
from .kinetic_monte_carlo import motif_statistics, path_statistics, run_ensemble
from .master_equation import (
    mean_energy,
    relaxation_time_from_energy,
    solve_master_equation,
    stationary_distribution,
)


def scaling_experiment(
    N_values: List[int],
    t0: float,
    k_spring: float,
    mass: float,
    alpha: float,
    kT: float,
    sigma: float,
    initial_energy_fraction: float = 0.9,
    n_trajectories: int = 1000,
    terminal_energy_fraction: float = 0.1,
    max_steps: int = 10000,
    seed: int = 42,
    me_threshold: float = 0.01,
) -> Dict:
    """运行完整的标度律验证实验。

    Args:
        N_values: 系统尺寸列表
        t0, k_spring, mass, alpha: 模型参数
        kT: 无量纲温度 k_B T / t₀
        sigma: 能量展宽参数（无量纲，单位 t₀）
        initial_energy_fraction: 初始态在能带中的位置（0→带底，1→带顶）
        n_trajectories: KMC 轨迹数
        terminal_energy_fraction: KMC 终止能量阈值所处带宽比例（0→带底）
        max_steps: 每条 KMC 的最大步数
        seed: 随机种子
        me_threshold: 主方程能量弛豫阈值

    Returns:
        dict，包含每个 N 的关键观测量与路径统计。
    """
    if len(N_values) == 0:
        raise ValueError("N_values 不能为空")
    if not (0.0 <= initial_energy_fraction <= 1.0):
        raise ValueError("initial_energy_fraction 必须在 [0,1] 内")
    if not (0.0 <= terminal_energy_fraction <= 1.0):
        raise ValueError("terminal_energy_fraction 必须在 [0,1] 内")

    N_list: List[int] = []
    gamma_list: List[float] = []
    step_list: List[float] = []
    tau_me_list: List[float] = []
    tau_kmc_list: List[float] = []
    n_hops_list: List[float] = []
    path_stats_list: List[Dict] = []

    for n_cells in N_values:
        W, k_grid, E_grid = build_rate_matrix(
            n_cells=int(n_cells),
            t0=t0,
            k_spring=k_spring,
            mass=mass,
            alpha=alpha,
            kT=kT,
            sigma=sigma,
            a=1.0,
            delta_mode="gaussian",
        )

        E = np.asarray(E_grid, dtype=float)
        E_min = float(np.min(E))
        E_max = float(np.max(E))

        E_init_target = E_min + float(initial_energy_fraction) * (E_max - E_min)
        initial_state = int(np.argmin(np.abs(E - E_init_target)))

        gamma0 = total_rate(W, state_idx=initial_state)
        dE_avg0 = characteristic_step(W, E_grid=E, state_idx=initial_state)

        P_inf = stationary_distribution(W)
        E_eq = float(np.dot(P_inf, E))

        tau_scale = 1.0 / max(gamma0, 1e-12)
        t_end = 50.0 * tau_scale
        t_eval = np.linspace(0.0, t_end, 400, dtype=float)
        P0 = np.zeros_like(E, dtype=float)
        P0[initial_state] = 1.0
        t, P = solve_master_equation(W=W, P0=P0, t_span=(0.0, t_end), t_eval=t_eval, method="RK45")
        E_mean = mean_energy(P=P, E_grid=E)
        tau_me = relaxation_time_from_energy(
            t=t,
            E_mean=E_mean,
            E_target=E_eq,
            threshold=me_threshold,
        )

        E_terminal = E_min + float(terminal_energy_fraction) * (E_max - E_min)

        def terminal_condition(_state: int, energy: float, _time: float) -> bool:
            return energy <= E_terminal

        trajectories = run_ensemble(
            W=W,
            E_grid=E,
            initial_state=initial_state,
            terminal_condition=terminal_condition,
            n_trajectories=n_trajectories,
            max_steps=max_steps,
            seed=seed,
        )
        stats = path_statistics(trajectories)
        motifs = motif_statistics(trajectories, E_min=E_min, E_max=E_max, n_bins=4)

        N_list.append(int(n_cells))
        gamma_list.append(float(gamma0))
        step_list.append(float(dE_avg0))
        tau_me_list.append(float(tau_me))
        tau_kmc_list.append(float(stats["total_time_mean"]))
        n_hops_list.append(float(stats["n_hops_mean"]))
        stats = dict(stats)
        stats.update(
            {
                "initial_state": int(initial_state),
                "E_init_target": float(E_init_target),
                "E_terminal": float(E_terminal),
                "E_eq": float(E_eq),
                "k_grid": np.asarray(k_grid, dtype=float),
                "E_grid": E,
                "motif_n_bins": int(motifs["n_bins"]),
                "motif_bin_edges": np.asarray(motifs["bin_edges"], dtype=float),
                "motif_counts_json": json.dumps(
                    motifs["counts"], ensure_ascii=True, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        path_stats_list.append(stats)

    return {
        "N": np.asarray(N_list, dtype=int),
        "total_rate": np.asarray(gamma_list, dtype=float),
        "char_step": np.asarray(step_list, dtype=float),
        "relax_time_ME": np.asarray(tau_me_list, dtype=float),
        "relax_time_KMC": np.asarray(tau_kmc_list, dtype=float),
        "n_hops_mean": np.asarray(n_hops_list, dtype=float),
        "path_stats": path_stats_list,
    }


def path_stats_experiment(
    N_values: List[int],
    t0: float,
    k_spring: float,
    mass: float,
    alpha: float,
    kT: float,
    sigma: float,
    initial_energy_fraction: float = 0.9,
    n_trajectories: int = 1000,
    terminal_energy_fraction: float = 0.1,
    max_steps: int = 10000,
    seed: int = 42,
) -> Dict:
    """仅计算 KMC 路径统计（用于可视化对照，不解主方程）。

    设计目的：
    - Fig3 需要展示 “低温强跨栏（尖锐）” 与 “更高温（更宽）” 的对照；
    - 若直接重复 scaling_experiment，会额外求解主方程，耗时且与目的无关。

    Returns:
        dict: { "N": ..., "path_stats": List[Dict] }
    """
    if len(N_values) == 0:
        raise ValueError("N_values 不能为空")
    if not (0.0 <= initial_energy_fraction <= 1.0):
        raise ValueError("initial_energy_fraction 必须在 [0,1] 内")
    if not (0.0 <= terminal_energy_fraction <= 1.0):
        raise ValueError("terminal_energy_fraction 必须在 [0,1] 内")

    N_list: List[int] = []
    path_stats_list: List[Dict] = []

    for n_cells in N_values:
        W, k_grid, E_grid = build_rate_matrix(
            n_cells=int(n_cells),
            t0=t0,
            k_spring=k_spring,
            mass=mass,
            alpha=alpha,
            kT=kT,
            sigma=sigma,
            a=1.0,
            delta_mode="gaussian",
        )

        E = np.asarray(E_grid, dtype=float)
        E_min = float(np.min(E))
        E_max = float(np.max(E))

        E_init_target = E_min + float(initial_energy_fraction) * (E_max - E_min)
        initial_state = int(np.argmin(np.abs(E - E_init_target)))
        E_terminal = E_min + float(terminal_energy_fraction) * (E_max - E_min)

        def terminal_condition(_state: int, energy: float, _time: float) -> bool:
            return energy <= E_terminal

        trajectories = run_ensemble(
            W=W,
            E_grid=E,
            initial_state=initial_state,
            terminal_condition=terminal_condition,
            n_trajectories=n_trajectories,
            max_steps=max_steps,
            seed=seed,
        )
        stats = dict(path_statistics(trajectories))
        motifs = motif_statistics(trajectories, E_min=E_min, E_max=E_max, n_bins=4)
        stats.update(
            {
                "initial_state": int(initial_state),
                "E_init_target": float(E_init_target),
                "E_terminal": float(E_terminal),
                "k_grid": np.asarray(k_grid, dtype=float),
                "E_grid": E,
                "motif_n_bins": int(motifs["n_bins"]),
                "motif_bin_edges": np.asarray(motifs["bin_edges"], dtype=float),
                "motif_counts_json": json.dumps(
                    motifs["counts"], ensure_ascii=True, sort_keys=True, separators=(",", ":")
                ),
            }
        )

        N_list.append(int(n_cells))
        path_stats_list.append(stats)

    return {
        "N": np.asarray(N_list, dtype=int),
        "path_stats": path_stats_list,
    }


def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """幂律拟合 y = A * x^β。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x 与 y 必须为等长一维数组")
    if x.shape[0] < 2:
        raise ValueError("至少需要两个数据点")
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("幂律拟合要求 x、y 均为正数")

    lx = np.log(x)
    ly = np.log(y)
    beta, logA = np.polyfit(lx, ly, deg=1)
    A = float(np.exp(logA))

    ly_hat = logA + beta * lx
    ss_res = float(np.sum((ly - ly_hat) ** 2))
    ss_tot = float(np.sum((ly - float(np.mean(ly))) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return A, float(beta), float(R2)
