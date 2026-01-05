"""动力学蒙特卡洛 (KMC) / Gillespie 算法模拟（无量纲时间）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Trajectory:
    """单条 KMC 轨迹记录。"""

    times: np.ndarray
    states: np.ndarray
    energies: np.ndarray

    @property
    def n_hops(self) -> int:
        """跳跃次数。"""
        return int(self.states.shape[0] - 1)

    @property
    def total_time(self) -> float:
        """总弛豫时间。"""
        if self.times.shape[0] == 0:
            return 0.0
        return float(self.times[-1] - self.times[0])


def gillespie_step(
    current_state: int,
    W: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[int, float]:
    """单步 Gillespie 跳跃。

    Args:
        current_state: 当前态索引
        W: 跃迁率矩阵，W[i,j] = W_{i→j}
        rng: 随机数生成器

    Returns:
        (next_state, wait_time): 下一态索引与等待时间 Δt

    Algorithm:
        1) Γ = Σ_j W[current, j]
        2) Δt = -ln(r1) / Γ
        3) 按概率 p_j = W[current, j] / Γ 选择下一态
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W 必须为方阵")
    n = W.shape[0]
    if current_state < 0 or current_state >= n:
        raise ValueError("current_state 越界")

    rates = W[current_state]
    gamma = float(np.sum(rates))
    if gamma <= 0.0:
        raise ValueError("当前态总出率为 0，无法继续 Gillespie 跳跃")

    r1 = float(rng.random())
    r1 = max(r1, np.finfo(float).tiny)
    wait_time = -np.log(r1) / gamma

    r2 = float(rng.random())
    probs = rates / gamma
    cdf = np.cumsum(probs)
    next_state = int(np.searchsorted(cdf, r2, side="right"))
    if next_state >= n:
        next_state = n - 1
    return next_state, float(wait_time)


def run_trajectory(
    W: np.ndarray,
    E_grid: np.ndarray,
    initial_state: int,
    terminal_condition: Callable[[int, float, float], bool],
    max_steps: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Trajectory:
    """运行单条 KMC 轨迹直到满足终止条件。

    Args:
        W: 跃迁率矩阵
        E_grid: 能量网格
        initial_state: 初始态索引
        terminal_condition: 终止条件函数 f(state, energy, time) -> bool
        max_steps: 最大步数（防止无限循环）
        rng: 随机数生成器

    Returns:
        轨迹记录
    """
    W = np.asarray(W, dtype=float)
    E = np.asarray(E_grid, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W 必须为方阵")
    if E.ndim != 1 or E.shape[0] != W.shape[0]:
        raise ValueError("E_grid 长度必须与 W 尺寸一致")
    if max_steps <= 0:
        raise ValueError("max_steps 必须为正整数")

    if rng is None:
        rng = np.random.default_rng()

    t = 0.0
    state = int(initial_state)
    if state < 0 or state >= W.shape[0]:
        raise ValueError("initial_state 越界")

    times = [t]
    states = [state]
    energies = [float(E[state])]

    for _ in range(max_steps):
        if terminal_condition(state, float(E[state]), t):
            break
        try:
            next_state, dt = gillespie_step(state, W=W, rng=rng)
        except ValueError:
            break
        t = t + float(dt)
        state = int(next_state)
        times.append(t)
        states.append(state)
        energies.append(float(E[state]))

    return Trajectory(
        times=np.asarray(times, dtype=float),
        states=np.asarray(states, dtype=int),
        energies=np.asarray(energies, dtype=float),
    )


def run_ensemble(
    W: np.ndarray,
    E_grid: np.ndarray,
    initial_state: int,
    terminal_condition: Callable[[int, float, float], bool],
    n_trajectories: int = 1000,
    max_steps: int = 10000,
    seed: int = 42,
) -> List[Trajectory]:
    """运行多条 KMC 轨迹用于统计。"""
    if n_trajectories <= 0:
        raise ValueError("n_trajectories 必须为正整数")
    rng = np.random.default_rng(int(seed))
    trajectories: List[Trajectory] = []
    for _ in range(int(n_trajectories)):
        trajectories.append(
            run_trajectory(
                W=W,
                E_grid=E_grid,
                initial_state=initial_state,
                terminal_condition=terminal_condition,
                max_steps=max_steps,
                rng=rng,
            )
        )
    return trajectories


def path_statistics(trajectories: List[Trajectory]) -> Dict:
    """统计路径特征。"""
    if len(trajectories) == 0:
        raise ValueError("trajectories 不能为空")

    n_hops = np.array([tr.n_hops for tr in trajectories], dtype=int)
    total_time = np.array([tr.total_time for tr in trajectories], dtype=float)

    max_hops = int(np.max(n_hops))
    hist = np.bincount(n_hops, minlength=max_hops + 1).astype(int)

    step_sizes: List[float] = []
    for tr in trajectories:
        if tr.energies.shape[0] >= 2:
            dE = tr.energies[:-1] - tr.energies[1:]
            step_sizes.extend([float(x) for x in dE])

    return {
        "n_trajectories": int(len(trajectories)),
        "n_hops_mean": float(np.mean(n_hops)),
        "n_hops_std": float(np.std(n_hops)),
        "n_hops_histogram": hist,
        "total_time_mean": float(np.mean(total_time)),
        "total_time_std": float(np.std(total_time)),
        "step_sizes": np.asarray(step_sizes, dtype=float),
    }
