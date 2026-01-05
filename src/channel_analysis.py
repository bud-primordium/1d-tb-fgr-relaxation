"""通道分析模块：有效出度、参与比、q 模式分布。"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def effective_out_degree(W: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """计算每个态的有效出度（行归一化阈值）。

    定义：p_ij = W_ij / Γ_i, D_eff(i) = #{j | p_ij > ε}

    Args:
        W: 跃迁率矩阵，W[i,j] = W_{i→j}
        epsilon: 阈值，占总出率比例超过此值才计入有效通道

    Returns:
        D_eff: 每个态的有效出度，shape (N,)
    """
    W = np.asarray(W, dtype=float)
    Gamma = W.sum(axis=1, keepdims=True)
    Gamma = np.maximum(Gamma, 1e-20)
    p = W / Gamma
    D_eff = np.sum(p > epsilon, axis=1)
    return D_eff.astype(int)


def participation_ratio(W: np.ndarray) -> np.ndarray:
    """计算参与比（无阈值）。

    定义：D_pr(i) = 1 / Σ_j p_ij²
    - 值域 [1, N]
    - 越接近 1 表示单通道主导
    - 越大表示多通道分散

    Args:
        W: 跃迁率矩阵

    Returns:
        D_pr: 每个态的参与比，shape (N,)
    """
    W = np.asarray(W, dtype=float)
    Gamma = W.sum(axis=1, keepdims=True)
    Gamma = np.maximum(Gamma, 1e-20)
    p = W / Gamma
    ipr = np.sum(p**2, axis=1)
    D_pr = 1.0 / np.maximum(ipr, 1e-20)
    return D_pr


def cumulative_channel_fraction(W: np.ndarray, state_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """计算单个态的累积通道概率曲线。

    Args:
        W: 跃迁率矩阵
        state_idx: 要分析的态索引

    Returns:
        (k_vals, cum_prob): 通道排名 k 和累积概率
    """
    W = np.asarray(W, dtype=float)
    rates = W[state_idx, :]
    total = rates.sum()
    if total < 1e-20:
        return np.array([0]), np.array([0.0])

    p = rates / total
    p_sorted = np.sort(p)[::-1]
    cum_prob = np.cumsum(p_sorted)
    k_vals = np.arange(1, len(p_sorted) + 1)
    return k_vals, cum_prob


def coverage_k90(W: np.ndarray) -> np.ndarray:
    """计算每个态的 k_90 覆盖数。

    定义：最小 k 使得 top-k 通道的累积概率 > 0.9

    Args:
        W: 跃迁率矩阵

    Returns:
        k90: 每个态的 k_90，shape (N,)
    """
    W = np.asarray(W, dtype=float)
    N = W.shape[0]
    k90 = np.zeros(N, dtype=int)

    for i in range(N):
        rates = W[i, :]
        total = rates.sum()
        if total < 1e-20:
            k90[i] = 0
            continue

        p = rates / total
        p_sorted = np.sort(p)[::-1]
        cum = 0.0
        for k, pk in enumerate(p_sorted, start=1):
            cum += pk
            if cum > 0.9:
                k90[i] = k
                break
        else:
            k90[i] = N

    return k90


def q_mode_distribution(W: np.ndarray, k_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算各 q 模式的总跃迁强度。

    q = k_j - k_i (mod 2π)

    Args:
        W: 跃迁率矩阵
        k_grid: 动量网格，shape (N,)

    Returns:
        (q_vals, intensity): q 值和对应的总跃迁强度
    """
    W = np.asarray(W, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    N = len(k_grid)

    q_vals = np.zeros(N)
    intensity = np.zeros(N)

    for dq in range(N):
        q_vals[dq] = k_grid[dq] - k_grid[0]
        if q_vals[dq] > np.pi:
            q_vals[dq] -= 2 * np.pi
        elif q_vals[dq] < -np.pi:
            q_vals[dq] += 2 * np.pi

        for i in range(N):
            j = (i + dq) % N
            intensity[dq] += W[i, j]

    sort_idx = np.argsort(q_vals)
    return q_vals[sort_idx], intensity[sort_idx]


def q_ki_heatmap(W: np.ndarray, k_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 (k_i, q) 热图数据。

    Args:
        W: 跃迁率矩阵
        k_grid: 动量网格

    Returns:
        (k_vals, q_vals, W_kq): k_i 值、q 值、跃迁强度热图
    """
    W = np.asarray(W, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    N = len(k_grid)

    k_vals = k_grid.copy()
    q_vals = np.zeros(N)
    W_kq = np.zeros((N, N))

    for dq in range(N):
        q_vals[dq] = k_grid[dq] - k_grid[0]
        if q_vals[dq] > np.pi:
            q_vals[dq] -= 2 * np.pi
        elif q_vals[dq] < -np.pi:
            q_vals[dq] += 2 * np.pi

    for i in range(N):
        for dq in range(N):
            j = (i + dq) % N
            W_kq[i, dq] = W[i, j]

    sort_idx = np.argsort(q_vals)
    q_vals = q_vals[sort_idx]
    W_kq = W_kq[:, sort_idx]

    return k_vals, q_vals, W_kq
