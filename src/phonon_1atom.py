"""单原子链声子与归一化工具。"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def displacement_from_q(
    q_vals: np.ndarray,
    q_amplitudes: np.ndarray,
    n_cells: int,
    a: float = 1.0,
    mass: float = 1.0,
) -> np.ndarray:
    """由 Q_q 生成实空间位移 u_n（质量加权坐标）。"""
    if q_vals.shape[0] != q_amplitudes.shape[0]:
        raise ValueError("q_vals 与 q_amplitudes 长度不一致")
    n_vals = np.arange(n_cells, dtype=float)
    phases = np.exp(1j * np.outer(q_vals, n_vals * a))
    u = (phases.T @ q_amplitudes) / np.sqrt(n_cells * mass)
    return u


def velocity_from_qdot(
    q_vals: np.ndarray,
    qdot: np.ndarray,
    n_cells: int,
    a: float = 1.0,
    mass: float = 1.0,
) -> np.ndarray:
    """由 Q_q 的时间导数生成实空间速度 u_dot。"""
    if q_vals.shape[0] != qdot.shape[0]:
        raise ValueError("q_vals 与 qdot 长度不一致")
    n_vals = np.arange(n_cells, dtype=float)
    phases = np.exp(1j * np.outer(q_vals, n_vals * a))
    u_dot = (phases.T @ qdot) / np.sqrt(n_cells * mass)
    return u_dot


def kinetic_energy_from_qdot(qdot: np.ndarray) -> float:
    """质量加权坐标下的动能：T = 1/2 sum |Qdot|^2。"""
    return 0.5 * float(np.sum(np.abs(qdot) ** 2))


def kinetic_energy_from_udot(u_dot: np.ndarray, mass: float = 1.0) -> float:
    """由实空间速度计算动能：T = 1/2 * M * sum |u_dot|^2。"""
    return 0.5 * mass * float(np.sum(np.abs(u_dot) ** 2))


def dispersion_monatomic(q_vals: np.ndarray, k_spring: float, mass: float) -> np.ndarray:
    """单原子链声学支色散关系。"""
    return np.sqrt(2.0 * k_spring / mass) * np.abs(np.sin(0.5 * q_vals))
