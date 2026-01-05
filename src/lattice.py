"""格点与动量网格工具。"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def pbc_index(index: int, n_cells: int) -> int:
    """周期边界条件下的索引映射。"""
    return index % n_cells


def k_grid(n_cells: int, a: float = 1.0) -> np.ndarray:
    """生成离散的 k 网格（第一布里渊区内）。"""
    m_vals = np.arange(n_cells, dtype=int)
    return 2.0 * np.pi * m_vals / (n_cells * a)


def q_grid(n_cells: int, a: float = 1.0) -> np.ndarray:
    """生成离散的 q 网格（与 k 网格一致）。"""
    return k_grid(n_cells, a=a)


def phase_factors(n_cells: int, k: float, a: float = 1.0) -> np.ndarray:
    """返回 e^{i k n a} 的相位因子数组。"""
    n_vals = np.arange(n_cells, dtype=int)
    return np.exp(1j * k * n_vals * a)


def cell_positions(n_cells: int, a: float = 1.0) -> np.ndarray:
    """返回每个原胞的位置坐标。"""
    return np.arange(n_cells, dtype=float) * a


def split_ab_indices(n_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回 A/B 子格点的索引列表（基于 A1,B1,A2,B2 的顺序）。"""
    total = 2 * n_cells
    indices = np.arange(total, dtype=int)
    a_indices = indices[0::2]
    b_indices = indices[1::2]
    return a_indices, b_indices
