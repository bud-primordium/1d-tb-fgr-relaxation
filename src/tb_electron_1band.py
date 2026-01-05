"""单带 TB 电子模型。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .lattice import pbc_index, phase_factors


def build_hamiltonian(
    n_cells: int,
    t0: float = 1.0,
    onsite: Optional[np.ndarray] = None,
    pbc: bool = True,
    displacements: Optional[np.ndarray] = None,
    alpha: float = 0.0,
) -> np.ndarray:
    """构造单带最近邻 TB 哈密顿量矩阵。"""
    if onsite is None:
        onsite = np.zeros(n_cells, dtype=float)
    if onsite.shape[0] != n_cells:
        raise ValueError("onsite 长度必须等于 n_cells")
    if displacements is not None:
        displacements = np.asarray(displacements, dtype=complex)
        if displacements.shape[0] != n_cells:
            raise ValueError("displacements 长度必须等于 n_cells")

    h = np.zeros((n_cells, n_cells), dtype=complex)
    np.fill_diagonal(h, onsite)

    for n in range(n_cells):
        n_next = n + 1
        if n_next >= n_cells:
            if not pbc:
                continue
            n_next = pbc_index(n_next, n_cells)
        hopping = t0
        if displacements is not None:
            hopping = t0 + alpha * (displacements[n_next] - displacements[n])
        h[n, n_next] += hopping
        h[n_next, n] += np.conj(hopping)

    return h


def diagonalize(h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对角化哈密顿量，返回本征值与本征矢（列向量）。"""
    evals, evecs = np.linalg.eigh(h)
    return evals, evecs


def bloch_state(n_cells: int, k: float, a: float = 1.0) -> np.ndarray:
    """生成规范化的 Bloch 态 |k⟩（单带）。"""
    phases = phase_factors(n_cells, k, a=a)
    return phases / np.sqrt(n_cells)


def dispersion(k: float, t0: float = 1.0, a: float = 1.0) -> float:
    """单带最近邻色散：E(k) = 2 t0 cos(ka)。"""
    return 2.0 * t0 * np.cos(k * a)


def build_square_well(
    n_cells: int,
    center: int,
    width: int,
    depth: float,
) -> np.ndarray:
    """构造方势阱的 on-site 能量数组。"""
    onsite = np.zeros(n_cells, dtype=float)
    half = width // 2
    for offset in range(-half, half + 1):
        idx = pbc_index(center + offset, n_cells)
        onsite[idx] = depth
    return onsite
