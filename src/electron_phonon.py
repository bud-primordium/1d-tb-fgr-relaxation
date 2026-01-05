"""电子-声子耦合矩阵元计算。"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .lattice import pbc_index


def dh_dq_monatomic(
    n_cells: int,
    q: float,
    alpha: float,
    a: float = 1.0,
    mass: float = 1.0,
    pbc: bool = True,
) -> np.ndarray:
    """构造单原子链 SSH 型耦合的 dH/dQ_q。

    约定：这里的 Q_q 不包含超胞归一化因子 1/sqrt(n_cells)，该因子在最终
    ⟨|d|^2⟩ 中以 1/n_cells 显式出现（与讨论总结口径一致）。
    """
    dh = np.zeros((n_cells, n_cells), dtype=complex)
    prefactor = alpha / np.sqrt(mass)

    for n in range(n_cells):
        n_next = n + 1
        if n_next >= n_cells:
            if not pbc:
                continue
            n_next = pbc_index(n_next, n_cells)
        phase_n = np.exp(1j * q * n * a)
        phase_next = np.exp(1j * q * (n + 1) * a)
        dt = prefactor * (phase_next - phase_n)
        dh[n, n_next] += dt
        dh[n_next, n] += dt

    return dh


def g_monatomic(
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    n_cells: int,
    q: float,
    alpha: float,
    a: float = 1.0,
    mass: float = 1.0,
    pbc: bool = True,
) -> complex:
    """计算单原子链的 g_{ij}(q)。"""
    dh = dh_dq_monatomic(
        n_cells=n_cells,
        q=q,
        alpha=alpha,
        a=a,
        mass=mass,
        pbc=pbc,
    )
    return np.vdot(psi_i, dh @ psi_j)


def g_monatomic_grid(
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    n_cells: int,
    q_vals: np.ndarray,
    alpha: float,
    a: float = 1.0,
    mass: float = 1.0,
    pbc: bool = True,
) -> np.ndarray:
    """计算单原子链的 g_{ij}(q) 分布。"""
    g_vals = np.zeros_like(q_vals, dtype=complex)
    for idx, q in enumerate(q_vals):
        g_vals[idx] = g_monatomic(
            psi_i=psi_i,
            psi_j=psi_j,
            n_cells=n_cells,
            q=q,
            alpha=alpha,
            a=a,
            mass=mass,
            pbc=pbc,
        )
    return g_vals


def dh_dq_ssh_diatomic(
    n_cells: int,
    q: float,
    evec: np.ndarray,
    alpha: float,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
    beta: float = 0.0,
    pbc: bool = True,
) -> np.ndarray:
    """构造 SSH 两带模型的 dH/dQ_{q,nu}（双原子链）。

    约定：这里的 Q_{q,nu} 不包含超胞归一化因子 1/sqrt(n_cells)，该因子在最终
    ⟨|d|^2⟩ 中以 1/n_cells 显式出现（与讨论总结口径一致）。
    """
    size = 2 * n_cells
    dh = np.zeros((size, size), dtype=complex)
    prefactor_a = 1.0 / np.sqrt(mass_a)
    prefactor_b = 1.0 / np.sqrt(mass_b)

    for n in range(n_cells):
        a_idx = 2 * n
        b_idx = 2 * n + 1
        n_next = n + 1
        if n_next >= n_cells:
            if not pbc:
                continue
            n_next = pbc_index(n_next, n_cells)
        a_next_idx = 2 * n_next

        phase_n = np.exp(1j * q * n * a)
        phase_next = np.exp(1j * q * (n + 1) * a)

        du_a = phase_n * evec[0] * prefactor_a
        du_b = phase_n * evec[1] * prefactor_b
        du_a_next = phase_next * evec[0] * prefactor_a

        delta_intra = du_b - du_a
        delta_inter = du_a_next - du_b

        dh[a_idx, b_idx] += alpha * delta_intra
        dh[b_idx, a_idx] += alpha * delta_intra
        dh[b_idx, a_next_idx] += alpha * delta_inter
        dh[a_next_idx, b_idx] += alpha * delta_inter

        if beta != 0.0:
            dh[a_idx, a_idx] += beta * du_a
            dh[b_idx, b_idx] += beta * du_b

    return dh


def g_ssh_diatomic(
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    n_cells: int,
    q: float,
    evec: np.ndarray,
    alpha: float,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
    beta: float = 0.0,
    pbc: bool = True,
) -> complex:
    """计算 SSH 双原子链模型的 g_{ij,nu}(q)。"""
    dh = dh_dq_ssh_diatomic(
        n_cells=n_cells,
        q=q,
        evec=evec,
        alpha=alpha,
        a=a,
        mass_a=mass_a,
        mass_b=mass_b,
        beta=beta,
        pbc=pbc,
    )
    return np.vdot(psi_i, dh @ psi_j)


def dh_ssh_from_displacements(
    u_a: np.ndarray,
    u_b: np.ndarray,
    alpha: float,
    beta: float = 0.0,
    pbc: bool = True,
) -> np.ndarray:
    """由实空间位移构造 dH/dλ（假定位移与 λ 线性关系）。"""
    if u_a.shape != u_b.shape:
        raise ValueError("u_a 与 u_b 形状必须一致")
    n_cells = u_a.shape[0]
    size = 2 * n_cells
    dh = np.zeros((size, size), dtype=complex)

    for n in range(n_cells):
        a_idx = 2 * n
        b_idx = 2 * n + 1
        n_next = n + 1
        if n_next >= n_cells:
            if not pbc:
                continue
            n_next = pbc_index(n_next, n_cells)
        a_next_idx = 2 * n_next

        delta_intra = u_b[n] - u_a[n]
        delta_inter = u_a[n_next] - u_b[n]

        dh[a_idx, b_idx] += alpha * delta_intra
        dh[b_idx, a_idx] += alpha * delta_intra
        dh[b_idx, a_next_idx] += alpha * delta_inter
        dh[a_next_idx, b_idx] += alpha * delta_inter

        if beta != 0.0:
            dh[a_idx, a_idx] += beta * u_a[n]
            dh[b_idx, b_idx] += beta * u_b[n]

    return dh


def g_ssh_from_displacements(
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    u_a: np.ndarray,
    u_b: np.ndarray,
    alpha: float,
    beta: float = 0.0,
    pbc: bool = True,
) -> complex:
    """由位移模式直接计算 SSH 的耦合矩阵元。"""
    dh = dh_ssh_from_displacements(u_a, u_b, alpha, beta=beta, pbc=pbc)
    return np.vdot(psi_i, dh @ psi_j)


def g_ssh_diatomic_grid(
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    n_cells: int,
    q_vals: np.ndarray,
    eigenvectors: np.ndarray,
    alpha: float,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
    beta: float = 0.0,
    pbc: bool = True,
) -> np.ndarray:
    """计算 SSH 双原子链的 g_{ij,nu}(q) 分布。"""
    if eigenvectors.shape != (q_vals.shape[0], 2, 2):
        raise ValueError("eigenvectors 形状必须为 (n_q, 2, 2)")
    g_vals = np.zeros((q_vals.shape[0], 2), dtype=complex)
    for idx, q in enumerate(q_vals):
        for mode in range(2):
            g_vals[idx, mode] = g_ssh_diatomic(
                psi_i=psi_i,
                psi_j=psi_j,
                n_cells=n_cells,
                q=q,
                evec=eigenvectors[idx, :, mode],
                alpha=alpha,
                a=a,
                mass_a=mass_a,
                mass_b=mass_b,
                beta=beta,
                pbc=pbc,
            )
    return g_vals
