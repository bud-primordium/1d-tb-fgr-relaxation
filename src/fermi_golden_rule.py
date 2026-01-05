"""Fermi 黄金定则散射速率计算。"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from .lattice import k_grid as make_k_grid
from .lattice import q_grid as make_q_grid
from .phonon_1atom import dispersion_monatomic
from .tb_electron_1band import dispersion as electron_dispersion


def bose_einstein(omega: np.ndarray, kT: float) -> np.ndarray:
    """玻色-爱因斯坦分布 n_q = 1/(exp(ℏω/kT) - 1)。

    Args:
        omega: 声子能量数组（无量纲，单位 t₀）
        kT: 无量纲温度 k_B T / t₀

    Returns:
        玻色占据数数组

    Note:
        - 约定：ℏ = k_B = t₀ = 1，因此所有能量均为无量纲（以 t₀ 为单位）。
        - 处理 ω=0 的情况（声学支 q=0），返回 0 以避免数值发散；对 SSH 单原子链而言
          q=0 时耦合也为 0，因此不会影响速率求和。
    """
    omega_arr = np.asarray(omega, dtype=float)
    if kT <= 0.0:
        return np.zeros_like(omega_arr, dtype=float)

    n = np.zeros_like(omega_arr, dtype=float)
    positive = omega_arr > 0.0
    if np.any(positive):
        x = omega_arr[positive] / float(kT)
        n[positive] = 1.0 / np.expm1(x)
    return n


def delta_broadened(
    energy: float,
    sigma: float,
    mode: Literal["gaussian", "lorentzian"] = "gaussian",
) -> float:
    """展宽的 δ 函数。

    Args:
        energy: 能量差 E_f - E_i ∓ ℏω
        sigma: 展宽参数
        mode: "gaussian" 或 "lorentzian"

    Returns:
        展宽后的 δ 函数值（若 energy 为数组则返回数组）
    """
    if sigma <= 0.0:
        raise ValueError("sigma 必须为正数")
    energy_arr = np.asarray(energy, dtype=float)

    if mode == "gaussian":
        prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        value = prefactor * np.exp(-0.5 * (energy_arr / sigma) ** 2)
    elif mode == "lorentzian":
        value = (1.0 / np.pi) * (sigma / (energy_arr**2 + sigma**2))
    else:
        raise ValueError(f"不支持的展宽类型: {mode}")

    if np.ndim(energy) == 0:
        return float(value)
    return value


def scattering_rate_single(
    k_i: float,
    k_f: float,
    E_i: float,
    E_f: float,
    q_grid: np.ndarray,
    omega_q: np.ndarray,
    g_matrix: np.ndarray,
    kT: float,
    sigma: float,
    n_cells: int,
    delta_mode: Literal["gaussian", "lorentzian"] = "gaussian",
) -> Tuple[float, float]:
    """计算从 k_i 到 k_f 的单步散射速率。

    Args:
        k_i, k_f: 初末态波矢
        E_i, E_f: 初末态能量
        q_grid: 声子波矢网格
        omega_q: 声子频率（与 q_grid 对应）
        g_matrix: 电声耦合矩阵元 g(k_i, q)，形状 (n_q,)
        kT: 无量纲温度 k_B T / t₀
        sigma: 能量展宽参数
        n_cells: 系统尺寸（用于归一化）
        delta_mode: 展宽函数类型

    Returns:
        (W_emission, W_absorption): 发射和吸收贡献

    Note:
        - 动量守恒由调用方检查，这里只对传入的 q 网格求和。
        - 约定：g 不含 1/√N，归一化在此函数内以 1/N 显式处理。
        - k_i、k_f 在此函数中不参与计算（保留是为了接口自解释与调试）。
    """
    q_vals = np.asarray(q_grid, dtype=float)
    omega_vals = np.asarray(omega_q, dtype=float)
    g_vals = np.asarray(g_matrix)
    if q_vals.shape != omega_vals.shape or q_vals.shape != g_vals.shape:
        raise ValueError("q_grid、omega_q、g_matrix 的形状必须一致")
    if n_cells <= 0:
        raise ValueError("n_cells 必须为正整数")

    n_bose = bose_einstein(omega_vals, kT=kT)
    delta_e = float(E_f) - float(E_i)
    g2 = np.abs(g_vals) ** 2

    delta_emission = delta_broadened(delta_e + omega_vals, sigma=sigma, mode=delta_mode)
    delta_absorption = delta_broadened(delta_e - omega_vals, sigma=sigma, mode=delta_mode)

    prefactor = 2.0 * np.pi / float(n_cells)
    w_emission = prefactor * float(np.sum(g2 * (n_bose + 1.0) * delta_emission))
    w_absorption = prefactor * float(np.sum(g2 * n_bose * delta_absorption))
    return w_emission, w_absorption


def _g_monatomic_bloch(
    k_i: float,
    k_f: np.ndarray,
    q_vals: np.ndarray,
    alpha: float,
    mass: float,
    a: float,
) -> np.ndarray:
    """SSH 单原子链在 Bloch 基底下的解析 g(k_i -> k_f; q)。

    该表达式与 electron_phonon.g_monatomic(bloch_state, bloch_state, ...) 等价，
    但避免显式构造 dH/dQ_q 矩阵以提升 build_rate_matrix 的复杂度。
    """
    prefactor = alpha / np.sqrt(mass)
    phase_q = np.exp(1j * q_vals * a) - 1.0
    phase_e = np.exp(1j * k_i * a) + np.exp(-1j * k_f * a)
    return prefactor * phase_q * phase_e


def build_rate_matrix(
    n_cells: int,
    t0: float,
    k_spring: float,
    mass: float,
    alpha: float,
    kT: float,
    sigma: float,
    a: float = 1.0,
    delta_mode: Literal["gaussian", "lorentzian"] = "gaussian",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建完整的散射速率矩阵。

    Args:
        n_cells: 系统尺寸
        t0: 跳跃积分
        k_spring: 弹簧常数
        mass: 原子质量
        alpha: 电声耦合常数
        kT: 无量纲温度 k_B T / t₀
        sigma: 展宽参数
        a: 晶格常数
        delta_mode: 展宽函数类型

    Returns:
        (W, k_grid, E_grid):
        - W: 速率矩阵，W[i,j] = W_{i→j}，形状 (n_cells, n_cells)
        - k_grid: 波矢网格
        - E_grid: 对应的能量

    Note:
        - 对角元 W[i,i] = 0
        - 细致平衡仅近似成立：展宽会引入非共振通道的混合贡献。
    """
    if n_cells <= 0:
        raise ValueError("n_cells 必须为正整数")
    if mass <= 0.0:
        raise ValueError("mass 必须为正数")

    k_vals = make_k_grid(n_cells=n_cells, a=a)
    e_vals = electron_dispersion(k_vals, t0=t0, a=a)
    q_vals = make_q_grid(n_cells=n_cells, a=a)
    omega_vals = dispersion_monatomic(q_vals=q_vals, k_spring=k_spring, mass=mass)
    n_bose = bose_einstein(omega_vals, kT=kT)

    prefactor = 2.0 * np.pi / float(n_cells)
    q_indices = np.arange(n_cells, dtype=int)

    W = np.zeros((n_cells, n_cells), dtype=float)
    for i in range(n_cells):
        j_indices = (i + q_indices) % n_cells
        k_i = float(k_vals[i])
        e_i = float(e_vals[i])
        k_f = k_vals[j_indices]
        e_f = e_vals[j_indices]

        g_vals = _g_monatomic_bloch(
            k_i=k_i,
            k_f=k_f,
            q_vals=q_vals,
            alpha=alpha,
            mass=mass,
            a=a,
        )
        g2 = np.abs(g_vals) ** 2

        delta_e = e_f - e_i
        delta_emission = delta_broadened(
            delta_e + omega_vals,
            sigma=sigma,
            mode=delta_mode,
        )
        delta_absorption = delta_broadened(
            delta_e - omega_vals,
            sigma=sigma,
            mode=delta_mode,
        )

        rates = prefactor * g2 * (
            (n_bose + 1.0) * delta_emission + n_bose * delta_absorption
        )
        W[i, j_indices] = rates
        W[i, i] = 0.0

    return W, k_vals, e_vals


def total_rate(W: np.ndarray, state_idx: int) -> float:
    """计算从给定态离开的总散射率 Γ_i = Σ_j W_{i→j}。"""
    w_row = np.asarray(W[state_idx], dtype=float)
    return float(np.sum(w_row))


def characteristic_step(
    W: np.ndarray,
    E_grid: np.ndarray,
    state_idx: int,
) -> float:
    """计算特征能量步长 ΔE_avg。"""
    w_row = np.asarray(W[state_idx], dtype=float)
    gamma = float(np.sum(w_row))
    if gamma <= 0.0:
        return 0.0
    e_vals = np.asarray(E_grid, dtype=float)
    delta_e = float(e_vals[state_idx]) - e_vals
    return float(np.sum(w_row * delta_e) / gamma)


def kT_from_kelvin(temperature_K: float, t0_eV: float = 1.0) -> float:
    """将温度（K）转换为无量纲 kT = k_B T / t₀。

    Args:
        temperature_K: 温度（开尔文）
        t0_eV: 跳跃积分能量（eV），默认 1 eV

    Returns:
        无量纲温度 kT
    """
    if t0_eV <= 0.0:
        raise ValueError("t0_eV 必须为正数")
    KB_EV_PER_K = 8.617333262145e-5
    return float(KB_EV_PER_K * temperature_K / t0_eV)


def tau_to_seconds(tau_dimless: float, t0_eV: float = 1.0) -> float:
    """将无量纲时间 τ（单位：ℏ/t₀）转换为秒。"""
    if t0_eV <= 0.0:
        raise ValueError("t0_eV 必须为正数")
    HBAR_EV_S = 6.582119569e-16
    return float(tau_dimless * HBAR_EV_S / t0_eV)
