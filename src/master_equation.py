"""主方程求解模块（无量纲单位）。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from scipy.integrate import solve_ivp
    from scipy.linalg import eig, expm
except Exception:  # pragma: no cover
    solve_ivp = None  # type: ignore[assignment]
    eig = None  # type: ignore[assignment]
    expm = None  # type: ignore[assignment]


def generator_from_rates(W: np.ndarray) -> np.ndarray:
    """由跃迁率矩阵 W[i,j]=W_{i→j} 构造生成矩阵 Q。

    主方程：
        dP/dt = Q^T P

    其中：
        Q[i,j] = W[i,j] (i!=j)
        Q[i,i] = -Σ_{j!=i} W[i,j]
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W 必须为方阵")
    if np.any(W < 0.0):
        raise ValueError("W 不允许出现负速率")

    Q = W.copy()
    out_rates = np.sum(Q, axis=1)
    np.fill_diagonal(Q, -out_rates)
    return Q


def solve_master_equation(
    W: np.ndarray,
    P0: np.ndarray,
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    method: str = "RK45",
) -> Tuple[np.ndarray, np.ndarray]:
    """求解主方程 dP/dt = Q^T P（无量纲时间）。

    Args:
        W: 跃迁率矩阵，W[i,j] = W_{i→j}，对角元通常为 0
        P0: 初始概率分布（长度 n）
        t_span: 时间区间 (t_start, t_end)
        t_eval: 输出时间点（若为 None 则由求解器决定）
        method: "RK45" 等 ODE 方法；或 "expm" 使用矩阵指数精确传播

    Returns:
        (t, P): 时间数组与概率矩阵，P[time_idx, state_idx]
    """
    Q = generator_from_rates(W)
    P0 = np.asarray(P0, dtype=float)
    if P0.ndim != 1 or P0.shape[0] != Q.shape[0]:
        raise ValueError("P0 必须为长度 n 的一维数组")

    total = float(np.sum(P0))
    if total <= 0.0:
        raise ValueError("P0 的总概率必须为正")
    P0 = P0 / total

    if method == "expm":
        if expm is None:  # pragma: no cover
            raise RuntimeError("未安装 scipy，无法使用 expm 传播")
        if t_eval is None:
            raise ValueError("method='expm' 需要提供 t_eval")
        t = np.asarray(t_eval, dtype=float)
        t0 = float(t_span[0])
        P_list = []
        for ti in t:
            A = expm(Q.T * (float(ti) - t0))
            P_list.append(np.asarray(A @ P0, dtype=float))
        P = np.stack(P_list, axis=0)
        return t, P

    if solve_ivp is None:  # pragma: no cover
        raise RuntimeError("未安装 scipy，无法使用 solve_ivp")

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return Q.T @ y

    result = solve_ivp(
        rhs,
        t_span=(float(t_span[0]), float(t_span[1])),
        y0=P0,
        t_eval=None if t_eval is None else np.asarray(t_eval, dtype=float),
        method=method,
        vectorized=False,
        rtol=1e-8,
        atol=1e-10,
    )
    if not result.success:
        raise RuntimeError(f"主方程求解失败: {result.message}")

    t = np.asarray(result.t, dtype=float)
    P = np.asarray(result.y.T, dtype=float)
    return t, P


def mean_energy(P: np.ndarray, E_grid: np.ndarray) -> np.ndarray:
    """计算各时刻的平均能量 ⟨E(t)⟩ = Σ_i P_i(t) E_i。"""
    E = np.asarray(E_grid, dtype=float)
    P_arr = np.asarray(P, dtype=float)
    if P_arr.ndim == 1:
        if P_arr.shape[0] != E.shape[0]:
            raise ValueError("P 与 E_grid 长度不一致")
        return np.array([float(np.dot(P_arr, E))], dtype=float)
    if P_arr.ndim != 2 or P_arr.shape[1] != E.shape[0]:
        raise ValueError("P 必须为形状 (n_time, n_state) 的数组")
    return P_arr @ E


def relaxation_time_from_energy(
    t: np.ndarray,
    E_mean: np.ndarray,
    E_target: float,
    threshold: float = 0.01,
) -> float:
    """从能量衰减曲线估算弛豫时间。

    定义：⟨E(τ)⟩ - E_target < threshold * (E_0 - E_target)
    """
    t = np.asarray(t, dtype=float)
    E_mean = np.asarray(E_mean, dtype=float)
    if t.ndim != 1 or E_mean.ndim != 1 or t.shape[0] != E_mean.shape[0]:
        raise ValueError("t 与 E_mean 必须为等长一维数组")
    if t.shape[0] == 0:
        return float("nan")

    E0 = float(E_mean[0])
    bound = float(E_target) + float(threshold) * (E0 - float(E_target))
    idx = np.argmax(E_mean <= bound)
    if E_mean[idx] <= bound:
        return float(t[idx])
    return float("nan")


def stationary_distribution(W: np.ndarray) -> np.ndarray:
    """计算稳态分布（求解 Q^T P = 0，sum(P) = 1）。

    使用 SVD 方法求解 Q^T 的零空间，比特征值分解更稳健。
    如果 SVD 失败，回退到 Boltzmann 分布（需要能量信息时）。
    """
    Q = generator_from_rates(W)
    n = Q.shape[0]

    # 方法1：SVD 求零空间（最稳健）
    try:
        U, S, Vh = np.linalg.svd(Q.T, full_matrices=True)
        # 最小奇异值对应的右奇异向量是零空间
        null_idx = np.argmin(S)
        # 检查最小奇异值是否足够小（相对于最大奇异值）
        if S.max() > 0 and S[null_idx] / S.max() < 1e-10:
            vec = Vh[null_idx, :]
            vec = np.real(vec)
            # 确保非负
            if np.min(vec) < 0:
                vec = -vec
            vec = np.maximum(vec, 0.0)
            s = float(np.sum(vec))
            if s > 0:
                P = vec / s
                # 验证残差
                residual = np.linalg.norm(Q.T @ P)
                if residual < 1e-6:
                    return P.astype(float)
    except Exception:
        pass

    # 方法2：求解增广线性系统 [Q^T; 1...1] @ P = [0; 1]
    try:
        A = np.vstack([Q.T, np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1.0
        P, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        P = np.maximum(P, 0.0)
        s_sum = float(np.sum(P))
        if s_sum > 0:
            P = P / s_sum
            residual = np.linalg.norm(Q.T @ P)
            if residual < 1e-6:
                return P.astype(float)
    except Exception:
        pass

    # 方法3：特征值分解（原方法，作为后备）
    if eig is None:
        evals, evecs = np.linalg.eig(Q.T)
    else:
        evals, evecs = eig(Q.T)

    evals = np.asarray(evals)
    evecs = np.asarray(evecs)
    idx = int(np.argmin(np.abs(evals)))
    vec = np.real(evecs[:, idx])

    if np.allclose(vec, 0.0):
        raise RuntimeError("未能提取稳态特征向量")

    vec = np.where(vec < 0.0, 0.0, vec)
    s = float(np.sum(vec))
    if s <= 0.0:
        vec = np.abs(np.real(evecs[:, idx]))
        s = float(np.sum(vec))
    if s <= 0.0:
        raise RuntimeError("稳态分布归一化失败")
    P = vec / s
    if P.shape != (n,):
        P = np.reshape(P, (n,))
    return P

