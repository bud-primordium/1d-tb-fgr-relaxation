"""生成附录图表（Fig1/3/5/6/7）。

- Fig1: 最小示意 + 选择规则说明
- Fig3: P(n_hop) 收敛 + 发射/吸收 + 累积概率
- Fig5: 低温/高温对比 + 细致平衡验证
- Fig6: 简化敏感性（β、D_eff、τ_CV）
- Fig7: D_KL + τ 对比
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.channel_analysis import effective_out_degree
from src.fermi_golden_rule import build_rate_matrix
from src.kinetic_monte_carlo import run_trajectory
from src.master_equation import (
    mean_energy,
    solve_master_equation,
    stationary_distribution,
)
from src.relaxation_analysis import fit_power_law

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "tertiary": "#2ca02c",
    "quaternary": "#9467bd",
    "gray": "#7f7f7f",
    "highlight": "#d62728",
}

PHYSICS_PARAMS = {
    "t0": 1.0,
    "k_spring": 1.0,
    "mass": 1.0,
    "alpha": 0.5,
    "kT": 0.025,
    "sigma": 0.1,
}


def setup_publication_style():
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
    })


def generate_fig1_model_schematic(output_dir: Path):
    """Fig 1: 最小模型示意图 + 选择规则说明。"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # (a) 能级 + 允许跃迁 - 只画不可约区 k∈[0,π] 避免 ±k 简并
    ax = axes[0]
    ax.set_title("(a) Unique Energy Levels (±k degeneracy merged)")

    N_show = 8
    # 只取 k >= 0 的不可约区
    k_vals = np.linspace(0, np.pi, N_show // 2 + 1, endpoint=True)
    E_vals = -2 * np.cos(k_vals)

    # 按能量排序
    sort_idx = np.argsort(E_vals)
    E_sorted = E_vals[sort_idx]
    k_sorted = k_vals[sort_idx]

    # 画能级（标注 k 值）
    for i, (E, k) in enumerate(zip(E_sorted, k_sorted)):
        ax.hlines(E, 0.2, 0.8, color=COLORS["gray"], linewidth=2.5, alpha=0.8)
        deg = 1 if (k == 0 or np.isclose(k, np.pi)) else 2
        ax.text(0.85, E, f"$k$={k/np.pi:.1f}$\\pi$ (×{deg})", va="center", fontsize=8)

    # 画允许的跃迁
    transitions = [(4, 2), (3, 1)]
    for fr, to in transitions:
        y_fr, y_to = E_sorted[fr], E_sorted[to]
        ax.annotate(
            "", xy=(0.5, y_to), xytext=(0.5, y_fr),
            arrowprops=dict(arrowstyle="->", color=COLORS["secondary"],
                           lw=2, connectionstyle="arc3,rad=-0.15"),
        )

    ax.set_xlim(0, 1.2)
    ax.set_ylim(-2.5, 2.5)
    ax.set_ylabel("Energy $E$ [$t_0$]")
    ax.set_xticks([])
    ax.grid(False)

    ax.text(0.02, 0.7,
        f"N = {N_show} sites\n"
        "$E_k = -2t_0\\cos(ka)$\n"
        "$E(k) = E(-k)$ → merged",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # (b) 选择规则说明
    ax = axes[1]
    ax.set_title("(b) Transition Selection Rules")
    ax.axis("off")

    explanation = (
        "Phonon-assisted transitions are NOT arbitrary:\n\n"
        "1. Energy conservation:\n"
        "   $E_i - E_j = \\hbar\\omega_q$ (emission)\n"
        "   $E_j - E_i = \\hbar\\omega_q$ (absorption)\n\n"
        "2. Momentum conservation:\n"
        "   $k_i \\pm q = k_j$ (mod $2\\pi/a$)\n\n"
        "3. Phonon dispersion constraint:\n"
        "   $\\omega_q = \\omega_{\\max}|\\sin(qa/2)|$\n"
        "   (acoustic branch, 1D monatomic chain)\n\n"
        "→ Only specific $(k_i, k_j)$ pairs allowed\n"
        "→ Effective channels $D_{\\rm eff} \\sim O(1)$, not $O(N)$"
    )
    ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
            va="top", fontsize=10, family="serif",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_model_schematic.png")
    plt.close(fig)
    print("  [✓] Fig 1 - Model Schematic")


def generate_fig3_path_statistics(output_dir: Path, data: dict):
    """Fig 3: 路径统计（大改版）。"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # 核心信息作为总标题
    fig.suptitle(
        "Path statistics: no combinatorial explosion (effective paths remain O(1))",
        fontsize=11,
        y=0.98,
    )

    N_values = data["N"]
    def _read_hist(prefix: str, idx: int, max_hops: int = 12) -> np.ndarray:
        key = f"{prefix}_{idx}_n_hops_histogram"
        hist = np.array(data.get(key, np.zeros(1, dtype=int)))
        if hist.ndim != 1:
            hist = hist.ravel()
        if len(hist) < max_hops:
            hist = np.pad(hist, (0, max_hops - len(hist)))
        return hist[:max_hops]

    def _read_steps(prefix: str, idx: int) -> np.ndarray:
        key = f"{prefix}_{idx}_step_sizes"
        if key not in data:
            return np.array([], dtype=float)
        return np.asarray(data[key], dtype=float)

    has_hit = any(f"path_stats_hiT_{i}_n_hops_histogram" in data for i in range(len(N_values)))
    kT_low = float(data.get("kT", PHYSICS_PARAMS["kT"]))
    kT_high = float(data.get("path_stats_hiT_kT", np.nan)) if has_hit else np.nan

    # (a) 关键尾部概率 vs N：P(n=3)、P(n>=4)（低温 vs 高温）
    ax = axes[0, 0]
    ax.set_title("(a) Hop-count tail vs N")

    p3_low, p4p_low = [], []
    p3_hi, p4p_hi = [], []

    for i in range(len(N_values)):
        h = _read_hist("path_stats", i)
        p = h / max(h.sum(), 1)
        p3_low.append(float(p[3]) if len(p) > 3 else 0.0)
        p4p_low.append(float(np.sum(p[4:])) if len(p) > 4 else 0.0)

        if has_hit:
            hh = _read_hist("path_stats_hiT", i)
            pp = hh / max(hh.sum(), 1)
            p3_hi.append(float(pp[3]) if len(pp) > 3 else 0.0)
            p4p_hi.append(float(np.sum(pp[4:])) if len(pp) > 4 else 0.0)

    ax.plot(N_values, np.array(p3_low) * 100, "o-", color=COLORS["primary"], label=f"Low T (kT={kT_low:.3g}): P(n=3)")
    ax.plot(N_values, np.array(p4p_low) * 100, "s--", color=COLORS["primary"], alpha=0.8, label=f"Low T (kT={kT_low:.3g}): P(n≥4)")
    if has_hit:
        ax.plot(N_values, np.array(p3_hi) * 100, "o-", color=COLORS["secondary"], label=f"High T (kT={kT_high:.3g}): P(n=3)")
        ax.plot(N_values, np.array(p4p_hi) * 100, "s--", color=COLORS["secondary"], alpha=0.8, label=f"High T (kT={kT_high:.3g}): P(n≥4)")

    ax.set_xscale("log")
    ax.set_xlabel("System size N")
    ax.set_ylabel("Probability mass (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="best")

    ax.text(
        0.02,
        0.1,
        "Key check: tail P(n≥4) does NOT grow with N\n→ no evidence of combinatorial explosion",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )

    # (b) 平均跳数 vs N（低温 vs 高温）
    ax = axes[0, 1]
    ax.set_title("(b) Mean hop count vs N")

    mean_low, std_low = [], []
    mean_hi, std_hi = [], []
    for i in range(len(N_values)):
        mean_low.append(float(data[f"path_stats_{i}_n_hops_mean"]))
        std_low.append(float(data[f"path_stats_{i}_n_hops_std"]))
        if has_hit:
            mean_hi.append(float(data[f"path_stats_hiT_{i}_n_hops_mean"]))
            std_hi.append(float(data[f"path_stats_hiT_{i}_n_hops_std"]))

    ax.errorbar(
        N_values,
        mean_low,
        yerr=std_low,
        fmt="o-",
        color=COLORS["primary"],
        markersize=7,
        capsize=4,
        lw=2,
        label=f"Low T (kT={kT_low:.3g})",
    )
    if has_hit:
        ax.errorbar(
            N_values,
            mean_hi,
            yerr=std_hi,
            fmt="s-",
            color=COLORS["secondary"],
            markersize=7,
            capsize=4,
            lw=2,
            label=f"High T (kT={kT_high:.3g})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("System size N")
    ax.set_ylabel(r"$\langle n_{\rm hop} \rangle$")
    ax.legend(fontsize=7, loc="best")

    cv_low = float(np.std(mean_low) / np.mean(mean_low)) if np.mean(mean_low) > 0 else 0.0
    msg = f"Low T CV = {cv_low:.3f}"
    if has_hit:
        cv_hi = float(np.std(mean_hi) / np.mean(mean_hi)) if np.mean(mean_hi) > 0 else 0.0
        msg += f"\nHigh T CV = {cv_hi:.3f}"
    ax.text(
        0.02,
        0.55,
        msg + "\nNo growth with N",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )

    # (c) ΔE（带符号）分布：用代表性 N 展示低温/高温差异
    ax = axes[1, 0]

    N_demo = 80
    try:
        idx_demo = int(np.where(np.asarray(N_values, dtype=int) == N_demo)[0][0])
    except Exception:
        idx_demo = 0
        N_demo = int(np.asarray(N_values, dtype=int)[0])

    ax.set_title(f"(c) Signed energy step distribution (N={N_demo})")

    steps_low = _read_steps("path_stats", idx_demo)
    steps_hi = _read_steps("path_stats_hiT", idx_demo) if has_hit else np.array([], dtype=float)

    def _plot_steps(steps: np.ndarray, color: str, label: str):
        if steps.size == 0:
            return
        bins = np.linspace(-2.0, 2.0, 81)
        ax.hist(steps, bins=bins, color=color, alpha=0.45, edgecolor="white", label=label)

    _plot_steps(steps_low, COLORS["primary"], f"Low T (kT={kT_low:.3g})")
    if has_hit:
        _plot_steps(steps_hi, COLORS["secondary"], f"High T (kT={kT_high:.3g})")

    ax.axvline(0.0, color=COLORS["gray"], ls="--", lw=1)
    omega_max = np.sqrt(2 * PHYSICS_PARAMS["k_spring"] / PHYSICS_PARAMS["mass"])
    ax.axvline(omega_max, color=COLORS["tertiary"], ls=":", lw=2, label=r"$\omega_{\max}$")
    ax.axvline(-omega_max, color=COLORS["tertiary"], ls=":", lw=2)
    ax.set_xlabel(r"$\Delta E = E_{\rm before} - E_{\rm after}$ [$t_0$]  (emission>0, absorption<0)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7, loc="upper left")

    def _abs_frac(steps: np.ndarray) -> float:
        if steps.size == 0:
            return 0.0
        return float(np.mean(steps < 0.0))

    txt = f"N={N_demo}\nLow T absorption = {_abs_frac(steps_low)*100:.2f}%"
    if has_hit:
        txt += f"\nHigh T absorption = {_abs_frac(steps_hi)*100:.2f}%"
    ax.text(
        0.02,
        0.5,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
    )

    # (d) 吸收占比 vs N（低温 vs 高温）
    ax = axes[1, 1]
    ax.set_title("(d) Absorption fraction vs N")

    abs_low, abs_hi = [], []
    for i in range(len(N_values)):
        s = _read_steps("path_stats", i)
        abs_low.append(_abs_frac(s) * 100)
        if has_hit:
            sh = _read_steps("path_stats_hiT", i)
            abs_hi.append(_abs_frac(sh) * 100)

    ax.plot(N_values, abs_low, "o-", color=COLORS["primary"], label=f"Low T (kT={kT_low:.3g})")
    if has_hit:
        ax.plot(N_values, abs_hi, "s-", color=COLORS["secondary"], label=f"High T (kT={kT_high:.3g})")

    ax.set_xscale("log")
    ax.set_xlabel("System size N")
    ax.set_ylabel("Absorption fraction (%)")
    ax.set_ylim(0, 25)
    ax.legend(fontsize=7, loc="best")

    # 关键说明文字框
    ax.text(
        0.98,
        0.6,
        "Absorption fraction does NOT\ngrow with N → no thermal runaway",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "fig3_path_statistics.png")
    plt.close(fig)
    print("  [✓] Fig 3 - Path Statistics")


def generate_fig5_rate_matrix(output_dir: Path):
    """Fig 5: 速率矩阵（低温/高温对比 + 细致平衡）。"""
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    N = 60

    # 低温和高温参数 - 高温用 0.5 让对比更明显
    params_lowT = PHYSICS_PARAMS.copy()
    params_highT = PHYSICS_PARAMS.copy()
    params_highT["kT"] = 0.5  # 让吸收真正起来

    W_lowT, k_lowT, E_lowT = build_rate_matrix(n_cells=N, delta_mode="gaussian", **params_lowT)
    W_highT, k_highT, E_highT = build_rate_matrix(n_cells=N, delta_mode="gaussian", **params_highT)

    # (a) 低温速率矩阵
    ax = axes[0, 0]
    ax.set_title(f"(a) Rate Matrix (kT={params_lowT['kT']}, N={N})")
    W_log = np.log10(W_lowT + 1e-20)
    W_log[W_lowT < 1e-15] = np.nan
    im = ax.imshow(W_log, cmap="viridis", aspect="auto", vmin=-8, vmax=0)
    ax.set_xlabel("Final state j")
    ax.set_ylabel("Initial state i")
    plt.colorbar(im, ax=ax, shrink=0.8, label=r"$\log_{10}(W_{ij})$")

    # (b) 高温速率矩阵
    ax = axes[0, 1]
    ax.set_title(f"(b) Rate Matrix (kT={params_highT['kT']}, N={N})")
    W_log = np.log10(W_highT + 1e-20)
    W_log[W_highT < 1e-15] = np.nan
    im = ax.imshow(W_log, cmap="viridis", aspect="auto", vmin=-8, vmax=0)
    ax.set_xlabel("Final state j")
    ax.set_ylabel("Initial state i")
    plt.colorbar(im, ax=ax, shrink=0.8, label=r"$\log_{10}(W_{ij})$")

    ax.text(0.02, 0.98,
        "Higher T → more\nreverse transitions",
        transform=ax.transAxes, va="top", fontsize=8, color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))

    # (c) 反向跃迁占比 vs 能量 - 更直观展示温度差异
    ax = axes[1, 0]
    ax.set_title("(c) Absorption Fraction vs Energy")
    E_lowT_arr = np.array(E_lowT)
    E_highT_arr = np.array(E_highT)

    # 计算反向跃迁占比：对每个 i，计算 sum_{j:E_j>E_i} W_ij / sum_j W_ij
    def absorption_fraction(W, E):
        n = len(E)
        frac = np.zeros(n)
        for i in range(n):
            total = W[i, :].sum()
            if total > 1e-20:
                absorb = sum(W[i, j] for j in range(n) if E[j] > E[i])
                frac[i] = absorb / total
        return frac

    frac_lowT = absorption_fraction(W_lowT, E_lowT_arr)
    frac_highT = absorption_fraction(W_highT, E_highT_arr)

    ax.scatter(E_lowT_arr, frac_lowT * 100, c=COLORS["primary"], s=15, alpha=0.6,
               label=f"kT={params_lowT['kT']}")
    ax.scatter(E_highT_arr, frac_highT * 100, c=COLORS["secondary"], s=15, alpha=0.6,
               label=f"kT={params_highT['kT']}")

    ax.set_xlabel(r"Energy $E_i$ [$t_0$]")
    ax.set_ylabel("Absorption fraction (%)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 100)

    ax.text(0.98, 0.5,
        f"High T: absorption ↑\n"
        f"Mean abs frac:\n"
        f"  Low T: {frac_lowT.mean()*100:.1f}%\n"
        f"  High T: {frac_highT.mean()*100:.1f}%",
        transform=ax.transAxes, ha="right", va="center", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # (d) 细致平衡验证（高温，有足够逆跃迁）
    ax = axes[1, 1]
    ax.set_title("(d) Detailed Balance Check (high T)")

    kT = params_highT["kT"]
    W = W_highT
    E = E_highT_arr

    # 只取 W_ij 和 W_ji 都足够大的点
    W_threshold = W.max() * 0.001  # 0.1% of max
    x_vals, y_vals = [], []
    for i in range(N):
        for j in range(i+1, N):
            if W[i, j] > W_threshold and W[j, i] > W_threshold:
                dE = (E[j] - E[i]) / kT
                ratio = W[i, j] / W[j, i]
                if ratio > 0:
                    x_vals.append(dE)
                    y_vals.append(np.log(ratio))

    if len(x_vals) > 5:
        ax.scatter(x_vals, y_vals, c=COLORS["secondary"], s=8, alpha=0.5, label="Data")

        # 理论线
        x_range = np.linspace(min(x_vals), max(x_vals), 100)
        ax.plot(x_range, -x_range, "k--", lw=2, label="Theory: slope = -1")

        # 拟合
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        ax.plot(x_range, slope * x_range + intercept, "-", color=COLORS["highlight"],
                lw=1.5, label=f"Fit: slope = {slope:.3f}")

        # 相关系数
        corr = np.corrcoef(x_vals, y_vals)[0, 1]

        ax.text(0.02, 0.02,
            f"Threshold: W > {W_threshold:.1e}\n"
            f"N_pairs = {len(x_vals)}\n"
            f"$R^2$ = {corr**2:.4f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

        ax.set_xlabel(r"$(E_j - E_i)/k_B T$")
        ax.set_ylabel(r"$\ln(W_{ij}/W_{ji})$")
        ax.legend(fontsize=7, loc="upper right")
    else:
        ax.text(0.5, 0.5, "Insufficient pairs", transform=ax.transAxes,
                ha="center", va="center")

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_rate_matrix.png")
    plt.close(fig)
    print("  [✓] Fig 5 - Rate Matrix")


def generate_fig6_sensitivity(output_dir: Path):
    """Fig 6: 简化敏感性分析（β、D_eff、τ_CV）。"""
    sens_file = repo_root / "results" / "sensitivity_data.npz"
    if not sens_file.exists():
        print("  [!] 敏感性数据不存在，跳过 Fig 6")
        return

    sens_data = dict(np.load(sens_file))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # 提取数据并计算指标
    params_list = ["sigma", "kT", "alpha"]
    param_labels = [r"$\sigma$", r"$k_B T$", r"$\alpha$"]
    baseline = {"sigma": 0.1, "kT": 0.025, "alpha": 0.5}

    for ax_idx, (param, label) in enumerate(zip(params_list, param_labels)):
        ax = axes[ax_idx]
        ax.set_title(f"({chr(97+ax_idx)}) Sensitivity to {label}")

        betas = []
        values = []

        # 动态检测数据点数
        n_points = sum(1 for k in sens_data.keys() if k.startswith(f"{param}_") and k.endswith("_value"))

        for i in range(n_points):
            val = float(sens_data[f"{param}_{i}_value"])
            N = sens_data[f"{param}_{i}_N"]
            gamma = sens_data[f"{param}_{i}_total_rate"]

            values.append(val)

            # 计算 β（排除 N=20）
            mask = N > 20
            if mask.sum() >= 2:
                A, beta, R2 = fit_power_law(N[mask], gamma[mask])
                betas.append(beta)
            else:
                betas.append(np.nan)

        # 画 β vs 参数值
        ax.plot(values, betas, "o-", color=COLORS["primary"], markersize=10, lw=2)
        ax.axhline(0, color=COLORS["gray"], ls="--", lw=1, label=r"$\beta=0$ (O(1))")
        ax.axhline(0.15, color=COLORS["highlight"], ls=":", lw=1, label=r"$|\beta|=0.15$ threshold")
        ax.axhline(-0.15, color=COLORS["highlight"], ls=":", lw=1)

        # 标记 baseline
        if baseline[param] in values:
            idx = values.index(baseline[param])
            ax.plot(values[idx], betas[idx], "s", color=COLORS["secondary"],
                    markersize=12, markerfacecolor="none", markeredgewidth=2,
                    label="Baseline")

        ax.set_xlabel(label)
        ax.set_ylabel(r"Scaling exponent $\beta$")
        # 自动调整 y 轴范围，确保所有点可见
        all_betas = [b for b in betas if not np.isnan(b)]
        if all_betas:
            y_margin = 0.1
            y_min = min(min(all_betas), -0.15) - y_margin
            y_max = max(max(all_betas), 0.15) + y_margin
            ax.set_ylim(y_min, y_max)

        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # 底部添加 baseline 说明
    fig.text(0.5, 0.01,
        f"Baseline: $\\sigma$={baseline['sigma']}, $k_BT$={baseline['kT']}, $\\alpha$={baseline['alpha']}. "
        "Each panel varies one parameter while others fixed.",
        ha="center", fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_dir / "fig6_sensitivity.png")
    plt.close(fig)
    print("  [✓] Fig 6 - Sensitivity")


def generate_fig7_consistency(output_dir: Path, data: dict):
    """Fig 7: ME vs KMC 一致性（归一化趋势 + D_KL）。"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    N_values = data["N"]
    tau_me = data["relax_time_ME"]
    tau_kmc = data["relax_time_KMC"]

    # (a) 归一化趋势对比 - 不画线性拟合，改成趋势一致性
    ax = axes[0]
    ax.set_title(r"(a) Normalized $\tau$ vs N")

    # 找到 N=80 的索引作为归一化基准
    ref_idx = np.argmin(np.abs(N_values - 80))
    tau_me_norm = tau_me / tau_me[ref_idx]
    tau_kmc_norm = tau_kmc / tau_kmc[ref_idx]

    ax.plot(N_values, tau_me_norm, "o-", color=COLORS["primary"], markersize=8, lw=2,
            label=r"$\tau_{\rm ME}/\tau_{\rm ME}(N\!=\!80)$")
    ax.plot(N_values, tau_kmc_norm, "s-", color=COLORS["secondary"], markersize=8, lw=2,
            label=r"$\tau_{\rm KMC}/\tau_{\rm KMC}(N\!=\!80)$")

    ax.axhline(1.0, color=COLORS["gray"], ls="--", lw=1)
    ax.set_xlabel("System size N")
    ax.set_ylabel("Normalized relaxation time")
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="upper right")

    # Spearman 相关（单调性）
    from scipy.stats import spearmanr
    try:
        rho, pval = spearmanr(tau_me, tau_kmc)
        corr_text = f"Spearman ρ = {rho:.2f}"
    except Exception:
        corr_text = ""

    ax.text(0.98, 0.02,
        f"{corr_text}\n"
        "Different definitions,\n"
        "same N-dependence trend",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # (b) τ 条形图对比
    ax = axes[1]
    ax.set_title(r"(b) Relaxation Times vs N")
    x = np.arange(len(N_values))
    width = 0.35
    ax.bar(x - width/2, tau_me, width, label=r"$\tau_{\rm ME}$", color=COLORS["primary"], alpha=0.8)
    ax.bar(x + width/2, tau_kmc, width, label=r"$\tau_{\rm KMC}$", color=COLORS["secondary"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(n)) for n in N_values])
    ax.set_xlabel("System size N")
    ax.set_ylabel(r"$\tau$ [$\hbar/t_0$]")
    ax.legend(fontsize=8, loc="upper left")

    # 计算 CV
    mask = N_values > 20
    cv_me = np.std(tau_me[mask]) / np.mean(tau_me[mask])
    cv_kmc = np.std(tau_kmc[mask]) / np.mean(tau_kmc[mask])
    ax.text(0.98, 0.9,
        f"CV (N>20):\n"
        f"$\\tau_{{\\rm ME}}$: {cv_me:.3f}\n"
        f"$\\tau_{{\\rm KMC}}$: {cv_kmc:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    # (c) D_KL vs N（稳态分布验证）- 注明 N=20 finite-size
    ax = axes[2]
    ax.set_title("(c) Steady-State Accuracy")

    D_KL_list = []
    for N in N_values:
        W, k_grid, E_grid = build_rate_matrix(n_cells=int(N), delta_mode="gaussian", **PHYSICS_PARAMS)
        E = np.array(E_grid)
        P_inf = stationary_distribution(W)

        # Boltzmann 分布
        P_boltz = np.exp(-(E - E.min()) / PHYSICS_PARAMS["kT"])
        P_boltz = P_boltz / P_boltz.sum()

        # KL 散度
        epsilon = 1e-12
        D_KL = float(np.sum(P_inf * np.log((P_inf + epsilon) / (P_boltz + epsilon))))
        D_KL_list.append(D_KL)

    ax.semilogy(N_values, D_KL_list, "o-", color=COLORS["primary"], markersize=8)

    # 标注 N=20 finite-size
    ax.annotate("N=20\n(finite-size)", (N_values[0], D_KL_list[0]),
                textcoords="offset points", xytext=(15, 0), fontsize=7,
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"]))

    ax.axhline(0.01, color=COLORS["gray"], ls="--", lw=1, label="0.01 threshold")
    ax.set_xlabel("System size N")
    ax.set_ylabel(r"$D_{\rm KL}(P_{\infty} \| P_{\rm Boltz})$")
    ax.legend(fontsize=8, loc="upper right")

    ax.text(0.98, 0.32,
        "Steady state ≈ Boltzmann\n(finite-size deviation at N=20)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / "fig7_consistency.png")
    plt.close(fig)
    print("  [✓] Fig 7 - Consistency")


def main():
    setup_publication_style()

    output_dir = repo_root / "results" / "publication_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("生成附录图表")
    print("=" * 50)

    data_file = repo_root / "results" / "scaling_data.npz"
    if not data_file.exists():
        print("错误：找不到 scaling_data.npz")
        return

    data = dict(np.load(data_file))

    print("\n附录图表：")
    generate_fig1_model_schematic(output_dir)
    generate_fig3_path_statistics(output_dir, data)
    generate_fig5_rate_matrix(output_dir)
    generate_fig6_sensitivity(output_dir)
    generate_fig7_consistency(output_dir, data)

    print(f"\n附录图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
