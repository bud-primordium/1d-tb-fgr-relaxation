"""生成附录图表。

当前建议：正文保留 Fig1/2/3/9/10；其余图放附录。

附录图表：
- Fig4: KMC 轨迹（多 N 对比）
- Fig5: 低温/高温对比 + 细致平衡验证
- Fig6: 单因子扫描（参数敏感性）
- Fig7: ME vs KMC 一致性（含稳态检查）
- Fig8: 有效通道数（D_eff / D_pr / 累积概率）
- Fig11: 一步跳阈值（ω_max 扫描）

提示：你已手工微调过 Fig1/Fig11 的 legend，本脚本默认不重画它们，避免覆盖。
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.channel_analysis import (
    cumulative_channel_fraction,
    effective_out_degree,
    participation_ratio,
)
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
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4))

    ax.set_title("Single-band dispersion + energy bins")

    t0 = float(PHYSICS_PARAMS["t0"])
    k_dense = np.linspace(0.0, np.pi, 600)
    E_dense = 2.0 * t0 * np.cos(k_dense)
    ax.plot(k_dense / np.pi, E_dense, color=COLORS["gray"], lw=2.0, alpha=0.9)

    # 叠加离散 k 点（示意 N 对应的离散态数）
    N_demo = 40
    k_disc = np.linspace(0.0, np.pi, N_demo // 2 + 1, endpoint=True)
    E_disc = 2.0 * t0 * np.cos(k_disc)
    ax.scatter(k_disc / np.pi, E_disc, s=12, color=COLORS["gray"], alpha=0.7, zorder=3)

    # motif 分箱：按归一化能量等分四段，对应不可约区的固定 k 区间
    # E_min=-2t0, E_max=2t0 -> 分界 E={1,0,-1}*t0 -> k={π/3, π/2, 2π/3}
    k_edges = np.array([0.0, np.pi / 3.0, np.pi / 2.0, 2.0 * np.pi / 3.0, np.pi], dtype=float)
    bin_names = ["A (high E)", "B", "C", "D (low E)"]
    bin_face = ["#ffe6e6", "#e6f2ff", "#fff2cc", "#e8f5e9"]
    for i in range(4):
        ax.axvspan(k_edges[i] / np.pi, k_edges[i + 1] / np.pi, color=bin_face[i], alpha=0.5, zorder=0)
        ax.text(
            0.5 * (k_edges[i] + k_edges[i + 1]) / np.pi,
            2.05 * t0,
            bin_names[i],
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 代表性 k 点（固定标签，便于直观看“跨栏 vs 逐级”两类路径）
    k_rep = np.array([np.pi / 6.0, 5.0 * np.pi / 12.0, 7.0 * np.pi / 12.0, 5.0 * np.pi / 6.0], dtype=float)
    E_rep = 2.0 * t0 * np.cos(k_rep)
    rep_labels = ["A*", "B*", "C*", "D*"]
    rep_colors = [COLORS["highlight"], COLORS["primary"], COLORS["secondary"], COLORS["tertiary"]]
    ax.scatter(k_rep / np.pi, E_rep, s=35, c=rep_colors, edgecolor="white", linewidth=0.8, zorder=4)
    for x, y, lab in zip(k_rep / np.pi, E_rep, rep_labels):
        ax.text(x, y + 0.08, lab, ha="center", va="bottom", fontsize=8)

    # 示例路径（仅示意：A*→B*→D* 与 A*→B*→C*→D*）
    def _arrow(x0, y0, x1, y1, color, rad=0.0):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=2.0,
                alpha=0.85,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )

    _arrow(k_rep[0] / np.pi, E_rep[0], k_rep[1] / np.pi, E_rep[1], COLORS["primary"], rad=-0.12)
    _arrow(k_rep[1] / np.pi, E_rep[1], k_rep[3] / np.pi, E_rep[3], COLORS["primary"], rad=-0.18)
    _arrow(k_rep[0] / np.pi, E_rep[0], k_rep[1] / np.pi, E_rep[1], COLORS["secondary"], rad=0.12)
    _arrow(k_rep[1] / np.pi, E_rep[1], k_rep[2] / np.pi, E_rep[2], COLORS["secondary"], rad=0.12)
    _arrow(k_rep[2] / np.pi, E_rep[2], k_rep[3] / np.pi, E_rep[3], COLORS["secondary"], rad=0.12)

    # 初态/终态定义（与主实验一致）
    E_min, E_max = -2.0 * t0, 2.0 * t0
    E_init = E_min + 0.9 * (E_max - E_min)
    E_term = E_min + 0.1 * (E_max - E_min)
    k_init_max = float(np.arccos(np.clip(E_init / (2.0 * t0), -1.0, 1.0)))
    k_term_min = float(np.arccos(np.clip(E_term / (2.0 * t0), -1.0, 1.0)))

    ax.axvspan(0.0, k_init_max / np.pi, color=COLORS["primary"], alpha=0.12, zorder=1)
    ax.axvspan(k_term_min / np.pi, 1.0, color=COLORS["secondary"], alpha=0.12, zorder=1)
    ax.axhline(E_init, color=COLORS["primary"], ls="--", lw=1.2, alpha=0.8)
    ax.axhline(E_term, color=COLORS["secondary"], ls="--", lw=1.2, alpha=0.8)

    ax.text(0.02, 0.8, "Initial region\n(top 10%)", transform=ax.transAxes, ha="left", va="top",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.text(0.98, 0.25, "Terminal region\n(bottom 10%)", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-2.3 * t0, 2.3 * t0)
    ax.set_xlabel(r"Momentum $k/\pi$ (irreducible zone)")
    ax.set_ylabel(r"Energy $E$ [$t_0$]")

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_model_schematic.png")
    plt.close(fig)
    print("  [✓] Fig 1 - Model Schematic")


def generate_fig4_trajectories(output_dir: Path):
    """Fig 4: KMC 轨迹（多 N 对比）。"""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    N_list = [20, 40, 80, 160]
    colors_traj = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, N in enumerate(N_list):
        ax = axes[idx // 2, idx % 2]

        W, k_grid, E_grid = build_rate_matrix(n_cells=N, delta_mode="gaussian", **PHYSICS_PARAMS)
        E = np.array(E_grid)
        E_min, E_max = E.min(), E.max()
        E_init = E_min + 0.9 * (E_max - E_min)
        initial_state = int(np.argmin(np.abs(E - E_init)))
        E_term = E_min + 0.1 * (E_max - E_min)

        def terminal_cond(state, energy, time):
            return energy <= E_term

        rng = np.random.default_rng(42 + idx)
        for t_idx in range(5):
            traj = run_trajectory(
                W=W,
                E_grid=E,
                initial_state=initial_state,
                terminal_condition=terminal_cond,
                max_steps=10000,
                rng=rng,
            )
            ax.step(traj.times, traj.energies, where="post", color=colors_traj[t_idx], alpha=0.7, lw=1.2)

        ax.axhline(E_term, color=COLORS["gray"], ls="--", lw=1, label="Terminal")
        ax.axhline(E_init, color=COLORS["gray"], ls=":", lw=1, label="Initial")
        ax.set_xlabel(r"Time [$\hbar/t_0$]")
        ax.set_ylabel(r"Energy [$t_0$]")
        ax.set_title(f"N = {N}")

        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()

    plt.suptitle("KMC Trajectories: Staircase Relaxation", fontsize=12, y=1.0)
    if "handles" in locals():
        LEGEND_BBOX = (0.5, 0.95)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=True,
            framealpha=0.9,
            bbox_to_anchor=LEGEND_BBOX,
        )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "fig4_trajectories.png")
    plt.close(fig)
    print("  [✓] Fig 4 - Trajectories")


def generate_fig8_effective_channels(output_dir: Path):
    """Fig 8: 有效通道数（附录版：降低点云噪声）。"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    N_list = [40, 80, 160, 320]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], COLORS["quaternary"]]

    all_D_eff = {}
    all_D_pr = {}
    all_E = {}

    for N in N_list:
        W, k_grid, E_grid = build_rate_matrix(n_cells=N, delta_mode="gaussian", **PHYSICS_PARAMS)
        E = np.array(E_grid)
        D_eff = effective_out_degree(W, epsilon=0.01)
        D_pr = participation_ratio(W)
        all_D_eff[N] = np.asarray(D_eff, dtype=float)
        all_D_pr[N] = np.asarray(D_pr, dtype=float)
        all_E[N] = E

    # (a) D_eff vs 能量：分箱均值（+四分位带），避免散点糊成一团
    ax = axes[0]
    ax.set_title("(a) $D_{\\rm eff}$ vs energy (binned mean)")

    n_bins = 20
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for i, N in enumerate(N_list):
        E = all_E[N]
        E_norm = (E - E.min()) / (E.max() - E.min())
        D = all_D_eff[N]

        means = np.full(n_bins, np.nan, dtype=float)
        q25 = np.full(n_bins, np.nan, dtype=float)
        q75 = np.full(n_bins, np.nan, dtype=float)
        for b in range(n_bins):
            mask = (E_norm >= bins[b]) & (E_norm < bins[b + 1])
            if not np.any(mask):
                continue
            vals = D[mask]
            means[b] = float(np.mean(vals))
            q25[b] = float(np.quantile(vals, 0.25))
            q75[b] = float(np.quantile(vals, 0.75))

        ax.plot(bin_centers, means, "-", color=colors[i], lw=1.8, label=f"N={N}")
        ax.fill_between(bin_centers, q25, q75, color=colors[i], alpha=0.15, linewidth=0)

    ax.set_xlabel(r"Normalized energy $(E - E_{\min})/(E_{\max} - E_{\min})$")
    ax.set_ylabel(r"$D_{\rm eff}$ (channels with $p_{ij} > 1\%$)")
    handles_n, labels_n = ax.get_legend_handles_labels()
    ax.set_ylim(0, max(float(np.nanmax(all_D_eff[N])) for N in N_list) + 2)

    # (b) 平均有效出度 vs N
    ax = axes[1]
    ax.set_title("(b) Mean effective channels vs N")
    mean_D_eff = [float(np.mean(all_D_eff[N])) for N in N_list]
    mean_D_pr = [float(np.mean(all_D_pr[N])) for N in N_list]

    ax.plot(
        N_list,
        mean_D_eff,
        "o-",
        color=COLORS["primary"],
        markersize=8,
        label=r"$\langle D_{\rm eff} \rangle$ (1% threshold)",
    )
    ax.plot(
        N_list,
        mean_D_pr,
        "s-",
        color=COLORS["secondary"],
        markersize=8,
        label=r"$\langle D_{\rm pr} \rangle$ (participation ratio)",
    )
    ax.plot(N_list, [n - 1 for n in N_list], "--", color=COLORS["gray"], lw=1.5, label="N-1 (naive)")

    ax.set_xlabel("System size N")
    ax.set_ylabel("Mean effective channels")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(N_list))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):d}" if x in N_list else ""))
    ax.minorticks_off()
    ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.9)

    ax.text(
        0.98,
        0.15,
        f"O(1) evidence:\n"
        f"$D_{{\\rm eff}}$ ≈ {float(np.mean(mean_D_eff)):.1f}\n"
        f"$D_{{\\rm pr}}$ ≈ {float(np.mean(mean_D_pr)):.1f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )

    # (c) 累积概率曲线（示意：少数通道占据大部分概率）
    ax = axes[2]
    ax.set_title("(c) Cumulative channel probability (N=80)")

    N_demo = 80
    W, k_grid, E_grid = build_rate_matrix(n_cells=N_demo, delta_mode="gaussian", **PHYSICS_PARAMS)
    E = np.array(E_grid)
    E_norm = (E - E.min()) / (E.max() - E.min())
    idx_high = int(np.argmin(np.abs(E_norm - 0.9)))
    idx_mid = int(np.argmin(np.abs(E_norm - 0.5)))
    idx_low = int(np.argmin(np.abs(E_norm - 0.1)))

    for state_idx, label, color in [
        (idx_high, "High E", COLORS["highlight"]),
        (idx_mid, "Mid E", COLORS["secondary"]),
        (idx_low, "Low E", COLORS["primary"]),
    ]:
        k_vals, cum_prob = cumulative_channel_fraction(W, state_idx)
        ax.plot(k_vals[:20], cum_prob[:20], "o-", color=color, markersize=4, label=label)

    ax.axhline(0.9, color=COLORS["gray"], ls="--", lw=1, label="90%")
    ax.set_xlabel("Number of top channels k")
    ax.set_ylabel("Cumulative probability")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.05)

    ax.text(
        0.4,
        0.4,
        "Few channels capture\nmost transition probability",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    LEGEND_BBOX = (0.5, 1.02)
    fig.legend(
        handles_n,
        labels_n,
        loc="upper center",
        ncol=4,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=LEGEND_BBOX,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "fig8_effective_channels.png")
    plt.close(fig)
    print("  [✓] Fig 8 - Effective Channels")


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

    # Spearman 相关（单调性）：避免对 SciPy 的硬依赖
    try:
        from scipy.stats import spearmanr  # type: ignore
        rho, _pval = spearmanr(tau_me, tau_kmc)
        corr_text = f"Spearman ρ = {rho:.2f}"
    except Exception:
        corr_text = ""

    ax.text(0.98, 0.02,
        f"{corr_text}\n"
        "Different definitions;\n"
        "both saturate for N≥40",
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

    # (c) D_KL vs N（稳态分布验证）
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

    idx_max = int(np.argmax(np.asarray(D_KL_list, dtype=float)))
    ax.annotate(
        f"max at N={int(N_values[idx_max])}",
        (float(N_values[idx_max]), float(D_KL_list[idx_max])),
        textcoords="offset points",
        xytext=(12, 0),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color=COLORS["gray"]),
    )

    ax.axhline(0.02, color=COLORS["gray"], ls="--", lw=1, label="0.02 reference")
    ax.set_xlabel("System size N")
    ax.set_ylabel(r"$D_{\rm KL}(P_{\infty} \| P_{\rm Boltz})$")
    ax.legend(fontsize=8, loc="upper right")

    ax.text(0.98, 0.32,
        f"All N: $D_{{\\rm KL}}\\lesssim {float(np.max(D_KL_list)):.3g}$\n"
        "→ stationary ≈ Boltzmann",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / "fig7_consistency.png")
    plt.close(fig)
    print("  [✓] Fig 7 - Consistency")


def generate_fig11_onehop_threshold(output_dir: Path):
    """Fig 11: 扫描 ω_max，观察“一步跳”(n_hop=1) 何时出现。"""
    data_path = repo_root / "results" / "onehop_scan_data.npz"
    if not data_path.exists():
        print("  [!] 缺少 onehop_scan_data.npz，跳过 Fig 11（先运行 scripts/run_onehop_scan.py）")
        return

    d = dict(np.load(data_path, allow_pickle=True))
    omega = np.asarray(d["omega_max_values"], dtype=float)
    p1 = np.asarray(d["p_onehop"], dtype=float)
    p2 = np.asarray(d["p_twohop"], dtype=float)
    mean_n = np.asarray(d["mean_nhop"], dtype=float)
    deltaE = np.asarray(d["deltaE_to_terminal"], dtype=float)
    n_cells = int(np.asarray(d["n_cells"]).item())
    kT = float(np.asarray(d["kT"]).item())

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # (a) P(n=1) / P(n=2)
    ax = axes[0]
    ax.set_title(r"(a) One-hop probability vs $\omega_{\max}$")
    ax.semilogy(omega, np.maximum(p1, 1e-6), "o-", color=COLORS["highlight"], label=r"$P(n_{\rm hop}=1)$")
    ax.semilogy(omega, np.maximum(p2, 1e-6), "s--", color=COLORS["secondary"], label=r"$P(n_{\rm hop}=2)$")
    ax.set_xlabel(r"$\omega_{\max}$")
    ax.set_ylabel("Probability")
    ax.set_ylim(1e-6, 1.0)

    # 简单阈值参考：需要单声子能量 ≥ 初态到终止阈值的能量落差
    thr = float(np.mean(deltaE))
    ax.axvline(thr, color=COLORS["gray"], ls=":", lw=1.5, label=r"mean $\Delta E_{\rm need}$")
    ax.text(
        0.57,
        0.15,
        rf"$N={n_cells}$, $kT={kT:g}$" + "\n" + rf"$\langle\Delta E_{{\rm need}}\rangle\approx {thr:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # (b) 平均跳数
    ax = axes[1]
    ax.set_title(r"(b) Mean hop count vs $\omega_{\max}$")
    ax.plot(omega, mean_n, "o-", color=COLORS["primary"], label=r"$\langle n_{\rm hop}\rangle$")
    ax.set_xlabel(r"$\omega_{\max}$")
    ax.set_ylabel(r"$\langle n_{\rm hop}\rangle$")
    ax.set_ylim(1.0, max(6.0, float(np.max(mean_n) + 0.5)))
    ax.axvline(thr, color=COLORS["gray"], ls=":", lw=1.5)
    ax.text(
        0.02,
        0.05,
        "Note: non-monotonicity is possible\n(energy matching + q-selectivity)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    LEGEND_BBOX = (0.5, 1.08)
    handles, labels = [], []
    for a in axes:
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, framealpha=0.9, bbox_to_anchor=LEGEND_BBOX)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "fig11_onehop_threshold.png")
    plt.close(fig)
    print("  [✓] Fig 11 - One-hop Threshold")


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
    generate_fig4_trajectories(output_dir)
    generate_fig5_rate_matrix(output_dir)
    generate_fig6_sensitivity(output_dir)
    generate_fig7_consistency(output_dir, data)
    generate_fig8_effective_channels(output_dir)
    generate_fig11_onehop_threshold(output_dir) 

    print(f"\n附录图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
