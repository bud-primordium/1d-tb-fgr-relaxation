"""生成核心图表

核心图表：
- Fig 2: 标度律（添加 N^0/N^1 参考线）
- Fig 4: KMC 轨迹
- Fig 8: 有效通道数（新增）
- Fig 9: q 模式分布（新增）
"""

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.channel_analysis import (
    cumulative_channel_fraction,
    effective_out_degree,
    participation_ratio,
    q_ki_heatmap,
    q_mode_distribution,
)
from src.fermi_golden_rule import build_rate_matrix
from src.kinetic_monte_carlo import run_trajectory
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


def generate_fig2_scaling_laws(output_dir: Path, data: Dict):
    """Fig 2: 核心标度律（添加 N^0/N^1 参考线）。"""
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))

    N = data["N"]
    gamma = data["total_rate"]
    char_step = data["char_step"]
    tau_me = data["relax_time_ME"]
    tau_kmc = data["relax_time_KMC"]

    finite_size_mask = N <= 20
    mask = ~finite_size_mask

    # (a) Γ vs N
    ax = axes[0, 0]
    ax.loglog(N, gamma, "o", color=COLORS["primary"], markersize=8, label="Data")
    ax.loglog(N[finite_size_mask], gamma[finite_size_mask], "o", color=COLORS["highlight"],
              markersize=10, markerfacecolor="none", markeredgewidth=2, label="N≤20 (finite-size)")

    # 添加 N^0 和 N^1 参考线
    N_ref = np.array([N.min(), N.max()])
    gamma_mean = np.mean(gamma[mask])
    ax.loglog(N_ref, [gamma_mean, gamma_mean], "--", color=COLORS["gray"], lw=1.5,
              label=r"$N^0$ (size-independent)")
    ax.loglog(N_ref, gamma_mean * N_ref / N_ref[0], ":", color=COLORS["tertiary"], lw=1.5,
              label=r"$N^1$ (linear)")

    if mask.sum() >= 2:
        A, beta, R2 = fit_power_law(N[mask], gamma[mask])
        N_fit = np.linspace(N[mask].min(), N[mask].max(), 100)
        ax.loglog(N_fit, A * N_fit**beta, "-", color=COLORS["secondary"], lw=2,
                  label=f"Fit: $N^{{{beta:.2f}}}$")

    ax.set_xlabel("System size N")
    ax.set_ylabel(r"Total rate $\Gamma$ [$t_0/\hbar$]")
    ax.set_title(r"(a) Scaling: $\Gamma \propto N^\beta$")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 1))

    # (b) ΔE vs N
    ax = axes[0, 1]
    ax.plot(N, char_step, "s-", color=COLORS["secondary"], markersize=8)
    ax.axhline(np.mean(char_step[mask]), color=COLORS["gray"], ls="--",
               label=f"Mean: {np.mean(char_step[mask]):.2f}")
    omega_max = np.sqrt(2 * PHYSICS_PARAMS["k_spring"] / PHYSICS_PARAMS["mass"])
    ax.axhline(omega_max, color=COLORS["tertiary"], ls=":", label=f"$\\omega_{{max}}$={omega_max:.2f}")
    ax.set_xlabel("System size N")
    ax.set_ylabel(r"Characteristic step $\Delta E$ [$t_0$]")
    ax.set_title("(b) Energy Step Convergence")
    ax.legend(fontsize=7)

    # (c) τ vs N
    ax = axes[1, 0]
    ax.plot(N, tau_me, "o-", color=COLORS["primary"], markersize=8, label=r"$\tau_{\rm ME}$")
    ax.plot(N, tau_kmc, "s-", color=COLORS["secondary"], markersize=8, label=r"$\tau_{\rm KMC}$")
    ax.set_xlabel("System size N")
    ax.set_ylabel(r"Relaxation time $\tau$ [$\hbar/t_0$]")
    ax.set_title("(c) Two Relaxation Time Definitions")
    ax.legend(fontsize=8, loc="center right")

    ax.text(0.02, 0.25,
        r"$\tau_{\rm ME}$: energy decay to equilibrium" + "\n"
        r"$\tau_{\rm KMC}$: mean first passage time",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # (d) Summary
    ax = axes[1, 1]
    ax.axis("off")

    A, beta, R2 = fit_power_law(N[mask], gamma[mask])
    cv_step = np.std(char_step[mask]) / np.mean(char_step[mask])
    cv_tau_me = np.std(tau_me[mask]) / np.mean(tau_me[mask])
    cv_tau_kmc = np.std(tau_kmc[mask]) / np.mean(tau_kmc[mask])

    summary = (
        "Summary (N > 20):\n\n"
        f"• Γ scaling exponent β = {beta:.3f}\n"
        f"  Target: |β| < 0.15 → {'✓ PASS' if abs(beta) < 0.15 else '✗'}\n\n"
        f"• ΔE coefficient of variation = {cv_step:.3f}\n"
        f"  Target: < 0.15 → {'✓ PASS' if cv_step < 0.15 else '✗'}\n\n"
        f"• τ_ME  CV = {cv_tau_me:.3f}\n"
        f"• τ_KMC CV = {cv_tau_kmc:.3f}\n\n"
        "Conclusion:\n"
        "Γ, ΔE, τ are all O(1) for N ≥ 40"
    )
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, va="top", fontsize=9, family="monospace")
    ax.set_title("(d) Summary")

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_scaling_laws.png")
    plt.close(fig)
    print("  [✓] Fig 2 - Scaling Laws")


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
            traj = run_trajectory(W=W, E_grid=E, initial_state=initial_state,
                                  terminal_condition=terminal_cond, max_steps=10000, rng=rng)
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
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, 0.95))
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "fig4_trajectories.png")
    plt.close(fig)
    print("  [✓] Fig 4 - Trajectories")


def generate_fig8_effective_channels(output_dir: Path):
    """Fig 8: 有效通道数（核心新增图）。"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    N_list = [40, 80, 160, 320]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"], COLORS["quaternary"]]

    all_D_eff = {}
    all_D_pr = {}
    all_E = {}

    for i, N in enumerate(N_list):
        W, k_grid, E_grid = build_rate_matrix(n_cells=N, delta_mode="gaussian", **PHYSICS_PARAMS)
        E = np.array(E_grid)
        D_eff = effective_out_degree(W, epsilon=0.01)
        D_pr = participation_ratio(W)
        all_D_eff[N] = D_eff
        all_D_pr[N] = D_pr
        all_E[N] = E

    # (a) 有效出度 vs 能量
    ax = axes[0]
    ax.set_title("(a) Effective Out-Degree vs Energy")
    for i, N in enumerate(N_list):
        E_norm = (all_E[N] - all_E[N].min()) / (all_E[N].max() - all_E[N].min())
        ax.scatter(E_norm, all_D_eff[N], c=colors[i], s=10, alpha=0.6, label=f"N={N}")
    ax.set_xlabel(r"Normalized energy $(E - E_{\min})/(E_{\max} - E_{\min})$")
    ax.set_ylabel(r"$D_{\rm eff}$ (channels with $p_{ij} > 1\%$)")
    handles_n, labels_n = ax.get_legend_handles_labels()
    ax.set_ylim(0, max(max(all_D_eff[N]) for N in N_list) + 2)

    # (b) 平均有效出度 vs N
    ax = axes[1]
    ax.set_title("(b) Mean Effective Out-Degree vs N")
    mean_D_eff = [np.mean(all_D_eff[N]) for N in N_list]
    mean_D_pr = [np.mean(all_D_pr[N]) for N in N_list]

    ax.plot(N_list, mean_D_eff, "o-", color=COLORS["primary"], markersize=8,
            label=r"$\langle D_{\rm eff} \rangle$ (1% threshold)")
    ax.plot(N_list, mean_D_pr, "s-", color=COLORS["secondary"], markersize=8,
            label=r"$\langle D_{\rm pr} \rangle$ (participation ratio)")

    # 添加 N-1 参考线（naive 预期）
    ax.plot(N_list, [n - 1 for n in N_list], "--", color=COLORS["gray"], lw=1.5,
            label="N-1 (naive: all channels)")

    ax.set_xlabel("System size N")
    ax.set_ylabel("Mean effective channels")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(N_list))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):d}" if x in N_list else ""))
    ax.minorticks_off()
    ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.9)

    # 添加注释
    ax.text(0.98, 0.15,
        f"O(1) evidence:\n"
        f"$D_{{\\rm eff}}$ ≈ {np.mean(mean_D_eff):.1f}\n"
        f"$D_{{\\rm pr}}$ ≈ {np.mean(mean_D_pr):.1f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    # (c) 累积概率曲线
    ax = axes[2]
    ax.set_title("(c) Cumulative Channel Probability")

    N_demo = 80
    W, k_grid, E_grid = build_rate_matrix(n_cells=N_demo, delta_mode="gaussian", **PHYSICS_PARAMS)
    E = np.array(E_grid)

    # 选几个不同能量的态
    E_norm = (E - E.min()) / (E.max() - E.min())
    idx_high = np.argmin(np.abs(E_norm - 0.9))
    idx_mid = np.argmin(np.abs(E_norm - 0.5))
    idx_low = np.argmin(np.abs(E_norm - 0.1))

    for state_idx, label, color in [(idx_high, "High E", COLORS["highlight"]),
                                     (idx_mid, "Mid E", COLORS["secondary"]),
                                     (idx_low, "Low E", COLORS["primary"])]:
        k_vals, cum_prob = cumulative_channel_fraction(W, state_idx)
        ax.plot(k_vals[:20], cum_prob[:20], "o-", color=color, markersize=4, label=label)

    ax.axhline(0.9, color=COLORS["gray"], ls="--", lw=1, label="90% threshold")
    ax.set_xlabel("Number of top channels k")
    ax.set_ylabel("Cumulative probability")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.05)

    ax.text(0.4, 0.4,
        "Few channels capture\nmost transition probability",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.legend(handles_n, labels_n, loc="upper center", ncol=4, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "fig8_effective_channels.png")
    plt.close(fig)
    print("  [✓] Fig 8 - Effective Channels")


def generate_fig9_q_distribution(output_dir: Path):
    """Fig 9: q 模式分布（核心新增图）。"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    N_list = [40, 80, 160]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"]]

    # (a) 热图 (k_i, q)
    ax = axes[0]
    ax.set_title("(a) Transition Rate vs ($k_i$, $q$)")

    N_demo = 80
    W, k_grid, E_grid = build_rate_matrix(n_cells=N_demo, delta_mode="gaussian", **PHYSICS_PARAMS)
    k_vals, q_vals, W_kq = q_ki_heatmap(W, np.array(k_grid))

    W_log = np.log10(W_kq + 1e-20)
    W_log[W_kq < 1e-15] = np.nan
    im = ax.imshow(W_log.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[k_vals.min(), k_vals.max(), q_vals.min(), q_vals.max()],
                   vmin=-8, vmax=0)
    ax.set_xlabel(r"Initial momentum $k_i$")
    ax.set_ylabel(r"Phonon momentum $q$")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"$\log_{10}(W_{ij})$")

    # 标记 q = ±π
    ax.axhline(np.pi, color="white", ls="--", lw=1, alpha=0.7)
    ax.axhline(-np.pi, color="white", ls="--", lw=1, alpha=0.7)

    # (b) 边际分布
    ax = axes[1]
    ax.set_title("(b) Total Rate vs Phonon Momentum $q$")

    q_peaks = []
    for i, N in enumerate(N_list):
        W, k_grid, E_grid = build_rate_matrix(n_cells=N, delta_mode="gaussian", **PHYSICS_PARAMS)
        q_vals, intensity = q_mode_distribution(W, np.array(k_grid))
        ax.plot(q_vals, intensity, "-", color=colors[i], lw=1.5, label=f"N={N}")

        # 找峰
        peak_idx = np.argmax(intensity)
        q_peaks.append(q_vals[peak_idx])

    ax.axvline(np.pi, color=COLORS["gray"], ls="--", lw=1, alpha=0.7)
    ax.axvline(-np.pi, color=COLORS["gray"], ls="--", lw=1, alpha=0.7)
    ax.set_xlabel(r"Phonon momentum $q$")
    ax.set_ylabel(r"$\sum_i W_{i,i+q}$")
    handles_n, labels_n = ax.get_legend_handles_labels()

    ax.text(0.5, 0.25,
        "Peak near BZ boundary\n($q ≈ ±π$: high-freq acoustic)",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # (c) 峰位置 vs N
    ax = axes[2]
    ax.set_title("(c) Dominant $q$ vs System Size")

    N_extended = [20, 40, 80, 160, 320]
    q_peaks_ext = []
    for N in N_extended:
        W, k_grid, E_grid = build_rate_matrix(n_cells=N, delta_mode="gaussian", **PHYSICS_PARAMS)
        q_vals, intensity = q_mode_distribution(W, np.array(k_grid))
        # 只看 q > 0 的峰（因为对称）
        pos_mask = q_vals > 0
        if pos_mask.sum() > 0:
            peak_idx = np.argmax(intensity[pos_mask])
            q_peaks_ext.append(q_vals[pos_mask][peak_idx])
        else:
            q_peaks_ext.append(np.nan)

    ax.plot(N_extended, q_peaks_ext, "o-", color=COLORS["primary"], markersize=8)
    ax.axhline(np.pi, color=COLORS["gray"], ls="--", lw=1.5, label=r"$\pi$ (BZ boundary)")
    ax.set_xlabel("System size N")
    ax.set_ylabel(r"$q_{\rm peak}$ (dominant phonon momentum)")
    ax.legend(fontsize=8, loc="upper left", frameon=True, framealpha=0.9)
    ax.set_ylim(0, np.pi * 1.2)

    ax.text(0.98, 0.02,
        "N-independent:\nselective phonon coupling",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    fig.legend(handles_n, labels_n, loc="upper center", ncol=3, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "fig9_q_distribution.png")
    plt.close(fig)
    print("  [✓] Fig 9 - q Distribution")


def main():
    setup_publication_style()

    output_dir = repo_root / "results" / "publication_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("生成核心图表（精简版）")
    print("=" * 50)

    data_file = repo_root / "results" / "scaling_data.npz"
    if not data_file.exists():
        print("错误：找不到 scaling_data.npz")
        return

    data = dict(np.load(data_file))

    print("\n核心图表：")
    generate_fig2_scaling_laws(output_dir, data)
    generate_fig4_trajectories(output_dir)
    generate_fig8_effective_channels(output_dir)
    generate_fig9_q_distribution(output_dir)

    print(f"\n核心图表已保存到: {output_dir}")
    print("\n推荐展示顺序：")
    print("  1. Fig 2 - 标度律（核心结论）")
    print("  2. Fig 4 - 轨迹（直观理解）")
    print("  3. Fig 8 - 有效通道数（回答路径爆炸问题）")
    print("  4. Fig 9 - q 分布（物理机制）")


if __name__ == "__main__":
    main()
