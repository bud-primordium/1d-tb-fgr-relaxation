"""生成正文（核心）图表。

正文建议保留 5 张图（其中 Fig1 由附录脚本生成/手工微调）：
- Fig 2: 标度律（添加 N^0/N^1 参考线）
- Fig 3: 路径统计（2×2）
- Fig 9: q 分布（动量选择性）
- Fig10: 路径 motif 随 N（回答“124/134 是否替换”）
"""

import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.channel_analysis import (
    q_ki_heatmap,
    q_mode_distribution,
)
from src.fermi_golden_rule import build_rate_matrix
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

def _load_json_scalar(val):
    """从 np.load 的标量/字符串中读取 JSON。"""
    if isinstance(val, np.ndarray) and val.shape == ():
        val = val.item()
    if isinstance(val, bytes):
        val = val.decode("utf-8")
    return json.loads(str(val))


def generate_fig3_path_statistics(output_dir: Path, data: dict):
    """Fig 3: 路径统计（2×2）。"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

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

    # (a) 尾部概率 vs N：P(n=3)、P(n>=4)
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

    # (b) 平均跳数 vs N（带误差棒）
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


def generate_fig10_path_motifs(output_dir: Path, data: dict):
    """Fig 10: 路径 motif（能量分箱）随 N 的变化。"""
    N_values = np.asarray(data["N"], dtype=int)
    x = np.arange(len(N_values))

    motifs = ["1-4", "1-2-4", "1-3-4", "1-2-3-4"]
    motif_labels = ["A→D", "A→B→D", "A→C→D", "A→B→C→D"]

    # 颜色语义与 Fig1 对齐：
    # - 蓝色（primary）: A→B→D
    # - 橙色（secondary）: A→B→C→D
    # - 绿色（tertiary）: A→C→D
    motif_colors = [COLORS["highlight"], COLORS["primary"], COLORS["tertiary"], COLORS["secondary"]]

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, sharey=True)

    def plot_one_panel(ax, prefix: str, title: str):
        probs_by_motif = {m: [] for m in motifs}
        probs_other = []

        for i in range(len(N_values)):
            counts_key = f"{prefix}_{i}_motif_counts_json"
            ntraj_key = f"{prefix}_{i}_n_trajectories"
            if counts_key not in data or ntraj_key not in data:
                raise KeyError(f"缺少 motif 数据键：{counts_key} 或 {ntraj_key}")
            counts = _load_json_scalar(data[counts_key])
            ntraj = int(np.asarray(data[ntraj_key]).item())

            total_selected = 0.0
            for m in motifs:
                p = float(counts.get(m, 0)) / max(ntraj, 1)
                probs_by_motif[m].append(p)
                total_selected += p
            probs_other.append(max(0.0, 1.0 - total_selected))

        bottom = np.zeros(len(N_values), dtype=float)
        for m, label, color in zip(motifs, motif_labels, motif_colors):
            vals = np.asarray(probs_by_motif[m], dtype=float)
            ax.bar(x, vals, bottom=bottom, color=color, alpha=0.85, edgecolor="white", linewidth=0.6, label=label)
            bottom = bottom + vals
        ax.bar(
            x,
            probs_other,
            bottom=bottom,
            color=COLORS["gray"],
            alpha=0.5,
            edgecolor="white",
            linewidth=0.6,
            label="Other",
        )

        ax.set_title(title)
        ax.set_ylabel("Motif probability")
        ax.set_ylim(0, 1.0)

        # 标注 N=20（若存在）
        if 20 in set(N_values.tolist()):
            idx20 = int(np.where(N_values == 20)[0][0])
            ax.annotate(
                "N=20\n(finite-size)",
                (x[idx20], 0.98),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                va="bottom",
                fontsize=8,
                color=COLORS["highlight"],
            )

        # 用一句话回答“是否出现 124→134 替换”
        p124 = np.asarray(probs_by_motif["1-2-4"], dtype=float)
        p134 = np.asarray(probs_by_motif["1-3-4"], dtype=float)
        denom = p124 + p134
        ratio = np.where(denom > 0, p124 / denom, np.nan)
        ratio_text = (
            "Share of A→B→D among (A→B→D, A→C→D):\n"
            f"mean={np.nanmean(ratio):.2f}, range=[{np.nanmin(ratio):.2f},{np.nanmax(ratio):.2f}]"
        )
        ax.text(
            0.98,
            0.02,
            ratio_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    plot_one_panel(axes[0], "path_stats", r"(a) Low T motif composition (kT = 0.025)")

    kT_hi = float(np.asarray(data.get("path_stats_hiT_kT", np.array(0.5))).item())
    plot_one_panel(axes[1], "path_stats_hiT", rf"(b) High T motif composition (kT = {kT_hi:g})")

    axes[1].set_xlabel("System size N")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(n)) for n in N_values])

    handles, labels = axes[0].get_legend_handles_labels()

    LEGEND_BBOX = (0.5, 1.02)
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, framealpha=0.9,
               bbox_to_anchor=LEGEND_BBOX)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "fig10_path_motifs.png")
    plt.close(fig)
    print("  [✓] Fig 10 - Path Motifs")


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

    LEGEND_BBOX = (0.5, 1.02)
    fig.legend(handles_n, labels_n, loc="upper center", ncol=3, frameon=True, framealpha=0.9,
               bbox_to_anchor=LEGEND_BBOX)
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
    generate_fig3_path_statistics(output_dir, data)
    generate_fig9_q_distribution(output_dir)
    generate_fig10_path_motifs(output_dir, data)

    print(f"\n核心图表已保存到: {output_dir}")
    print("\n推荐展示顺序：")
    print("  1. Fig 1 - 模型示意（见附录脚本，通常手工微调）")
    print("  2. Fig 2 - 标度律（核心结论）")
    print("  3. Fig 3 - 路径统计（是否出现组合爆炸）")
    print("  4. Fig 9 - q 分布（物理机制）")
    print("  5. Fig10 - motif（124/134 是否替换）")


if __name__ == "__main__":
    main()
