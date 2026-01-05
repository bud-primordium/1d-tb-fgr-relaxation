"""运行完整的标度律验证实验并保存数据。

执行主实验：
- 不同系统尺寸 N
- 固定物理参数（默认值）
- 输出 scaling_data.npz 供后续可视化使用
"""

import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.relaxation_analysis import fit_power_law, path_stats_experiment, scaling_experiment

# 参数配置（对应 0105_可视化_plan.md 第 2.1-2.7 节）
DEFAULT_PARAMS = {
    "t0": 1.0,
    "k_spring": 1.0,
    "mass": 1.0,
    "alpha": 0.5,
    "kT": 0.025,  # 室温（若 t0 = 1 eV 对应 ~290 K）
    "sigma": 0.1,  # 固定展宽
    "N_values": [20, 40, 80, 160, 320],
    "initial_energy_fraction": 0.9,  # 带顶附近
    "terminal_energy_fraction": 0.1,  # 带底附近
    "n_trajectories": 1000,
    "max_steps": 10000,
    "seed": 42,
}

# Fig3 对照工况：更高温度让吸收出现、路径统计不再“尖锐”
PATH_STATS_CONTRAST = {
    "name": "hiT",
    "kT": 0.5,
    "terminal_energy_fraction": DEFAULT_PARAMS["terminal_energy_fraction"],
    "n_trajectories": DEFAULT_PARAMS["n_trajectories"],
    "max_steps": DEFAULT_PARAMS["max_steps"],
    "seed": DEFAULT_PARAMS["seed"],
}


def _flatten_path_stats(prefix: str, path_stats_list: list) -> dict:
    """把 path_stats 列表展开为 np.savez 可写的扁平键值对。"""
    arrays = {}
    for i, stats in enumerate(path_stats_list):
        for key, val in stats.items():
            if isinstance(val, np.ndarray):
                arrays[f"{prefix}_{i}_{key}"] = val
            elif key == "step_sizes":
                arrays[f"{prefix}_{i}_{key}"] = val
            else:
                arrays[f"{prefix}_{i}_{key}"] = np.array(val)
    return arrays


def main():
    """运行主实验并保存结果。"""
    print("=" * 60)
    print("FGR 带内弛豫标度律验证实验")
    print("=" * 60)
    print("\n参数配置：")
    for key, val in DEFAULT_PARAMS.items():
        print(f"  {key:25s} = {val}")
    print("\n开始运行...")

    results = scaling_experiment(**DEFAULT_PARAMS)

    # 额外生成 Fig3 的对照路径统计（高温）
    contrast = PATH_STATS_CONTRAST
    contrast_path = path_stats_experiment(
        N_values=DEFAULT_PARAMS["N_values"],
        t0=DEFAULT_PARAMS["t0"],
        k_spring=DEFAULT_PARAMS["k_spring"],
        mass=DEFAULT_PARAMS["mass"],
        alpha=DEFAULT_PARAMS["alpha"],
        kT=contrast["kT"],
        sigma=DEFAULT_PARAMS["sigma"],
        initial_energy_fraction=DEFAULT_PARAMS["initial_energy_fraction"],
        n_trajectories=contrast["n_trajectories"],
        terminal_energy_fraction=contrast["terminal_energy_fraction"],
        max_steps=contrast["max_steps"],
        seed=contrast["seed"],
    )

    # 保存数据
    output_path = repo_root / "results" / "scaling_data.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 展开 path_stats 列表为单独的数组（供 npz 保存）
    path_stats_arrays = _flatten_path_stats("path_stats", results["path_stats"])
    path_stats_hit_arrays = _flatten_path_stats("path_stats_hiT", contrast_path["path_stats"])

    np.savez(
        output_path,
        N=results["N"],
        total_rate=results["total_rate"],
        char_step=results["char_step"],
        relax_time_ME=results["relax_time_ME"],
        relax_time_KMC=results["relax_time_KMC"],
        n_hops_mean=results["n_hops_mean"],
        **path_stats_arrays,
        **path_stats_hit_arrays,
        path_stats_hiT_kT=np.array(contrast["kT"]),
        path_stats_hiT_terminal_energy_fraction=np.array(contrast["terminal_energy_fraction"]),
        path_stats_hiT_n_trajectories=np.array(contrast["n_trajectories"]),
        **DEFAULT_PARAMS,  # 保存参数供复现
    )

    print(f"\n结果已保存到: {output_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("实验摘要")
    print("=" * 60)
    print(f"{'N':>6s} {'Γ':>10s} {'ΔE_avg':>10s} {'τ_ME':>10s} {'τ_KMC':>10s} {'<n_hop>':>10s}")
    print("-" * 60)
    for i, N in enumerate(results["N"]):
        print(
            f"{N:6d} {results['total_rate'][i]:10.4f} {results['char_step'][i]:10.4f} "
            f"{results['relax_time_ME'][i]:10.4f} {results['relax_time_KMC'][i]:10.4f} "
            f"{results['n_hops_mean'][i]:10.2f}"
        )

    # 简单的标度律拟合
    A_gamma, beta_gamma, R2_gamma = fit_power_law(results["N"], results["total_rate"])
    print(f"\nΓ(N) 拟合: Γ = {A_gamma:.3f} * N^{beta_gamma:.4f}  (R² = {R2_gamma:.4f})")

    # 特征步长收敛性
    char_step_mean = float(np.mean(results["char_step"]))
    char_step_std = float(np.std(results["char_step"]))
    char_step_cv = char_step_std / char_step_mean if char_step_mean > 0 else 0.0
    print(f"ΔE_avg 收敛: μ = {char_step_mean:.3f}, σ/μ = {char_step_cv:.4f}")

    # 弛豫时间收敛性
    tau_me_mean = float(np.mean(results["relax_time_ME"]))
    tau_me_std = float(np.std(results["relax_time_ME"]))
    tau_me_cv = tau_me_std / tau_me_mean if tau_me_mean > 0 else 0.0
    print(f"τ_ME 收敛:   μ = {tau_me_mean:.3f}, σ/μ = {tau_me_cv:.4f}")

    tau_kmc_mean = float(np.mean(results["relax_time_KMC"]))
    tau_kmc_std = float(np.std(results["relax_time_KMC"]))
    tau_kmc_cv = tau_kmc_std / tau_kmc_mean if tau_kmc_mean > 0 else 0.0
    print(f"τ_KMC 收敛:  μ = {tau_kmc_mean:.3f}, σ/μ = {tau_kmc_cv:.4f}")

    print("\n" + "=" * 60)
    print("实验完成！可以运行 scripts/generate_figures.py 生成图表。")
    print("=" * 60)


if __name__ == "__main__":
    main()
