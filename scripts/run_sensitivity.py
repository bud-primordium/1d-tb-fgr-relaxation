"""参数敏感性分析（单因子扫描）。

固定其他参数，每次只变一个参数：
- σ 依赖性
- kT 依赖性
- α 依赖性
"""

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.relaxation_analysis import scaling_experiment

# 基准参数
BASE_PARAMS = {
    "t0": 1.0,
    "k_spring": 1.0,
    "mass": 1.0,
    "alpha": 0.5,
    "kT": 0.025,
    "sigma": 0.1,
    "N_values": [20, 40, 80, 160, 320],
    "initial_energy_fraction": 0.9,
    "terminal_energy_fraction": 0.1,
    "n_trajectories": 1000,
    "max_steps": 10000,
    "seed": 42,
}

# 单因子扫描参数（扩展到 ≥5 点）
SIGMA_VALUES = [0.05, 0.07, 0.1, 0.15, 0.2]
KT_VALUES = [0.01, 0.02, 0.025, 0.05, 0.1]
ALPHA_VALUES = [0.2, 0.35, 0.5, 0.75, 1.0]


def run_sensitivity_sigma():
    """展宽 σ 依赖性。"""
    print("\n[1/3] 扫描 σ...")
    results_list = []
    for sigma in SIGMA_VALUES:
        print(f"  运行 σ = {sigma}...")
        params = BASE_PARAMS.copy()
        params["sigma"] = sigma
        res = scaling_experiment(**params)
        results_list.append({"sigma": sigma, "data": res})
    return results_list


def run_sensitivity_temperature():
    """温度 kT 依赖性。"""
    print("\n[2/3] 扫描 kT...")
    results_list = []
    for kT in KT_VALUES:
        print(f"  运行 kT = {kT}...")
        params = BASE_PARAMS.copy()
        params["kT"] = kT
        res = scaling_experiment(**params)
        results_list.append({"kT": kT, "data": res})
    return results_list


def run_sensitivity_alpha():
    """电声耦合 α 依赖性。"""
    print("\n[3/3] 扫描 α...")
    results_list = []
    for alpha in ALPHA_VALUES:
        print(f"  运行 α = {alpha}...")
        params = BASE_PARAMS.copy()
        params["alpha"] = alpha
        res = scaling_experiment(**params)
        results_list.append({"alpha": alpha, "data": res})
    return results_list


def main():
    """运行敏感性分析并保存。"""
    print("=" * 60)
    print("参数敏感性分析（单因子扫描）")
    print("=" * 60)

    # 运行三组扫描
    sigma_results = run_sensitivity_sigma()
    kT_results = run_sensitivity_temperature()
    alpha_results = run_sensitivity_alpha()

    # 保存
    output_path = repo_root / "results" / "sensitivity_data.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {}

    # 保存 σ 扫描
    for i, item in enumerate(sigma_results):
        save_dict[f"sigma_{i}_value"] = np.array(item["sigma"])
        save_dict[f"sigma_{i}_N"] = item["data"]["N"]
        save_dict[f"sigma_{i}_total_rate"] = item["data"]["total_rate"]
        save_dict[f"sigma_{i}_char_step"] = item["data"]["char_step"]

    # 保存 kT 扫描
    for i, item in enumerate(kT_results):
        save_dict[f"kT_{i}_value"] = np.array(item["kT"])
        save_dict[f"kT_{i}_N"] = item["data"]["N"]
        save_dict[f"kT_{i}_total_rate"] = item["data"]["total_rate"]
        save_dict[f"kT_{i}_char_step"] = item["data"]["char_step"]

    # 保存 α 扫描
    for i, item in enumerate(alpha_results):
        save_dict[f"alpha_{i}_value"] = np.array(item["alpha"])
        save_dict[f"alpha_{i}_N"] = item["data"]["N"]
        save_dict[f"alpha_{i}_total_rate"] = item["data"]["total_rate"]
        save_dict[f"alpha_{i}_char_step"] = item["data"]["char_step"]

    np.savez(output_path, **save_dict)

    print(f"\n敏感性分析结果已保存到: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
