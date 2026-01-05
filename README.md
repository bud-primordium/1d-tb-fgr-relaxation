# FGR 带内弛豫数值模拟

基于一维紧束缚模型验证带内弛豫的 O(1) 标度律。

## 项目背景

本项目是 [1d-tb-nac-scaling](https://github.com/bud-primordium/1d-tb-nac-scaling) 的延续，旨在讨论：

> 带内弛豫过程中，当系统尺寸 N 增大引入大量中间态时，多步路径数爆炸是否导致弛豫时间发散？

理论预期（非相干速率网络框架）：
- 总散射率 Γ ~ O(1)
- 特征步长 ΔE_avg 收敛到有限值
- 弛豫时间在热力学极限收敛

## 单位约定（无量纲）

代码内部采用无量纲化约定：ℏ = k_B = t₀ = 1。

- 温度使用 `kT = k_B T / t₀`（无量纲）；若取 `t₀ ≈ 1 eV`，室温 `300 K` 对应 `kT ≈ 0.025`
- 速率单位为 `t₀/ℏ`，时间单位为 `ℏ/t₀`（数值通常为 O(1)）

## 项目结构

```
fgr_tb_demo/
├── src/
│   ├── lattice.py              # [复用] 格点工具
│   ├── tb_electron_1band.py    # [复用] 单带 TB
│   ├── electron_phonon.py      # [复用] 电声耦合
│   ├── phonon_1atom.py         # [复用] 声子色散
│   ├── fermi_golden_rule.py    # [新增] FGR 速率
│   ├── master_equation.py      # [新增] 主方程
│   ├── kinetic_monte_carlo.py  # [新增] KMC
│   └── relaxation_analysis.py  # [新增] 分析
├── scripts/                    # 可视化脚本
├── tests/                      # 单元测试
├── report/                     # LaTeX 报告
└── results/                    # 输出图片
```

## 核心公式

Fermi 黄金定则（FGR）：

$$
W_{k \to k'} = \frac{2\pi}{\hbar} \sum_{q,\nu} |g(k,q)|^2 \left[ (n_{q\nu}+1) \delta(E_{k'} - E_k + \hbar\omega_{q\nu}) + n_{q\nu} \delta(E_{k'} - E_k - \hbar\omega_{q\nu}) \right]
$$
