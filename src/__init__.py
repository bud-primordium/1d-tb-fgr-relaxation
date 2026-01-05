"""FGR 弛豫模拟包。

基于一维紧束缚模型验证带内弛豫的 O(1) 标度律。

模块结构：
- lattice: 格点与动量网格工具（复用自 nac_tb_demo）
- tb_electron_1band: 单带 TB 模型（复用自 nac_tb_demo）
- electron_phonon: 电声耦合矩阵元（复用自 nac_tb_demo）
- phonon_1atom: 单原子链声子（复用自 nac_tb_demo）
- fermi_golden_rule: FGR 散射速率计算（新增）
- master_equation: 主方程求解（新增）
- kinetic_monte_carlo: KMC/Gillespie 模拟（新增）
- relaxation_analysis: 弛豫分析工具（新增）
"""

from .lattice import k_grid, q_grid, pbc_index, phase_factors
from .tb_electron_1band import build_hamiltonian, diagonalize, bloch_state, dispersion
from .electron_phonon import g_monatomic, dh_dq_monatomic
from .phonon_1atom import dispersion_monatomic
