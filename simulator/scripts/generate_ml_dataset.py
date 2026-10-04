#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_ml_dataset.py

为机器学习批量生成单晶旋转衍射模拟数据。

用法示例:
    python scripts/generate_ml_dataset.py --cif 2238482.cif --n 100 --seed 42

输出:
    在项目根目录下创建 ml_data/<cif_stem>/combined_matrix_0000.csv ...
    每个 CSV 文件与 4_demo_rotation_predictor.py 生成的 combined_matrix.csv 列一致。

随机参数范围（可在下方 RANDOM_RANGES 字典中修改）:
    - direct_dist:            100.0 ~ 300.0   mm      样品到探测器距离
    - orientation_matrix:     SO(3) 随机旋转矩阵，det = +1
    - num_images:             与 oscillation_range 一致（每帧 1°）
    - oscillation_range:      90.0  ~ 360.0   度      扫描范围
    - scan_start_deg:         -5.0  ~ 5.0     度      扫描起始角度
    - mosaicity_deg:          0.0   ~ 1.0     度      晶体镶嵌度
    - beam_divergence_deg:    0.0   ~ 0.1     度      光束发散度
    - domain_size_ang:        500.0 ~ 5000.0  Å       相干晶粒尺寸
    - d_min:                  0.5   ~ 1.0     Å       最小晶面间距
    - d_max:                  10.0  ~ 20.0    Å       最大晶面间距

非随机参数（固定值，来自 rotation_manual_config.yaml）:
    - fit2d 中除 direct_dist 外均固定
        * wavelength: 0.683858e-10  m
        * center_x:   2091.37       px
        * center_y:   2461.99       px
        * pixel_x:    75.0          μm
        * pixel_y:    75.0          μm
        * tilt:       0.0           deg
        * tilt_plan_rotation: 0.0   deg
        * max_shape:  [4148, 4362]
    - wavelength_angstrom: 0.683858  Å
    - rotation_axis:       [0.0, -1.0, 0.0]
    - rotation_angle_deg:  0.0
    - apply_lp_correction: true
    - intensity_distribution: "gaussian"

命令行可手动调整参数:
    - --n:                    生成样本数量
    - --seed:                 全局随机种子
    - --output-dir:           输出根目录
    - --config:               基础 YAML 配置文件路径
    - --use-fz / --no-use-fz: 是否开启基本区投影（默认开启）
    - --random-scan-start / --no-random-scan-start:
                              scan_start_deg 是否随机（默认开启，-5~5°）；
                              关闭后固定为 0°，样本多样性仅由初始取向控制
    - --intensity-threshold:  衍射斑相对强度阈值 I/max(I)，默认 1e-4
    - --show-crystal-info / --no-show-crystal-info:
                              是否在 CSV 头部显示晶体结构信息块（默认开启）
    - --decimals:             CSV 数据区小数位数（默认 8，对应 float64 数据）
    - --guide:                打印完整中文接口手册并退出（无需其他参数）
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import yaml

# 动态添加项目根目录到 sys.path
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _script_dir = os.getcwd()
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.rotation_predictor import RotationPredictor
from models.orientation_model import OrientationModel
from models.crystal_model import CrystalModel
from models.symmetry_convention import proper_laue_operations, canonicalize_orientation, CONVENTION
from models.lattice_label_convention import write_lattice_labels


# -----------------------------
# 随机参数范围配置
# 用户可按 ML 训练需求修改以下范围
# -----------------------------
RANDOM_RANGES = {
    "direct_dist": (100.0, 300.0),           # 样品到探测器距离，单位 mm
    "oscillation_range": (90.0, 360.0),      # 扫描范围，单位度
    "scan_start_deg": (-5.0, 5.0),           # 扫描起始角度，单位度
    "mosaicity_deg": (0.0, 1.0),             # 晶体镶嵌度，单位度
    "beam_divergence_deg": (0.0, 0.1),       # 光束发散度，单位度
    "domain_size_ang": (500.0, 5000.0),      # 相干晶粒尺寸，单位 Å
    "d_min": (0.5, 1.0),                     # 最小晶面间距，单位 Å
    "d_max": (10.0, 20.0),                   # 最大晶面间距，单位 Å
}

# -----------------------------
# 非随机参数：fit2d 中除 direct_dist 外均固定
# -----------------------------
FIXED_FIT2D = {
    "wavelength": 0.683858e-10,              # 波长，单位米
    "center_x": 2091.37,                     # 光束中心 fast 方向像素坐标
    "center_y": 2461.99,                     # 光束中心 slow 方向像素坐标
    "pixel_x": 75.0,                         # fast 方向像素尺寸，单位 μm
    "pixel_y": 75.0,                         # slow 方向像素尺寸，单位 μm
    "tilt": 0.0,                             # 探测器倾斜角，单位度
    "tilt_plan_rotation": 0.0,               # 倾斜面旋转角，单位度
    "max_shape": [4148, 4362],               # 探测器形状 [ny, nx]
}

# -----------------------------
# 其他非随机参数
# -----------------------------
FIXED_PARAMETERS = {
    "wavelength_angstrom": 0.683858,         # 波长，单位 Å
    "rotation_angle_deg": 0.0,               # 扫描起始前额外旋转角，单位度
    "rotation_axis": [0.0, -1.0, 0.0],       # 旋转轴，右手定则
    "apply_lp_correction": True,             # 是否应用洛伦兹-偏振校正
    "intensity_distribution": "gaussian",    # 强度在帧间分布类型
}

# 衍射斑相对强度阈值：仅保留 I/max(I) >= threshold 的衍射斑
# 可通过命令行 --intensity-threshold 手动修改
DEFAULT_INTENSITY_THRESHOLD = 1e-4


_GUIDE_TEXT = """
======================================================================
 generate_ml_dataset.py 接口手册（单 CIF 调试用）
======================================================================

【输入（必填）】
  --cif FILENAME            单个 CIF 文件名（在 test_representative_100_cifs 中查找）

【生成规模】
  --n INT                   生成样本数量（默认 10）
  --seed INT                全局随机种子（默认 None 不固定；设置后可复现）

【物理模型开关】
  --use-fz                  开启基本区投影（默认，等价 --no-use-fz 关闭）
  --no-use-fz               关闭基本区投影，保留完整 SO(3) 随机取向
  --random-scan-start       scan_start_deg 随机（默认，-5~5°）
  --no-random-scan-start    scan_start_deg 固定为 0°，
                            样本多样性仅由初始取向矩阵控制

【强度过滤】
  --intensity-threshold FLOAT
                            衍射斑相对强度阈值 I/max(I)（默认 1e-4）

【输出内容控制】
  --show-crystal-info       CSV 头部显示晶体结构信息块（默认，等价
                            --no-show-crystal-info 关闭）
  --decimals INT            CSV 数据区小数位数（默认 8，float64 数据）

【路径】
  --output-dir DIR          输出根目录（默认：项目根目录/ml_data）
  --config FILE             基础 YAML 配置文件
                            （默认：config/rotation_manual_config.yaml）

【帮助】
  --guide                   显示本手册并退出（无需其他参数）
  -h, --help                argparse 原生帮助

【常用命令】
  python generate_ml_dataset.py --cif 2238482.cif --n 100 --seed 42

【随机参数范围（在本文件 RANDOM_RANGES 字典中修改）】
  direct_dist        100.0~300.0 mm      oscillation_range   90.0~360.0 deg
  scan_start_deg     -5.0~5.0 deg        num_images          = oscillation_range（1°/帧）
  mosaicity_deg      0.0~1.0 deg         beam_divergence_deg 0.0~0.1 deg
  domain_size_ang    500.0~5000.0 A      d_min/d_max         0.5~1.0 / 10.0~20.0 A

【批量生产提示】
  多 CIF、多进程并行、逐帧图像序列输出请使用：
      scripts/generate_ml_dataset_parallel.py --guide
======================================================================
"""


def generate_random_orientation(seed=None, point_group=None):
    """
    生成一个随机的 SO(3) 取向矩阵，满足 det = +1。

    使用项目已有的 OrientationModel.random_orientation() 进行 Haar 测度采样，
    与 4_demo_rotation_predictor.py 中的随机取向生成方式保持一致。

    参数:
        seed (int, optional): 随机种子
        point_group (str, optional): 点群符号，若提供则投影到该点群的基本区

    返回:
        ndarray: 3×3 旋转矩阵
    """
    result = OrientationModel.random_orientation(
        output_format="matrix",
        seed=seed,
        point_group=point_group,
    )
    return result["u_matrix"]


def generate_random_config(base_config, cif_path, rng, point_group=None, use_fz=True,
                           intensity_threshold=DEFAULT_INTENSITY_THRESHOLD,
                           random_scan_start=True, orientation_seed=None, orientation_operations=None):
    """
    基于 base_config 生成一份随机参数配置。

    参数:
        base_config (dict): 从 rotation_manual_config.yaml 加载的基础配置
        cif_path (str): CIF 文件绝对路径
        rng (np.random.Generator): numpy 随机数生成器
        point_group (str, optional): 保留旧调用签名；FZ 使用实际 CIF 设置下的 Laue 操作
        use_fz (bool): 是否对随机取向做基本区投影，默认 True
        intensity_threshold (float): 衍射斑相对强度阈值 I/max(I)
        random_scan_start (bool): scan_start_deg 是否随机（默认 True）；
            False 时固定为 0°，样本多样性仅由初始取向矩阵控制

    返回:
        dict: 可用于 RotationPredictor 的完整配置
    """
    cfg = dict(base_config)

    # 1. 晶体输入：强制使用 CIF，移除手动晶胞参数避免冲突
    cfg["cif_path"] = cif_path
    cfg.pop("unit_cell_parameters", None)
    cfg.pop("space_group_symbol", None)

    # 2. 探测器参数：fit2d 中只有 direct_dist 随机
    fit2d = dict(FIXED_FIT2D)
    fit2d["direct_dist"] = float(rng.uniform(*RANDOM_RANGES["direct_dist"]))
    cfg["fit2d"] = fit2d

    # 3. 波长固定
    cfg["wavelength_angstrom"] = FIXED_PARAMETERS["wavelength_angstrom"]

    # 4. 初始取向矩阵：随机 SO(3) 矩阵，det = +1
    #    使用 rng.integers 派生子种子，保证整体可复现
    orient_seed = (
        int(rng.integers(0, 2**31))
        if orientation_seed is None and rng.bit_generator.seed_seq is not None
        else orientation_seed
    )
    if use_fz and orientation_operations is None:
        crystal = CrystalModel({"cif_path": cif_path})
        crystal.load()
        orientation_operations = proper_laue_operations(crystal._crystal_symmetry)
    orientation = generate_random_orientation(seed=orient_seed)
    if use_fz:
        orientation, _ = canonicalize_orientation(orientation, orientation_operations)
        cfg["_orientation_convention"] = CONVENTION
    else:
        cfg["_orientation_convention"] = "unrestricted-so3"
    cfg["orientation_matrix"] = orientation.tolist()
    cfg["_use_fz"] = use_fz  # 记录是否使用 FZ 投影，用于后续写入元数据
    cfg["_orientation_seed"] = orient_seed

    # 5. 移除其他取向输入方式，避免与 orientation_matrix 冲突
    for key in ["euler_angles", "angle1_deg", "angle2_deg", "rotation_angle", "initial_orientation"]:
        cfg.pop(key, None)

    # 6. 扫描参数：oscillation_range 随机，num_images 与其一致（每帧 1°）
    #    scan_start_deg 默认随机；random_scan_start=False 时固定为 0°，
    #    使所有样本角度轴对齐，样本多样性仅由初始取向矩阵控制
    oscillation_range = float(rng.uniform(*RANDOM_RANGES["oscillation_range"]))
    cfg["oscillation_range"] = oscillation_range
    cfg["num_images"] = int(round(oscillation_range))
    if random_scan_start:
        cfg["scan_start_deg"] = float(rng.uniform(*RANDOM_RANGES["scan_start_deg"]))
    else:
        cfg["scan_start_deg"] = 0.0

    # 7. 强度计算参数随机
    cfg["mosaicity_deg"] = float(rng.uniform(*RANDOM_RANGES["mosaicity_deg"]))
    cfg["beam_divergence_deg"] = float(rng.uniform(*RANDOM_RANGES["beam_divergence_deg"]))
    cfg["domain_size_ang"] = float(rng.uniform(*RANDOM_RANGES["domain_size_ang"]))

    # 8. 分辨率范围随机，且保证 d_min < d_max
    d_min = float(rng.uniform(*RANDOM_RANGES["d_min"]))
    d_max = float(rng.uniform(*RANDOM_RANGES["d_max"]))
    while d_max <= d_min:
        d_max = float(rng.uniform(*RANDOM_RANGES["d_max"]))
    cfg["d_min"] = d_min
    cfg["d_max"] = d_max

    # 9. 固定参数
    cfg["rotation_angle_deg"] = FIXED_PARAMETERS["rotation_angle_deg"]
    cfg["rotation_axis"] = FIXED_PARAMETERS["rotation_axis"]
    cfg["apply_lp_correction"] = FIXED_PARAMETERS["apply_lp_correction"]
    cfg["intensity_distribution"] = FIXED_PARAMETERS["intensity_distribution"]
    cfg["_intensity_threshold"] = intensity_threshold  # 记录强度阈值，用于过滤与元数据

    return cfg


def build_combined_matrix(predictor, results, cfg=None):
    """
    从 predict_with_intensity 结果构造 combined_matrix。

    列与 4_demo_rotation_predictor.py 输出一致：
        x_px, y_px, angle, I_LP, h, k, l, qx, qy, qz

    参数:
        predictor (RotationPredictor): 已执行预测的预测器实例
        results (dict): predict_with_intensity 返回的结果字典
        cfg (dict, optional): 配置字典，用于读取强度阈值

    返回:
        ndarray: N×10 的 combined_matrix
    """
    predicted_hkls = results["predicted_hkls"].T      # (N, 3)
    predicted_px = results["predicted_px"]            # (N, 3)
    predicted_intensity_lp = results["predicted_intensity_lp"]  # (N,)
    predicted_q_hkls = results["predicted_q_hkls"]    # (N, 3)

    # 强度归一化到 1e4，并根据相对强度阈值过滤衍射斑
    if len(predicted_intensity_lp) > 0 and np.max(predicted_intensity_lp) > 0:
        max_intensity = np.max(predicted_intensity_lp)
        predicted_intensity_lp = predicted_intensity_lp / max_intensity * 1e4

        threshold = DEFAULT_INTENSITY_THRESHOLD
        if cfg is not None:
            threshold = cfg.get("_intensity_threshold", DEFAULT_INTENSITY_THRESHOLD)
        mask = predicted_intensity_lp >= threshold * 1e4
        predicted_intensity_lp = predicted_intensity_lp[mask]
        predicted_px = predicted_px[mask]
        predicted_hkls = predicted_hkls[mask]
        predicted_q_hkls = predicted_q_hkls[mask]

    combined_matrix = np.hstack([
        predicted_px,                                    # (N, 3)
        predicted_intensity_lp[:, np.newaxis],           # (N, 1)
        predicted_hkls,                                  # (N, 3)
        predicted_q_hkls,                                # (N, 3)
    ])

    # DIALS xyzcal.px[:, 2] is a zero-based continuous array index.
    frame_width = predictor.oscillation_range / predictor.num_images
    combined_matrix[:, 2] = (
        predictor.scan_start_deg + combined_matrix[:, 2] * frame_width
    )
    # 按角度排序
    combined_matrix = combined_matrix[np.argsort(combined_matrix[:, 2])]

    return combined_matrix


def build_config_comment(cfg, index, total, point_group=None, output_mode=None, crystal_info=None):
    """
    将本次生成使用的所有参数格式化为 CSV 文件开头的注释行。

    参数:
        cfg (dict): 实际传入 RotationPredictor 的配置
        index (int): 当前样本序号（0-based）
        total (int): 样本总数
        point_group (str, optional): 晶体点群符号
        output_mode (str, optional): 输出模式，"integrated"（积分斑点表）
            或 "per_frame"（逐帧展开）；None 时按旧格式输出（不含 Output
            mode 行，列名为积分斑点表格式），保持向后兼容。
        crystal_info (dict, optional): CrystalModel.get_crystal_info() 返回的
            晶体结构信息字典；提供时在 CIF path 之后插入晶体信息块，
            None 时不插入（旧调用行为不变）。

    返回:
        str: 以 '#' 开头的多行注释字符串
    """
    U = np.array(cfg["orientation_matrix"])
    mode_lines = []
    if output_mode is not None:
        mode_lines = [f"# Output mode: {output_mode}"]
    if output_mode == "per_frame":
        columns_line = "x_px,y_px,frame_index,angle,I_frame,I_total,h,k,l,qx,qy,qz"
    else:
        columns_line = "x_px,y_px,angle,I_LP,h,k,l,qx,qy,qz"

    # 晶体结构信息块（来自 CIF，由 CrystalModel.get_crystal_info() 提供）
    crystal_lines = []
    if crystal_info is not None:
        ci = crystal_info
        uc = ci["unit_cell"]
        rc = ci["reciprocal_cell"]
        crystal_lines = ["# --- Crystal structure info (from CIF) ---"]
        if ci["chemical_formula"] is not None:
            crystal_lines.append(f"# Chemical formula: {ci['chemical_formula']}")
            crystal_lines.append(f"# Atoms per unit cell: {ci['n_atoms_cell']}")
        crystal_lines += [
            f"# Space group: {ci['space_group_symbol']} (No. {ci['space_group_number']})",
            f"# Hall symbol: {ci['hall_symbol']}",
            f"# Crystal system: {ci['crystal_system']}",
            f"# Lattice centring: {ci['lattice_centring']}",
            f"# Unit cell (A, deg): a={uc[0]:.6f} b={uc[1]:.6f} c={uc[2]:.6f} "
            f"alpha={uc[3]:.6f} beta={uc[4]:.6f} gamma={uc[5]:.6f}",
            f"# Unit cell volume (A^3): {ci['unit_cell_volume']:.6f}",
            f"# Reciprocal cell (A^-1, deg): a*={rc[0]:.6f} b*={rc[1]:.6f} c*={rc[2]:.6f} "
            f"alpha*={rc[3]:.6f} beta*={rc[4]:.6f} gamma*={rc[5]:.6f}",
        ]

    lines = [
        f"# Sample index: {index + 1} / {total}",
        f"# CIF path: {cfg.get('cif_path', 'N/A')}",
        f"# RNG version: {cfg.get('_rng_version', 'legacy')}",
        f"# Master seed: {cfg.get('_master_seed', 'None')}",
        f"# Sample seed: {cfg.get('_sample_seed', 'None')}",
        f"# Orientation seed: {cfg.get('_orientation_seed', 'None')}",
    ] + crystal_lines + [
        f"# Point group: {point_group if point_group else 'N/A'}",
        f"# Orientation convention: {cfg.get('_orientation_convention', 'unspecified')}",
        "# Angle convention: dials-zero-based-array-index-v1",
        f"# Use fundamental zone projection: {cfg.get('_use_fz', True)}",
    ] + mode_lines + [
        f"# Intensity threshold (I/max(I)): {cfg.get('_intensity_threshold', DEFAULT_INTENSITY_THRESHOLD):.2e}",
        f"# direct_dist (mm): {cfg['fit2d']['direct_dist']:.6f}",
        f"# center_x (px): {cfg['fit2d']['center_x']:.6f}",
        f"# center_y (px): {cfg['fit2d']['center_y']:.6f}",
        f"# pixel_x (um): {cfg['fit2d']['pixel_x']:.6f}",
        f"# pixel_y (um): {cfg['fit2d']['pixel_y']:.6f}",
        f"# wavelength (A): {cfg['wavelength_angstrom']:.6f}",
        f"# oscillation_range (deg): {cfg['oscillation_range']:.6f}",
        f"# num_images: {cfg['num_images']}",
        f"# scan_start_deg (deg): {cfg['scan_start_deg']:.6f}",
        f"# rotation_axis: {cfg['rotation_axis']}",
        f"# rotation_angle_deg (deg): {cfg['rotation_angle_deg']:.6f}",
        f"# mosaicity_deg (deg): {cfg['mosaicity_deg']:.6f}",
        f"# beam_divergence_deg (deg): {cfg['beam_divergence_deg']:.6f}",
        f"# domain_size_ang (A): {cfg['domain_size_ang']:.6f}",
        f"# d_min (A): {cfg['d_min']:.6f}",
        f"# d_max (A): {cfg['d_max']:.6f}",
        f"# apply_lp_correction: {cfg['apply_lp_correction']}",
        f"# intensity_distribution: {cfg['intensity_distribution']}",
        f"# orientation_matrix row 0: {U[0, 0]:+.8f} {U[0, 1]:+.8f} {U[0, 2]:+.8f}",
        f"# orientation_matrix row 1: {U[1, 0]:+.8f} {U[1, 1]:+.8f} {U[1, 2]:+.8f}",
        f"# orientation_matrix row 2: {U[2, 0]:+.8f} {U[2, 1]:+.8f} {U[2, 2]:+.8f}",
        f"# Columns: {columns_line}",
    ]
    return "\n".join(lines) + "\n"


def parse_args():
    """解析命令行参数。"""
    # --guide 优先于 argparse 必填校验：直接打印接口手册并退出
    if "--guide" in sys.argv:
        print(_GUIDE_TEXT)
        sys.exit(0)
    parser = argparse.ArgumentParser(
        description="为机器学习生成单晶旋转衍射模拟数据集"
    )
    parser.add_argument(
        "--cif",
        required=True,
        help="CIF 文件名，例如 2238482.cif（将自动在 test_representative_100_cifs 中查找）",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="生成样本数量 N（默认 10）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="全局随机种子；设置后整个数据集可复现",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出根目录（默认：项目根目录/ml_data）",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="基础 YAML 配置文件路径（默认：config/rotation_manual_config.yaml）",
    )
    parser.add_argument(
        "--use-fz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否对随机取向做基本区投影（默认开启）",
    )
    parser.add_argument(
        "--intensity-threshold",
        type=float,
        default=DEFAULT_INTENSITY_THRESHOLD,
        help="衍射斑相对强度阈值 I/max(I)，仅保留 >= threshold 的衍射斑（默认 1e-4）",
    )
    parser.add_argument(
        "--show-crystal-info",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在 CSV 头部显示晶体结构信息块（默认开启）",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=8,
        help="CSV 数据区小数位数（默认 8，对应 float64 数据）",
    )
    parser.add_argument(
        "--random-scan-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scan_start_deg 是否随机（默认开启，-5~5°）；关闭后固定为 0°，样本多样性仅由初始取向控制",
    )
    parser.add_argument(
        "--export-primitive-labels", action="store_true",
        help="另存联合 primitive Niggli 晶胞/取向/HKL 标签及结构基底变换；原 CSV 不改写",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 项目路径定位
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # CIF 文件路径：默认在项目根目录同级的 test_representative_100_cifs 中查找
    cifs_dir = os.path.join(project_root, "..", "test_representative_100_cifs")
    cif_path = os.path.abspath(os.path.join(cifs_dir, args.cif))
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF 文件不存在：{cif_path}")

    # 基础配置文件路径
    if args.config is None:
        args.config = os.path.join(project_root, "config", "rotation_manual_config.yaml")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"YAML 配置文件不存在：{args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # 输出目录：<output_dir>/<cif_stem>/
    cif_stem = Path(args.cif).stem
    if args.output_dir is None:
        output_dir = os.path.join(project_root, "ml_data", cif_stem)
    else:
        output_dir = os.path.join(args.output_dir, cif_stem)
    os.makedirs(output_dir, exist_ok=True)

    # 全局随机种子
    if args.seed is not None:
        np.random.seed(args.seed)

    # 自动推导点群，用于 FZ 投影与信息展示
    tmp_cm = CrystalModel({"cif_path": cif_path})
    tmp_cm.load()
    point_group = tmp_cm.get_point_group_symbol()
    # 晶体结构信息（默认开启，写入 CSV 头部；同一 CIF 只需提取一次）
    crystal_info = tmp_cm.get_crystal_info() if args.show_crystal_info else None
    print(f"处理 CIF: {args.cif}")
    print(f"  空间群: {tmp_cm.get_space_group_symbol()}")
    print(f"  点群:   {point_group}")
    print(f"  FZ 投影: {'开启' if args.use_fz else '关闭'}")
    print(f"  输出目录: {output_dir}")
    print(f"  生成数量: {args.n}")
    print(f"  随机种子: {args.seed if args.seed is not None else 'None（不固定）'}")
    print("-" * 60)

    # 创建主随机数生成器
    rng = np.random.default_rng(args.seed)

    # 生成 N 个样本
    for i in range(args.n):
        # 每个样本使用独立的子种子，保证单样本可复现且样本间不相关
        sample_seed = int(rng.integers(0, 2**31))
        sample_rng = np.random.default_rng(sample_seed)

        cfg = generate_random_config(
            base_config, cif_path, sample_rng,
            point_group=point_group, use_fz=args.use_fz,
            orientation_operations=proper_laue_operations(tmp_cm._crystal_symmetry),
            intensity_threshold=args.intensity_threshold,
            random_scan_start=args.random_scan_start,
        )
        cfg["_rng_version"] = "legacy"
        cfg["_master_seed"] = args.seed
        cfg["_sample_seed"] = sample_seed

        predictor = RotationPredictor(cfg)
        results = predictor.predict_with_intensity()

        combined_matrix = build_combined_matrix(predictor, results, cfg=cfg)

        output_path = os.path.join(output_dir, f"combined_matrix_{i:04d}.csv")
        comment = build_config_comment(
            cfg, i, args.n, point_group=point_group, crystal_info=crystal_info, output_mode="integrated"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(comment)
            np.savetxt(f, combined_matrix, delimiter=",",
                       fmt=f"%.{args.decimals}f", comments="")
        if args.export_primitive_labels:
            write_lattice_labels(output_path, tmp_cm.get_crystal_symmetry(), cif_path, overwrite=True)
        print(
            f"[{i+1:04d}/{args.n:04d}] "
            f"direct_dist={cfg['fit2d']['direct_dist']:.2f}mm  "
            f"range={cfg['oscillation_range']:.1f}°  "
            f"num_images={cfg['num_images']}  "
            f"start={cfg['scan_start_deg']:.2f}°  "
            f"FZ={cfg['_use_fz']}"
        )
        print(
            f"             "
            f"mosaicity={cfg['mosaicity_deg']:.3f}°  "
            f"divergence={cfg['beam_divergence_deg']:.4f}°  "
            f"domain={cfg['domain_size_ang']:.1f}Å  "
            f"d_min={cfg['d_min']:.3f}Å  "
            f"d_max={cfg['d_max']:.3f}Å  "
            f"I_thr={cfg['_intensity_threshold']:.2e}  "
            f"reflections={len(combined_matrix):5d}"
        )

    print("-" * 60)
    print(f"数据集生成完成：{output_dir}")
    print(f"共生成 {args.n} 个文件：combined_matrix_0000.csv ~ combined_matrix_{args.n-1:04d}.csv")


if __name__ == "__main__":
    main()


# 使用示例：
#   1) 默认开启基本区投影，生成 100 个样本：
#      python scripts/generate_ml_dataset.py --cif 2238482.cif --n 100 --seed 42
#
#   2) 显式开启基本区投影：
#      python scripts/generate_ml_dataset.py --cif 2238482.cif --n 100 --seed 42 --use-fz
#
#   3) 关闭基本区投影，保留完整 SO(3) 随机取向：
#      python scripts/generate_ml_dataset.py --cif 2238482.cif --n 100 --seed 42 --no-use-fz
#
#   4) 指定输出目录：
#      python scripts/generate_ml_dataset.py --cif 2238482.cif --n 100 --output-dir D:\ml_data
#
#   5) 调整衍射斑强度阈值（例如保留千分之一以上的斑点）：
#      python scripts/generate_ml_dataset.py --cif 2238482.cif --n 100 --seed 42 --intensity-threshold 1e-3
