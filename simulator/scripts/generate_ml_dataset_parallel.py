#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_ml_dataset_parallel.py

为机器学习批量生成单晶旋转衍射模拟数据（多进程并行版本）。

与 generate_ml_dataset.py 的区别：
    - 支持 --cif-dir 扫描整个 CIF 目录批量生成（也支持 --cif 单文件验证）
    - 多进程并行（ProcessPoolExecutor），充分利用多核 CPU
    - 同一 CIF 的 CrystalModel 全集（含结构因子强度表）在 worker 内只构建一次，
      多个随机取向共享，跳过重复的 CIF 解析与结构因子计算
    - CSV 数据区以 float32 精度写出，文件体积约为 float64 的一半

随机参数范围（与 generate_ml_dataset.py 的 RANDOM_RANGES 一致，可在该文件中修改）:
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

非随机参数（固定值，来自 generate_ml_dataset.py 的 FIXED_FIT2D / FIXED_PARAMETERS）:
    - fit2d 中除 direct_dist 外均固定
    - wavelength_angstrom: 0.683858  Å
    - rotation_axis:       [0.0, -1.0, 0.0]
    - rotation_angle_deg:  0.0
    - apply_lp_correction: true
    - intensity_distribution: "gaussian"

命令行参数:
    - --cif:                  单个 CIF 文件名（验证用，与 --cif-dir 互斥）
    - --cif-dir:              CIF 目录（批量用，默认 test_representative_100_cifs）
    - --n-per-cif:            每个 CIF 生成的样本数（默认 1000）
    - --chunk-size:           兼容旧命令；新版按 CIF 调度，不再按样本切块
    - --workers:              并行进程数（本机基准推荐且默认 8）
    - --seed:                 全局随机种子；设置后整个数据集可复现
    - --use-fz / --no-use-fz: 是否开启基本区投影（默认开启）
    - --random-scan-start / --no-random-scan-start:
                              scan_start_deg 是否随机（默认开启，-5~5°）；
                              关闭后固定为 0°，样本多样性仅由初始取向控制
    - --intensity-threshold:  衍射斑相对强度阈值 I/max(I)，默认 1e-4
    - --output-mode:          输出模式：integrated（积分斑点表，默认）、
                              per_frame（逐帧图像序列）、both（两者都生成）
    - --show-crystal-info / --no-show-crystal-info:
                              是否在 CSV 头部显示晶体结构信息块（默认开启）
    - --decimals:             CSV 数据区小数位数（默认 6，对应 float32 数据）
    - --output-dir:           输出根目录（默认：项目根目录/ml_data）
    - --config:               基础 YAML 配置文件路径
    - --guide:                打印完整中文接口手册并退出（无需其他参数）

使用示例（PowerShell）:
    # 验证：1 个 CIF × 10 个随机取向
    python crystal_diffraction_simulation_program\\scripts\\generate_ml_dataset_parallel.py `
        --cif 2238482.cif `
        --n-per-cif 10 `
        --seed 42

    # 全量：100 个 CIF × 1000 个随机取向
    python crystal_diffraction_simulation_program\\scripts\\generate_ml_dataset_parallel.py `
        --cif-dir test_representative_100_cifs `
        --n-per-cif 1000 `
        --workers 8 `
        --seed 42 `
        --use-fz `
        --intensity-threshold 1e-4

    # 逐帧图像序列模式
    python crystal_diffraction_simulation_program\\scripts\\generate_ml_dataset_parallel.py `
        --cif-dir test_representative_100_cifs `
        --n-per-cif 1000 `
        --workers 8 `
        --seed 42 `
        --output-mode per_frame

输出:
    <output_dir>/<cif_stem>/combined_matrix_0000.csv ...   （integrated 模式）
    <output_dir>/<cif_stem>/per_frame_matrix_0000.csv ...  （per_frame 模式）
    每个 CSV 开头为参数元数据注释（# 开头）。
    integrated 模式数据列（每斑一行，总积分强度）:
        x_px,y_px,angle,I_LP,h,k,l,qx,qy,qz
    per_frame 模式数据列（每斑 × 每记录帧一行，逐帧分摊强度）:
        x_px,y_px,frame_index,angle,I_frame,I_total,h,k,l,qx,qy,qz
        其中 frame_index 为 0-based 帧号；I_frame = 帧权重 × I_total；
        仅写出 I_frame >= intensity_threshold × max(I) 的行，
        高斯 rocking curve 长尾中低于阈值的帧自然被截断。
"""

import os
import sys
import time
import argparse
import glob
import json
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# 多进程外层已经并行，禁止 BLAS/OpenMP 在每个 worker 内再次扩张线程。
for _thread_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")
os.environ.setdefault(
    "NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "xrd_predict_numba_cache")
)
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "xrd_predict_matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xrd_predict_cache")
)

import numpy as np
import yaml

# 动态添加 scripts 目录与项目根目录到 sys.path
# （Windows 多进程 spawn 模式下 worker 会重新 import 本模块，此处理必须模块级执行）
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _script_dir = os.getcwd()
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
for _p in (_script_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_ml_dataset import (
    RANDOM_RANGES,
    DEFAULT_INTENSITY_THRESHOLD,
    generate_random_config,
    build_combined_matrix,
    build_config_comment,
)
from models.rotation_predictor import RotationPredictor
from models.lattice_label_convention import write_lattice_labels, labels_complete
from models.crystal_model import CrystalModel
from models.symmetry_convention import proper_laue_operations


_GUIDE_TEXT = """
======================================================================
 generate_ml_dataset_parallel.py 接口手册（多进程并行批量生成）
======================================================================

【输入（二选一，必填）】
  --cif FILENAME            单个 CIF 文件名（验证用，在 CIF 目录中查找）
  --cif-dir DIR             CIF 目录（批量用）
                            默认：项目根目录同级的 test_representative_100_cifs

【生成规模】
  --n-per-cif INT           每个 CIF 生成的样本数（默认 1000）
  --chunk-size INT          兼容旧命令；新版按 CIF 调度，不再按样本切块
  --workers INT             并行进程数（本机基准推荐且默认 8；
                            设为 1 表示串行，便于调试与对比验证）
  --seed INT                全局随机种子（默认 None 不固定；
                            设置后整个数据集可复现，与并行进程数无关）

【物理模型开关】
  --use-fz                  开启基本区投影（默认，等价 --no-use-fz 关闭）
  --no-use-fz               关闭基本区投影，保留完整 SO(3) 随机取向
  --random-scan-start       scan_start_deg 随机（默认，-5~5°）
  --no-random-scan-start    scan_start_deg 固定为 0°，
                            样本多样性仅由初始取向矩阵控制

【强度过滤】
  --intensity-threshold FLOAT
                            衍射斑相对强度阈值 I/max(I)（默认 1e-4）；
                            提高可减少每文件行数、压缩磁盘体积

【输出模式（三选一）】
  --output-mode integrated  积分斑点表（默认）：每斑一行，总积分强度
                            文件 combined_matrix_XXXX.csv
                            列：x_px,y_px,angle,I_LP,h,k,l,qx,qy,qz
  --output-mode per_frame   逐帧图像序列：每斑 × 每记录帧一行
                            文件 per_frame_matrix_XXXX.csv
                            列：x_px,y_px,frame_index,angle,I_frame,
                                I_total,h,k,l,qx,qy,qz
  --output-mode both        两种文件同时生成

【输出内容控制】
  --show-crystal-info       CSV 头部显示晶体结构信息块（默认，等价
                            --no-show-crystal-info 关闭）
  --decimals INT            CSV 数据区小数位数（默认 6，float32 数据）

【路径】
  --output-dir DIR          输出根目录（默认：项目根目录/ml_data）
  --config FILE             基础 YAML 配置文件
                            （默认：config/rotation_manual_config.yaml）

【帮助】
  --guide                   显示本手册并退出（无需其他参数）
  -h, --help                argparse 原生帮助

【常用命令】
  # 验证：1 个 CIF × 10 个随机取向
  python generate_ml_dataset_parallel.py --cif 2238482.cif --n-per-cif 10 --seed 42

  # 全量：100 个 CIF × 1000 个随机取向
  python generate_ml_dataset_parallel.py --cif-dir test_representative_100_cifs \
      --n-per-cif 1000 --workers 8 --seed 42 --use-fz \
      --intensity-threshold 1e-4 --output-mode integrated

【随机参数范围（在 generate_ml_dataset.py 的 RANDOM_RANGES 中修改）】
  direct_dist        100.0~300.0 mm      oscillation_range   90.0~360.0 deg
  scan_start_deg     -5.0~5.0 deg        num_images          = oscillation_range（1°/帧）
  mosaicity_deg      0.0~1.0 deg         beam_divergence_deg 0.0~0.1 deg
  domain_size_ang    500.0~5000.0 A      d_min/d_max         0.5~1.0 / 10.0~20.0 A
======================================================================
"""


def build_per_frame_matrix(predictor, results, cfg):
    """
    从 predict_with_intensity 结果构造逐帧展开矩阵（per_frame 输出模式）。

    利用 RotationPredictor 已计算的 per_spot_frame_weights（rocking curve
    逐帧积分权重，Σw = 1），将每个斑点按帧分摊：
        I_frame = w_i × I_total
    仅保留 I_frame >= intensity_threshold × max(I_total) 的 (斑点, 帧) 行，
    该单一条件同时实现弱斑点过滤与高斯长尾帧的自然截断。

    参数:
        predictor (RotationPredictor): 已执行预测的预测器实例
        results (dict): predict_with_intensity 返回的结果字典
        cfg (dict): 配置字典，用于读取强度阈值

    返回:
        ndarray: M×12 矩阵，按 frame_index 主序、angle 次序排序，列为：
            x_px, y_px, frame_index, angle, I_frame, I_total, h, k, l, qx, qy, qz
    """
    predicted_px = results["predicted_px"]               # (N, 3)
    predicted_hkls = results["predicted_hkls"].T         # (N, 3)
    predicted_q_hkls = results["predicted_q_hkls"]       # (N, 3)
    i_total = results["predicted_intensity_lp"]          # (N,)
    weights = results["per_spot_frame_weights"]          # (N, num_images)

    n_rows = 0
    out = np.zeros((n_rows, 12), dtype=float)
    if len(i_total) == 0 or np.max(i_total) <= 0:
        return out

    # 强度归一化到 1e4（与 build_combined_matrix 同一约定）
    i_total = i_total / np.max(i_total) * 1e4

    threshold = DEFAULT_INTENSITY_THRESHOLD
    if cfg is not None:
        threshold = cfg.get("_intensity_threshold", DEFAULT_INTENSITY_THRESHOLD)

    # 逐帧强度；I_frame >= 阈值 同时完成斑点过滤与长尾截断
    i_frame_all = weights * i_total[:, np.newaxis]       # (N, num_images)
    spot_idx, frame_idx = np.nonzero(i_frame_all >= threshold * 1e4)
    if len(spot_idx) == 0:
        return out

    # 斑点中心角度（与 combined_matrix 第 3 列同一换算）
    frame_width = predictor.oscillation_range / predictor.num_images
    angle = predictor.scan_start_deg + predicted_px[:, 2] * frame_width

    out = np.column_stack([
        predicted_px[spot_idx, 0],                       # x_px
        predicted_px[spot_idx, 1],                       # y_px
        frame_idx.astype(float),                         # frame_index (0-based)
        angle[spot_idx],                                 # angle（中心角度）
        i_frame_all[spot_idx, frame_idx],                # I_frame
        i_total[spot_idx],                               # I_total
        predicted_hkls[spot_idx],                        # h, k, l
        predicted_q_hkls[spot_idx],                      # qx, qy, qz
    ])

    # 按 frame_index 主序、angle 次序排序，便于按帧切片重建图像
    order = np.lexsort((out[:, 3], out[:, 2]))
    return out[order]


def build_per_frame_matrix_sparse(predictor, results, cfg,
                                  max_block_elements=2_000_000):
    """分块生成稀疏逐帧表，不保留完整 N_reflections × N_frames 矩阵。"""
    predicted_px = results["predicted_px"]
    predicted_hkls = results["predicted_hkls"].T
    predicted_q_hkls = results["predicted_q_hkls"]
    i_total = np.asarray(results["predicted_intensity_lp"], dtype=float)
    if len(i_total) == 0 or np.max(i_total) <= 0:
        return np.zeros((0, 12), dtype=float)

    i_total = i_total / np.max(i_total) * 1e4
    threshold = cfg.get("_intensity_threshold", DEFAULT_INTENSITY_THRESHOLD) * 1e4
    frame_width = predictor.oscillation_range / predictor.num_images
    angle = predictor.scan_start_deg + predicted_px[:, 2] * frame_width
    blocks = []

    for bs, be, weights in predictor._iter_frame_weight_blocks(
        predicted_px, results["rocking_widths_deg"],
        max_block_elements=max_block_elements,
    ):
        i_frame = weights * i_total[bs:be, np.newaxis]
        local_spot, frame_idx = np.nonzero(i_frame >= threshold)
        if len(local_spot) == 0:
            continue
        spot_idx = local_spot + bs
        blocks.append(np.column_stack([
            predicted_px[spot_idx, 0],
            predicted_px[spot_idx, 1],
            frame_idx.astype(float),
            angle[spot_idx],
            i_frame[local_spot, frame_idx],
            i_total[spot_idx],
            predicted_hkls[spot_idx],
            predicted_q_hkls[spot_idx],
        ]))

    if not blocks:
        return np.zeros((0, 12), dtype=float)
    out = np.concatenate(blocks, axis=0)
    return out[np.lexsort((out[:, 3], out[:, 2]))]


def _atomic_write_text(path, text_value):
    """在目标目录写临时文件，落盘后原子替换。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text_value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_matrix(path, comment, matrix, decimals):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(comment)
            np.savetxt(handle, matrix, delimiter=",", fmt=f"%.{decimals}f", comments="")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _csv_complete(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return any(line.startswith("# Columns:") for line in handle)
    except OSError:
        return False


def _required_outputs(output_dir, index, mode):
    paths = []
    if mode in ("integrated", "both"):
        paths.append(os.path.join(output_dir, f"combined_matrix_{index:04d}.csv"))
    if mode in ("per_frame", "both"):
        paths.append(os.path.join(output_dir, f"per_frame_matrix_{index:04d}.csv"))
    return paths


def _code_version(project_root):
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(project_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _sample_random_state(task, sample_index):
    """返回参数 RNG、可记录的 sample seed 与独立 orientation seed。"""
    if task["rng_version"] == "legacy":
        sample_seed = task["legacy_sample_seeds"][sample_index]
        return np.random.default_rng(sample_seed), sample_seed, None, [sample_index]

    sample_ss = np.random.SeedSequence(
        task["master_seed"], spawn_key=(task["cif_index"], sample_index)
    )
    parameter_ss, orientation_ss = sample_ss.spawn(2)
    sample_seed = int(parameter_ss.generate_state(1, dtype=np.uint64)[0])
    orientation_seed = int(orientation_ss.generate_state(1, dtype=np.uint32)[0])
    return (
        np.random.default_rng(parameter_ss), sample_seed, orientation_seed,
        [task["cif_index"], sample_index],
    )


def _manifest_record(task, index, status, sample_seed, orientation_seed,
                     spawn_key, elapsed, error=None, n_rows=None, cfg=None):
    record = {
        "cif": task["cif_stem"],
        "cif_path": task["cif_path"],
        "sample_index": index,
        "status": status,
        "rng_version": task["rng_version"],
        "master_seed": task["master_seed"],
        "sample_seed": sample_seed,
        "orientation_seed": orientation_seed,
        "spawn_key": spawn_key,
        "code_version": task["code_version"],
        "elapsed_seconds": elapsed,
        "row_count": n_rows,
        "error": error,
        "primitive_labels": task.get("export_primitive_labels", False),
    }
    if cfg is not None:
        record["parameters"] = {
            "direct_dist_mm": cfg["fit2d"]["direct_dist"],
            "orientation_matrix": cfg["orientation_matrix"],
            "oscillation_range_deg": cfg["oscillation_range"],
            "num_images": cfg["num_images"],
            "scan_start_deg": cfg["scan_start_deg"],
            "mosaicity_deg": cfg["mosaicity_deg"],
            "beam_divergence_deg": cfg["beam_divergence_deg"],
            "domain_size_ang": cfg["domain_size_ang"],
            "d_min_ang": cfg["d_min"],
            "d_max_ang": cfg["d_max"],
        }
    return record


def process_cif(task):
    """worker 函数：一个任务处理一个 CIF，使昂贵结构因子仅构建一次。"""
    t0 = time.time()
    cif_path = task["cif_path"]
    cif_stem = task["cif_stem"]
    output_dir = task["output_dir"]

    stats = {
        "cif_stem": cif_stem,
        "chunk_id": 0,
        "n_ok": 0,
        "n_skipped": 0,
        "failures": [],
        "refl_min": None,
        "refl_max": None,
        "elapsed": 0.0,
        "fatal": None,
        "manifest_path": os.path.join(output_dir, "manifest.jsonl"),
    }

    # 1. 构建共享 CrystalModel 全集 + 强度查表字典（每个 CIF 仅一次）
    try:
        d_min_full = RANDOM_RANGES["d_min"][0]
        d_max_full = RANDOM_RANGES["d_max"][1]
        shared_cm = CrystalModel({"cif_path": cif_path})
        shared_cm.load()
        point_group = shared_cm.get_point_group_symbol()
        orientation_operations = proper_laue_operations(shared_cm._crystal_symmetry)
        shared_cm.build_miller_set_with_intensities(
            d_min=d_min_full, d_max=d_max_full
        )
        intensity_dict = shared_cm.get_intensity_dict()
        # 晶体结构信息（默认开启，同一 CIF 在分块内不变，构建一次供所有样本的 CSV 头使用）
        crystal_info = (
            shared_cm.get_crystal_info() if task["show_crystal_info"] else None
        )
    except Exception as exc:
        stats["fatal"] = f"CrystalModel 全集构建失败: {exc}"
        stats["elapsed"] = time.time() - t0
        _atomic_write_text(
            stats["manifest_path"],
            json.dumps({
                "cif": cif_stem, "cif_path": cif_path, "status": "fatal",
                "code_version": task["code_version"], "error": stats["fatal"],
            }, ensure_ascii=False, sort_keys=True) + "\n",
        )
        return stats

    os.makedirs(output_dir, exist_ok=True)

    # 2. 逐样本生成；每完成一个 CIF 后原子落盘该 CIF 的 manifest。
    records = []
    previous_records = {}
    if task["resume"] and os.path.isfile(stats["manifest_path"]):
        try:
            with open(stats["manifest_path"], "r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if "sample_index" in record:
                        previous_records[record["sample_index"]] = record
        except (OSError, ValueError, TypeError):
            previous_records = {}
    for global_idx in range(task["n_per_cif"]):
        sample_t0 = time.time()
        sample_rng, sample_seed, orientation_seed, spawn_key = _sample_random_state(
            task, global_idx
        )
        required = _required_outputs(output_dir, global_idx, task["output_mode"])
        complete = required and all(_csv_complete(path) for path in required)
        if task["resume"] and complete and task.get("export_primitive_labels", False):
            # Existing observations can acquire labels without being simulated
            # again. Their source hashes guard against stale label sidecars.
            for path in required:
                if not labels_complete(path):
                    write_lattice_labels(path, shared_cm.get_crystal_symmetry(), cif_path, overwrite=True)
            complete = all(labels_complete(path) for path in required)
        if task["resume"] and complete:
            stats["n_skipped"] += 1
            record = previous_records.get(global_idx)
            if record is None:
                record = _manifest_record(
                    task, global_idx, "skipped", sample_seed, orientation_seed,
                    spawn_key, time.time() - sample_t0,
                )
            else:
                record = dict(record)
                record["status"] = "skipped"
                record["resume_check_seconds"] = time.time() - sample_t0
            record["primitive_labels"] = task.get("export_primitive_labels", False)
            records.append(record)
            continue
        try:
            cfg = generate_random_config(
                task["base_config"],
                cif_path,
                sample_rng,
                point_group=point_group,
                use_fz=task["use_fz"],
                intensity_threshold=task["intensity_threshold"],
                random_scan_start=task["random_scan_start"],
                orientation_seed=orientation_seed,
                orientation_operations=orientation_operations,
            )
            cfg["_rng_version"] = task["rng_version"]
            cfg["_master_seed"] = task["master_seed"]
            cfg["_sample_seed"] = sample_seed
            # 注入共享对象，RotationPredictor 内部自动走复用分支
            cfg["_shared_crystal_model"] = shared_cm
            cfg["_shared_intensity_dict"] = intensity_dict

            predictor = RotationPredictor(cfg)
            results = predictor.predict_with_intensity(
                compute_frame_distribution=False
            )

            output_mode = task["output_mode"]
            n_refl = 0

            if output_mode in ("integrated", "both"):
                combined = build_combined_matrix(predictor, results, cfg=cfg)
                combined = combined.astype(np.float32)
                comment = build_config_comment(
                    cfg, global_idx, task["n_per_cif"],
                    point_group=point_group, output_mode="integrated",
                    crystal_info=crystal_info,
                )
                out_path = os.path.join(
                    output_dir, f"combined_matrix_{global_idx:04d}.csv"
                )
                _atomic_write_matrix(
                    out_path, comment, combined, task["decimals"]
                )
                if task.get("export_primitive_labels", False):
                    write_lattice_labels(out_path, shared_cm.get_crystal_symmetry(), cif_path, overwrite=True)
                n_refl = len(combined)

            if output_mode in ("per_frame", "both"):
                per_frame = build_per_frame_matrix_sparse(predictor, results, cfg=cfg)
                per_frame = per_frame.astype(np.float32)
                comment = build_config_comment(
                    cfg, global_idx, task["n_per_cif"],
                    point_group=point_group, output_mode="per_frame",
                    crystal_info=crystal_info,
                )
                out_path = os.path.join(
                    output_dir, f"per_frame_matrix_{global_idx:04d}.csv"
                )
                _atomic_write_matrix(
                    out_path, comment, per_frame, task["decimals"]
                )
                if task.get("export_primitive_labels", False):
                    write_lattice_labels(out_path, shared_cm.get_crystal_symmetry(), cif_path, overwrite=True)
                if output_mode == "per_frame":
                    n_refl = len(per_frame)

            stats["n_ok"] += 1
            stats["refl_min"] = (
                n_refl if stats["refl_min"] is None else min(stats["refl_min"], n_refl)
            )
            stats["refl_max"] = (
                n_refl if stats["refl_max"] is None else max(stats["refl_max"], n_refl)
            )
            records.append(_manifest_record(
                task, global_idx, "ok", sample_seed,
                cfg.get("_orientation_seed"), spawn_key,
                time.time() - sample_t0, n_rows=n_refl, cfg=cfg,
            ))
        except Exception as exc:
            stats["failures"].append((global_idx, sample_seed, str(exc)))
            records.append(_manifest_record(
                task, global_idx, "failed", sample_seed, orientation_seed,
                spawn_key, time.time() - sample_t0, error=str(exc),
            ))

    _atomic_write_text(
        stats["manifest_path"],
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
    )
    stats["elapsed"] = time.time() - t0
    return stats


# 兼容外部可能存在的旧导入名称；新调度器不再切分同一 CIF。
process_chunk = process_cif


def preflight_cif(cif_path, use_fz=True):
    """验证 CIF、点群和全集结构因子，并返回用于重任务优先调度的规模。"""
    t0 = time.time()
    result = {
        "cif_path": cif_path,
        "cif_stem": os.path.splitext(os.path.basename(cif_path))[0],
        "status": "invalid",
        "point_group": None,
        "reflection_count": None,
        "elapsed_seconds": None,
        "error": None,
    }
    try:
        cm = CrystalModel({"cif_path": cif_path})
        cm.load()
        result["point_group"] = cm.get_point_group_symbol()
        built = cm.build_miller_set_with_intensities(
            d_min=RANDOM_RANGES["d_min"][0],
            d_max=RANDOM_RANGES["d_max"][1],
        )
        # 同时触发 orix 点群兼容性检查，不改变任何生产数据。
        if use_fz:
            generate_random_config(
                {}, cif_path, np.random.default_rng(0),
                point_group=result["point_group"], use_fz=True,
                orientation_seed=0,
                orientation_operations=proper_laue_operations(cm._crystal_symmetry),
            )
        result["reflection_count"] = len(built["hkl_array"])
        result["status"] = "valid"
    except Exception as exc:
        result["error"] = str(exc)
    result["elapsed_seconds"] = time.time() - t0
    return result


def parse_args():
    """解析命令行参数。"""
    # --guide 优先于 argparse 必填校验：直接打印接口手册并退出
    if "--guide" in sys.argv:
        print(_GUIDE_TEXT)
        sys.exit(0)
    parser = argparse.ArgumentParser(
        description="为机器学习批量生成单晶旋转衍射模拟数据集（多进程并行）"
    )
    cif_group = parser.add_mutually_exclusive_group(required=True)
    cif_group.add_argument(
        "--cif",
        default=None,
        help="单个 CIF 文件名（验证用，在 --cif-dir 目录中查找）",
    )
    cif_group.add_argument(
        "--cif-dir",
        default=None,
        help="CIF 目录（批量用，默认查找 test_representative_100_cifs）",
    )
    parser.add_argument(
        "--n-per-cif",
        type=int,
        default=1000,
        help="每个 CIF 生成的样本数（默认 1000）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="兼容旧命令；新版按 CIF 调度，此参数不再切分结构因子缓存",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行进程数（本机安全默认 8；1 表示串行）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="全局随机种子；设置后整个数据集可复现（与并行进程数无关）",
    )
    parser.add_argument(
        "--rng-version", choices=["seedsequence-v1", "legacy"],
        default="seedsequence-v1",
        help="随机流版本；legacy 用于复现合作方旧版",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="跳过头部完整的已有输出（默认开启）",
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help="仅预检所有 CIF、点群与结构因子并写报告，不生成样本",
    )
    parser.add_argument(
        "--use-fz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否对随机取向做基本区投影（默认开启）",
    )
    parser.add_argument(
        "--random-scan-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scan_start_deg 是否随机（默认开启，-5~5°）；关闭后固定为 0°，样本多样性仅由初始取向控制",
    )
    parser.add_argument(
        "--intensity-threshold",
        type=float,
        default=DEFAULT_INTENSITY_THRESHOLD,
        help="衍射斑相对强度阈值 I/max(I)，仅保留 >= threshold 的衍射斑（默认 1e-4）",
    )
    parser.add_argument(
        "--output-mode",
        choices=["integrated", "per_frame", "both"],
        default="integrated",
        help="输出模式：integrated 积分斑点表（默认）、per_frame 逐帧图像序列、both 两者",
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
        default=6,
        help="CSV 数据区小数位数（默认 6，对应 float32 数据）",
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
        "--export-primitive-labels", action="store_true",
        help="另存联合 primitive Niggli 晶胞/取向/HKL 标签及结构基底变换；原 CSV 不改写",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    # 项目路径定位
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # CIF 目录与文件列表
    default_cifs_dir = os.path.abspath(
        os.path.join(project_root, "..", "test_representative_100_cifs")
    )
    cifs_dir = os.path.abspath(args.cif_dir) if args.cif_dir else default_cifs_dir

    if args.cif is not None:
        cif_path = os.path.join(cifs_dir, args.cif)
        if not os.path.exists(cif_path):
            raise FileNotFoundError(f"CIF 文件不存在：{cif_path}")
        cif_paths = [cif_path]
    else:
        cif_paths = sorted(glob.glob(os.path.join(cifs_dir, "*.cif")))
        if not cif_paths:
            raise FileNotFoundError(f"CIF 目录中未找到 .cif 文件：{cifs_dir}")

    # 基础配置文件
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(project_root, "config", "rotation_manual_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML 配置文件不存在：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    if args.n_per_cif <= 0:
        raise ValueError("--n-per-cif 必须为正整数")
    if args.workers <= 0:
        raise ValueError("--workers 必须为正整数")
    if not 0.0 <= args.intensity_threshold <= 1.0:
        raise ValueError("--intensity-threshold 必须在 [0, 1] 范围内")
    if args.decimals < 0:
        raise ValueError("--decimals 不能为负数")

    # 输出根目录
    output_root = args.output_dir
    if output_root is None:
        output_root = os.path.join(project_root, "ml_data")
    os.makedirs(output_root, exist_ok=True)

    workers = args.workers
    code_version = _code_version(project_root)

    if args.preflight:
        if workers == 1:
            preflight_results = [preflight_cif(path, args.use_fz) for path in cif_paths]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(preflight_cif, path, args.use_fz): path
                    for path in cif_paths
                }
                preflight_results = [future.result() for future in as_completed(futures)]
        preflight_results.sort(key=lambda item: item["cif_stem"])
        report = {
            "code_version": code_version,
            "use_fz": args.use_fz,
            "valid_count": sum(r["status"] == "valid" for r in preflight_results),
            "invalid_count": sum(r["status"] != "valid" for r in preflight_results),
            "results": preflight_results,
        }
        report_path = os.path.join(output_root, "preflight_report.json")
        _atomic_write_text(report_path, json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n")
        print(f"预检完成：{report['valid_count']} valid, "
              f"{report['invalid_count']} invalid")
        print(f"报告：{report_path}")
        return 1 if report["invalid_count"] else 0

    # seedsequence-v1 在未指定 seed 时生成并公开实际 master seed。
    master_seed = args.seed
    if args.rng_version == "seedsequence-v1" and master_seed is None:
        master_seed = int(np.random.SeedSequence().entropy)
    legacy_rng = np.random.default_rng(args.seed)

    # 每个 CIF 一个任务：结构因子在整个 CIF 的全部样本中只构建一次。
    tasks = []
    for cif_index, cif_path in enumerate(cif_paths):
        cif_stem = os.path.splitext(os.path.basename(cif_path))[0]
        legacy_sample_seeds = (
            [int(legacy_rng.integers(0, 2**31)) for _ in range(args.n_per_cif)]
            if args.rng_version == "legacy" else None
        )
        cif_output_dir = os.path.join(output_root, cif_stem)
        tasks.append({
            "cif_path": cif_path,
            "cif_stem": cif_stem,
            "cif_index": cif_index,
            "legacy_sample_seeds": legacy_sample_seeds,
            "master_seed": master_seed,
            "rng_version": args.rng_version,
            "n_per_cif": args.n_per_cif,
            "base_config": base_config,
            "use_fz": args.use_fz,
            "intensity_threshold": args.intensity_threshold,
            "output_mode": args.output_mode,
            "show_crystal_info": args.show_crystal_info,
            "export_primitive_labels": args.export_primitive_labels,
            "decimals": args.decimals,
            "random_scan_start": args.random_scan_start,
            "resume": args.resume,
            "code_version": code_version,
            "output_dir": cif_output_dir,
        })

    # 若已有预检报告，则优先提交反射数较大的 CIF，减轻尾部负载不均。
    preflight_path = os.path.join(output_root, "preflight_report.json")
    preflight_invalid = []
    if os.path.isfile(preflight_path):
        try:
            with open(preflight_path, "r", encoding="utf-8") as handle:
                preflight_items = json.load(handle).get("results", [])
            size_by_cif = {
                item["cif_stem"]: item.get("reflection_count") or 0
                for item in preflight_items
            }
            status_by_cif = {
                item["cif_stem"]: item for item in preflight_items
            }
            preflight_invalid = [
                item for item in preflight_items
                if item.get("status") != "valid"
                and item["cif_stem"] in {t["cif_stem"] for t in tasks}
            ]
            tasks = [
                task for task in tasks
                if status_by_cif.get(task["cif_stem"], {}).get("status", "valid") == "valid"
            ]
            tasks.sort(key=lambda t: size_by_cif.get(t["cif_stem"], 0), reverse=True)
        except (OSError, ValueError, TypeError):
            pass

    requested_samples = len(cif_paths) * args.n_per_cif
    total_samples = len(tasks) * args.n_per_cif
    print(f"CIF 目录:   {cifs_dir}")
    print(f"CIF 数量:   {len(cif_paths)}")
    print(f"每 CIF 样本数: {args.n_per_cif}（计划生成 {total_samples} / 请求 {requested_samples}）")
    if preflight_invalid:
        print("预检隔离 CIF: " + ", ".join(item["cif_stem"] for item in preflight_invalid))
    print(f"CIF 任务数: {len(tasks)}（每 CIF 仅构建一次结构因子）")
    print(f"并行进程数: {workers}")
    print(f"FZ 投影:    {'开启' if args.use_fz else '关闭'}")
    print(f"强度阈值:   {args.intensity_threshold:.2e}")
    print(f"输出模式:   {args.output_mode}")
    print(f"输出根目录: {output_root}")
    print(f"RNG 版本:   {args.rng_version}")
    print(f"Master seed: {master_seed if master_seed is not None else 'None（legacy 不固定）'}")
    print(f"断点续跑:   {'开启' if args.resume else '关闭'}")
    print("-" * 72)

    # 执行（workers=1 时串行，便于调试与对比验证）
    all_stats = []
    n_done = 0
    if workers == 1:
        for task in tasks:
            stats = process_cif(task)
            all_stats.append(stats)
            n_done += 1
            _print_progress(stats, n_done, len(tasks), t_start)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_cif, t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    stats = future.result()
                except Exception as exc:
                    stats = {
                        "cif_stem": task["cif_stem"], "chunk_id": 0,
                        "n_ok": 0, "n_skipped": 0, "failures": [],
                        "refl_min": None, "refl_max": None, "elapsed": 0.0,
                        "fatal": f"worker 异常退出: {exc}", "manifest_path": None,
                    }
                all_stats.append(stats)
                n_done += 1
                _print_progress(stats, n_done, len(tasks), t_start)

    # 汇总
    total_ok = sum(s["n_ok"] for s in all_stats)
    total_skipped = sum(s.get("n_skipped", 0) for s in all_stats)
    total_failures = [
        (s["cif_stem"], idx, seed, err)
        for s in all_stats
        for idx, seed, err in s["failures"]
    ]
    fatal_chunks = [s for s in all_stats if s["fatal"] is not None]
    total_elapsed = time.time() - t_start

    # 汇总所有按 CIF 原子写出的 manifest；重复运行会重建而非追加重复记录。
    manifest_lines = []
    for stats in sorted(all_stats, key=lambda s: s["cif_stem"]):
        path = stats.get("manifest_path")
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                manifest_lines.extend(handle.readlines())
    for item in preflight_invalid:
        manifest_lines.append(json.dumps({
            "cif": item["cif_stem"], "cif_path": item["cif_path"],
            "status": "quarantined", "code_version": code_version,
            "error": item.get("error"),
        }, ensure_ascii=False, sort_keys=True) + "\n")
    _atomic_write_text(os.path.join(output_root, "run_manifest.jsonl"), "".join(manifest_lines))
    failure_records = []
    for line in manifest_lines:
        record = json.loads(line)
        if record.get("status") in ("failed", "fatal", "quarantined"):
            failure_records.append(line)
    _atomic_write_text(os.path.join(output_root, "failures.jsonl"), "".join(failure_records))

    print("-" * 72)
    print(f"数据集生成完成：{output_root}")
    print(f"总耗时: {total_elapsed / 60.0:.1f} 分钟")
    print(f"成功样本: {total_ok}，断点跳过: {total_skipped} / {total_samples}")
    if fatal_chunks:
        print(f"致命失败分块（{len(fatal_chunks)} 个）：")
        for s in fatal_chunks:
            print(f"  {s['cif_stem']} chunk#{s['chunk_id']}: {s['fatal']}")
    if total_failures:
        print(f"失败样本（{len(total_failures)} 个）：")
        for cif_stem, idx, seed, err in total_failures[:50]:
            print(f"  {cif_stem} 样本{idx:04d} (seed={seed}): {err}")
        if len(total_failures) > 50:
            print(f"  ... 其余 {len(total_failures) - 50} 个从略")
    return 1 if preflight_invalid or fatal_chunks or total_failures else 0


def _print_progress(stats, n_done, n_total, t_start):
    """打印单个分块完成后的进度信息（仅主进程调用）。"""
    elapsed_total = time.time() - t_start
    eta = elapsed_total / n_done * (n_total - n_done) if n_done > 0 else 0.0
    if stats["fatal"] is not None:
        print(
            f"[{n_done:4d}/{n_total}] {stats['cif_stem']}#{stats['chunk_id']} "
            f"致命失败: {stats['fatal']} | 累计 {elapsed_total / 60:.1f} min"
        )
    else:
        refl_range = (
            f"{stats['refl_min']}~{stats['refl_max']}"
            if stats["refl_min"] is not None
            else "N/A"
        )
        print(
            f"[{n_done:4d}/{n_total}] {stats['cif_stem']}#{stats['chunk_id']} "
            f"ok={stats['n_ok']} skip={stats.get('n_skipped', 0)} "
            f"fail={len(stats['failures'])} "
            f"refs={refl_range} "
            f"耗时={stats['elapsed']:.1f}s | 累计 {elapsed_total / 60:.1f} min "
            f"ETA {eta / 60:.1f} min"
        )


if __name__ == "__main__":
    sys.exit(main())
