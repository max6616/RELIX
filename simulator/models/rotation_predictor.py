# -*- coding: utf-8 -*-
"""
rotation_predictor.py

旋转扫描衍射斑预测器。

基于 cctbx + DIALS + pyFAI，预测单晶样品在指定旋转范围内
所有可能被探测器记录的衍射斑位置、q 矢量和结构因子强度。

本类仅负责流程编排（Controller 角色），所有具体计算均委托给：
- CrystalModel：晶体加载、Miller 集生成、强度计算、B 矩阵
- DetectorModel：探测器几何、DIALS 探测器构造
- OrientationModel：取向矩阵 U、DIALS A 矩阵

输入：dict 或 YAML 配置文件。
输出：dict，包含 predicted_hkls、predicted_px、predicted_mm、
      predicted_intensity、predicted_q_hkls 等字段。
"""
import os
import sys

# 动态添加项目根目录到 sys.path
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _script_dir = os.getcwd()
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.append(_project_root)

import numpy as np
import yaml
from scipy.special import erf
from dxtbx.model import Goniometer, Crystal, Scan, Experiment
from dxtbx.model.beam import BeamFactory
from dials.algorithms.spot_prediction import ScanStaticReflectionPredictor
from dials.array_family import flex

from models.detector_model import DetectorModel
from models.crystal_model import CrystalModel
from models.orientation_model import OrientationModel


class RotationPredictor:
    """
    单晶旋转扫描衍射斑预测器。

    参数:
        config (dict): 配置字典，必须包含以下键：
            - d_min (float): 最小 d 间距，单位 Å
            - d_max (float): 最大 d 间距，单位 Å
            - num_images (int): 扫描帧数
            - oscillation_range (float): 总扫描角度，单位度，默认 1.0

        晶体输入（二选一）：
            - cif_path (str): CIF 文件路径
            - 或 unit_cell_parameters (list) + space_group_symbol (str)

        探测器输入（三选一）：
            - poni_path (str): PONI 文件路径
            - 或 poni (dict): PONI 参数字典
            - 或 fit2d (dict): Fit2D 风格参数字典

        晶体初始取向（三选一）：
            - angle1_deg (float) + angle2_deg (float):
                先绕 X 轴旋转 angle1_deg，再绕 Z 轴旋转 angle2_deg
            - 或 orientation_matrix (list[list]): 直接给定 3×3 取向矩阵 U
            - 或 euler_angles (list): 欧拉角 [omega, chi, phi]，单位度

        旋转轴与起始角（可选，仅与 angle1_deg+angle2_deg 方式联用）：
            - rotation_angle_deg (float): 扫描起始前绕 rotation_axis 的额外旋转角，默认 0
            - rotation_axis (list/tuple): 额外旋转轴，默认 [0, 1, 0]

        波长（可选）：
            - wavelength_angstrom (float): 波长，单位 Å；
              未指定时从 PONI/Fit2D 读取并转换为 Å
    """

    def __init__(self, config):
        self.config = dict(config)
        self._validate_config()

        # 状态参数：记录各类输入的加载方式
        self._crystal_input_mode = None      # "cif" | "manual"
        self._detector_input_mode = None     # "poni_file" | "poni_dict" | "fit2d"
        self._wavelength_source = None       # "config" | "poni" | "fit2d"
        self._orientation_input_mode = None  # "rotation_predictor_angles" 等
        self._intensity_available = False
        self._cif_loaded = False

        # 文件路径（可能为 None）
        self.cif_path = None
        if "cif_path" in self.config and self.config["cif_path"] is not None:
            self.cif_path = self._resolve_path(self.config["cif_path"])
            self.config["cif_path"] = self.cif_path

        self.poni_path = None
        if "poni_path" in self.config and self.config["poni_path"] is not None:
            self.poni_path = self._resolve_path(self.config["poni_path"])
            self.config["poni_path"] = self.poni_path

        # 分辨率范围
        self.d_min = float(self.config["d_min"])
        self.d_max = float(self.config["d_max"])

        # 晶体取向：支持 orientation_matrix、euler_angles、angle1_deg+angle2_deg 三种方式
        self.angle1_deg = float(self.config.get("angle1_deg", 0.0))
        self.angle2_deg = float(self.config.get("angle2_deg", 0.0))
        self.rotation_angle_deg = float(self.config.get("rotation_angle_deg", 0.0))
        self.rotation_axis = tuple(self.config.get("rotation_axis", [0.0, 1.0, 0.0]))

        # 初始取向模式：fixed（默认）或 random
        # 当未提供 initial_orientation 段时，等价于 mode="fixed"
        self._initial_orientation_cfg = self.config.get("initial_orientation", {}) or {}
        self._initial_orientation_mode = str(self._initial_orientation_cfg.get("mode", "fixed"))
        self._random_orientation_dict = None  # 记录 random 模式的生成结果

        # 扫描参数
        self.num_images = int(self.config["num_images"])
        self.oscillation_range = float(self.config.get("oscillation_range", 1.0))
        self.scan_start_deg = float(self.config.get("scan_start_deg", 0.0))

        # 强度计算可选参数（用于 predict_with_intensity）
        self.mosaicity_deg = float(self.config.get("mosaicity_deg", 0.0))
        self.beam_divergence_deg = float(self.config.get("beam_divergence_deg", 0.0))
        self.domain_size_ang = float(self.config.get("domain_size_ang", 1.0e6))
        self.intensity_distribution = str(self.config.get("intensity_distribution", "gaussian"))
        self.apply_lp_correction = bool(self.config.get("apply_lp_correction", True))

        # 波长：优先使用配置值，否则从 PONI/Fit2D 读取
        if "wavelength_angstrom" in self.config and self.config["wavelength_angstrom"] is not None:
            self.wavelength_angstrom = float(self.config["wavelength_angstrom"])
            self._wavelength_source = "config"
        else:
            self.wavelength_angstrom = None
            self._wavelength_source = None

        # 子 Model 实例，逐步填充
        self._crystal_model = None
        self._detector_model = None
        self._orientation_model = None

        # 内部状态，逐步填充
        self.structure = None
        self.crystal_symmetry = None
        self.unit_cell = None
        self.space_group_symbol = None
        self.hkls = None
        self.d_spacings = None
        self.intensities = None

        self.ai = None
        self.U_rotated = None
        self.A_dials = None

        self.experiment = None
        self.predicted = None

        # 输出结果
        self.predicted_hkls = None
        self.predicted_px = None
        self.predicted_mm = None
        self.predicted_intensity = None
        self.predicted_q_hkls = None

    @classmethod
    def from_yaml(cls, yaml_path):
        """从 YAML 配置文件构造预测器。"""
        yaml_path = cls._resolve_static_path(yaml_path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def to_dict(self):
        """返回当前配置字典。"""
        return dict(self.config)

    # ------------------------------------------------------------------
    # 1. 晶体部分：委托 CrystalModel
    # ------------------------------------------------------------------
    def load_crystal(self):
        """加载晶体并生成 Miller 集合与强度。"""
        # 复用外部注入的共享 CrystalModel（批量生成场景）：
        # 全集已构建，仅按当前 d 范围过滤子集，跳过 CIF 解析与结构因子计算。
        # 未注入时完全走原有代码路径，行为不变。
        shared_cm = self.config.get("_shared_crystal_model")
        if shared_cm is not None and shared_cm.is_cif_loaded():
            self._crystal_model = shared_cm
            self.structure = shared_cm.get_structure()
            self.crystal_symmetry = shared_cm.get_crystal_symmetry()
            self.unit_cell = shared_cm.get_unit_cell()
            self.space_group_symbol = shared_cm.get_space_group_symbol()
            self._crystal_input_mode = shared_cm.get_input_mode()
            self._cif_loaded = True

            result = shared_cm.filter_miller_set_by_d_range(
                d_min=self.d_min,
                d_max=self.d_max,
            )

            # CrystalModel 返回 (N, 3)，本类内部沿用原 (3, N) 约定
            self.hkls = result["hkl_array"].T
            self.d_spacings = result["d_spacings"]
            self.intensities = result["intensities"]
            self._intensity_available = result["intensity_available"]

            return self

        self._crystal_model = CrystalModel(self.config)
        self._crystal_model.load()

        self.structure = self._crystal_model.get_structure()
        self.crystal_symmetry = self._crystal_model.get_crystal_symmetry()
        self.unit_cell = self._crystal_model.get_unit_cell()
        self.space_group_symbol = self._crystal_model.get_space_group_symbol()

        self._crystal_input_mode = self._crystal_model.get_input_mode()
        self._cif_loaded = self._crystal_model.is_cif_loaded()

        result = self._crystal_model.build_miller_set_with_intensities(
            d_min=self.d_min,
            d_max=self.d_max,
        )

        # CrystalModel 返回 (N, 3)，本类内部沿用原 (3, N) 约定
        self.hkls = result["hkl_array"].T
        self.d_spacings = result["d_spacings"]
        self.intensities = result["intensities"]
        self._intensity_available = result["intensity_available"]

        return self

    # ------------------------------------------------------------------
    # 2. 探测器部分：委托 DetectorModel
    # ------------------------------------------------------------------
    def load_detector(self):
        """加载探测器几何。"""
        self._detector_model = DetectorModel(self.config)
        self.ai = self._detector_model.ai
        self._detector_input_mode = self._detector_model.get_input_mode()

        if self.wavelength_angstrom is None:
            self.wavelength_angstrom = self._detector_model.get_wavelength_angstrom()[
                "wavelength_angstrom"
            ]
            self._wavelength_source = self._detector_input_mode

        return self

    # ------------------------------------------------------------------
    # 3. 取向部分：委托 OrientationModel
    # ------------------------------------------------------------------
    def build_orientation(self):
        """构造旋转后的 U 矩阵与 DIALS setting matrix A。"""
        # 若是 random 模式：调用 OrientationModel.random_orientation() 生成 U，
        # 注入 self.config["orientation_matrix"] 后再走原 fixed 逻辑（保持单入口）
        if self._initial_orientation_mode == "random":
            self._random_orientation_dict = OrientationModel.random_orientation(
                output_format=self._initial_orientation_cfg.get("output_format", "matrix"),
                seed=self._initial_orientation_cfg.get("random_seed"),
                point_group=self._initial_orientation_cfg.get("point_group"),
            )
            self.config["orientation_matrix"] = self._random_orientation_dict["u_matrix"].tolist()

        self._orientation_model = OrientationModel(self.config)
        self._orientation_input_mode = self._orientation_model.get_input_mode()
        self.U_rotated = self._orientation_model.get_u_matrix()["u_matrix"]

        # DIALS 的 B 矩阵不含 2π 因子
        b_result = self._crystal_model.get_b_matrix()
        b_dials = b_result["b_matrix_no_2pi"]

        a_result = self._orientation_model.build_dials_a_matrix(b_dials)
        self.A_dials = a_result["a_matrix"]

        return self

    # ------------------------------------------------------------------
    # 4. dials 部分：组装 Experiment
    # ------------------------------------------------------------------
    def build_experiment(self):
        """构造 DIALS Experiment，包含 Beam、Detector、Goniometer、Crystal、Scan。"""
        # Beam
        beam = BeamFactory.make_beam(
            sample_to_source=(0.0, 0.0, -1.0),
            wavelength=self.wavelength_angstrom,
        )

        # Detector
        detector = self._detector_model.get_dials_detector()["detector"]

        # Goniometer
        goniometer = Goniometer(rotation_axis=self.rotation_axis)

        # Crystal
        crystal_dials = Crystal(self.A_dials.flatten().tolist(), self.space_group_symbol)

        # Scan
        step = self.oscillation_range / self.num_images
        scan = Scan(
            image_range=(1, self.num_images),
            oscillation=(self.scan_start_deg, step),
        )

        # Experiment
        self.experiment = Experiment(
            beam=beam,
            detector=detector,
            goniometer=goniometer,
            crystal=crystal_dials,
            scan=scan,
            identifier="0",
        )

        return self

    # ------------------------------------------------------------------
    # 5. 预测部分：执行 ScanStaticReflectionPredictor 并提取结果
    # ------------------------------------------------------------------
    def predict(self):
        """执行完整的旋转扫描预测流程，返回结果字典。"""
        self.load_crystal()
        self.load_detector()
        self.build_orientation()
        self.build_experiment()

        crystal_dials = self.experiment.crystal

        predictor = ScanStaticReflectionPredictor(
            experiment=self.experiment,
            dmin=self.d_min,
        )

        predicted = predictor.for_ub(crystal_dials.get_A())
        predicted.compute_d_single(self.experiment)

        # 过滤 dmax
        mask_dmax = predicted["d"] > self.d_max
        predicted.del_selected(mask_dmax)

        # 附加 cctbx 计算的强度；无 CIF 时强度统一为 1.0
        if self._intensity_available:
            # 批量生成场景可注入预构建的全集查表字典，避免每样本重建
            intensity_dict = self.config.get("_shared_intensity_dict")
            if intensity_dict is None:
                intensity_dict = {
                    tuple(hkl): intensity
                    for hkl, intensity in zip(self.hkls.T, self.intensities)
                }
            default_intensity = 0.0
        else:
            intensity_dict = {}
            default_intensity = 1.0

        # 一次性构造 flex 数组，避免逐项写入 flex.double 的 Python/C++ 边界开销。
        # 查表规则和默认值与原实现完全一致。
        predicted["intensity.cctbx"] = flex.double([
            float(intensity_dict.get(tuple(hkl), default_intensity))
            for hkl in predicted["miller_index"]
        ])

        self.predicted = predicted

        # 提取 numpy 数组结果
        n_predicted = len(predicted)
        if n_predicted == 0:
            print("[RotationPredictor] 警告：DIALS 未预测到任何反射，请检查取向、波长和 d 范围。")
            self.predicted_hkls = np.zeros((3, 0), dtype=int)
            self.predicted_px = np.zeros((0, 3), dtype=float)
            self.predicted_mm = np.zeros((0, 3), dtype=float)
            self.predicted_intensity = np.zeros(0, dtype=float)
            self.predicted_q_hkls = np.zeros((0, 3), dtype=float)
        else:
            self.predicted_hkls = np.array(list(predicted["miller_index"])).T
            self.predicted_px = np.array(predicted["xyzcal.px"]).reshape(-1, 3)
            self.predicted_mm = np.array(predicted["xyzcal.mm"]).reshape(-1, 3)
            self.predicted_intensity = np.array(predicted["intensity.cctbx"])

            # 计算 q 矢量：DIALS 的 s1 为 (1/λ) * 方向，s0 = (0, 0, 1/λ)
            s1 = np.array(predicted["s1"]).reshape(-1, 3)
            s0 = np.array([0.0, 0.0, 1.0 / self.wavelength_angstrom])
            self.predicted_q_hkls = 2 * np.pi * (s1 - s0)

        return self.get_results()

    def get_results(self):
        """返回预测结果字典。"""
        return {
            "predicted_hkls": self.predicted_hkls,
            "predicted_px": self.predicted_px,
            "predicted_mm": self.predicted_mm,
            "predicted_intensity": self.predicted_intensity,
            "predicted_q_hkls": self.predicted_q_hkls,
            "predicted_table": self.predicted,
            "config": self.to_dict(),
        }

    # ------------------------------------------------------------------
    # 强度计算与帧分配（predict_with_intensity 专用）
    # ------------------------------------------------------------------
    def _apply_lp_correction(self, intensities, d_spacings, q_vectors):
        """
        对结构因子强度 |F|² 应用洛伦兹-偏振校正。

        旋转法总积分强度：
            I_obs = |F|² · L · P
        其中：
            L = 1 / (sin 2θ · |sin α|)
            P = (1 + cos² 2θ) / 2
        α 为旋转轴与 q 矢量的夹角。

        参数:
            intensities (ndarray): 结构因子强度 |F|²，形状 (N,)
            d_spacings (ndarray): d 间距，单位 Å，形状 (N,)
            q_vectors (ndarray): q 矢量（实验室坐标系），形状 (N, 3)

        返回:
            ndarray: LP 校正后的总积分强度，形状 (N,)
        """
        intensities = np.asarray(intensities, dtype=float)
        d_spacings = np.asarray(d_spacings, dtype=float)
        q_vectors = np.asarray(q_vectors, dtype=float)

        n = len(intensities)
        if n == 0:
            return np.zeros(0, dtype=float)

        # 布拉格角
        sin_theta = self.wavelength_angstrom / (2.0 * d_spacings)
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        two_theta = 2.0 * np.arcsin(sin_theta)
        cos_two_theta = np.cos(two_theta)
        sin_two_theta = np.sin(two_theta)

        # 偏振因子（非偏振光）
        polarization = 0.5 * (1.0 + cos_two_theta ** 2)

        # 旋转轴归一化
        axis = np.array(self.rotation_axis, dtype=float)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            raise ValueError("旋转轴长度不能为零")
        axis = axis / axis_norm

        # 旋转轴与 q 矢量的夹角 alpha
        q_norm = np.linalg.norm(q_vectors, axis=1)
        q_norm_safe = np.where(q_norm < 1e-12, 1.0, q_norm)
        cos_alpha = np.einsum("ij,j->i", q_vectors, axis) / q_norm_safe
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        sin_alpha = np.sqrt(1.0 - cos_alpha ** 2)
        sin_alpha = np.where(sin_alpha < 1e-12, 1e-12, sin_alpha)

        # 洛伦兹因子，处理 sin(2θ) 接近 0 的情况
        sin_two_theta_safe = np.where(
            np.abs(sin_two_theta) < 1e-12, 1e-12, sin_two_theta
        )
        lorentz = 1.0 / (sin_two_theta_safe * sin_alpha)

        return intensities * lorentz * polarization

    def _compute_rocking_widths(self, d_spacings):
        """
        计算每个反射的 rocking width（FWHM，单位度）。

        总峰宽由三部分平方和相加：
        - mosaicity（晶体镶嵌度）
        - beam_divergence（光束发散度）
        - Scherrer 展宽：λ / (2 · domain_size · cos θ)

        其中 θ 为布拉格角，满足 sin θ = λ / (2d)。
        因此 Scherrer 项随 d 减小（高角度）而增大， rocking width 也随 d 变化。

        参数:
            d_spacings (ndarray): d 间距，单位 Å，形状 (N,)

        返回:
            ndarray: 每个反射的 rocking width，单位度，形状 (N,)
        """
        d_spacings = np.asarray(d_spacings, dtype=float)
        n = len(d_spacings)
        if n == 0:
            return np.zeros(0, dtype=float)

        # mosaicity 和 beam divergence（度 -> 弧度）
        eta_rad = np.deg2rad(self.mosaicity_deg)
        delta_rad = np.deg2rad(self.beam_divergence_deg)

        # Scherrer 展宽（弧度），domain_size 极大时忽略
        if self.domain_size_ang > 1.0e9:
            scherrer_rad = 0.0
        else:
            sin_theta = self.wavelength_angstrom / (2.0 * d_spacings)
            sin_theta = np.clip(sin_theta, -1.0, 1.0)
            cos_theta = np.sqrt(1.0 - sin_theta ** 2)
            cos_theta_safe = np.where(cos_theta < 1e-12, 1e-12, cos_theta)
            scherrer_rad = self.wavelength_angstrom / (
                2.0 * self.domain_size_ang * cos_theta_safe
            )

        # 总峰宽：平方和开根号
        total_rad = np.sqrt(eta_rad ** 2 + delta_rad ** 2 + scherrer_rad ** 2)

        return np.rad2deg(total_rad)

    def _distribute_intensity_to_frames(
        self, predicted_px, total_intensities, rocking_widths_deg
    ):
        """
        将每个反射的总积分强度按 rocking curve 分配到各帧。

        DIALS 的 predicted_px[:, 2] 是连续数组索引（0-based 实数），
        先转换为中心角度 φ₀，再以 φ₀ 为中心定义 rocking curve，
        最后将曲线在每个帧的角度区间内积分得到权重。

        本实现为向量化版本：所有反射的 rocking curve 积分通过
        (n_block, num_images+1) 矩阵批量计算，并带分块保护，
        当 n × (num_images+1) 超过阈值时分块处理，避免内存溢出。
        数值结果与逐反射循环实现一致。

        参数:
            predicted_px (ndarray): DIALS 预测像素坐标 [x, y, frame]，形状 (N, 3)
            total_intensities (ndarray): 每个反射的总积分强度，形状 (N,)
            rocking_widths_deg (ndarray): 每个反射的 rocking width FWHM（度），形状 (N,)

        返回:
            dict: 包含以下键：
                - per_frame_total_intensity: (num_images,) 每帧总强度
                - per_spot_frame_assignment: (N,) 每个反射主归属帧（0-based）
                - per_spot_frame_weights: (N, num_images) 每个反射分配到各帧的权重
        """
        predicted_px = np.asarray(predicted_px, dtype=float)
        total_intensities = np.asarray(total_intensities, dtype=float)
        rocking_widths_deg = np.asarray(rocking_widths_deg, dtype=float)

        n = len(predicted_px)
        num_images = self.num_images
        frame_width = self.oscillation_range / self.num_images
        scan_start = self.scan_start_deg

        if n == 0:
            return {
                "per_frame_total_intensity": np.zeros(num_images, dtype=float),
                "per_spot_frame_assignment": np.zeros(0, dtype=int),
                "per_spot_frame_weights": np.zeros((0, num_images), dtype=float),
            }

        weights = np.zeros((n, num_images), dtype=float)
        for bs, be, block_weights in self._iter_frame_weight_blocks(
            predicted_px, rocking_widths_deg
        ):
            weights[bs:be] = block_weights

        # 强度分配
        intensity_per_frame = weights * total_intensities[:, np.newaxis]
        per_frame_total = np.sum(intensity_per_frame, axis=0)
        per_spot_assignment = np.argmax(weights, axis=1)

        return {
            "per_frame_total_intensity": per_frame_total,
            "per_spot_frame_assignment": per_spot_assignment,
            "per_spot_frame_weights": weights,
        }

    def _iter_frame_weight_blocks(self, predicted_px, rocking_widths_deg,
                                  max_block_elements=2.0e7):
        """逐块生成帧权重；数学过程与原密集矩阵实现相同。"""
        predicted_px = np.asarray(predicted_px, dtype=float)
        rocking_widths_deg = np.asarray(rocking_widths_deg, dtype=float)
        n = len(predicted_px)
        num_images = self.num_images
        if n == 0:
            return

        frame_width = self.oscillation_range / num_images
        scan_start = self.scan_start_deg
        frame_centers = predicted_px[:, 2]
        phi0 = scan_start + frame_centers * frame_width
        frame_edges = np.linspace(
            scan_start, scan_start + num_images * frame_width, num_images + 1
        )
        block_size = max(1, int(max_block_elements // (num_images + 1)))

        for bs in range(0, n, block_size):
            be = min(bs + block_size, n)
            m = be - bs
            phi0_b = phi0[bs:be]
            width_b = rocking_widths_deg[bs:be]
            centers_b = frame_centers[bs:be]

            if self.intensity_distribution == "gaussian":
                sigma = width_b / 2.355
                sigma_safe = np.where(sigma < 1e-12, 1.0, sigma)
                cdf_vals = 0.5 * (
                    1.0 + erf(
                        (frame_edges[np.newaxis, :] - phi0_b[:, np.newaxis])
                        / (sigma_safe[:, np.newaxis] * np.sqrt(2.0))
                    )
                )
                profile = np.diff(cdf_vals, axis=1)
            else:
                half_width = 0.5 * width_b
                left_edges = np.maximum(
                    frame_edges[:-1][np.newaxis, :],
                    (phi0_b - half_width)[:, np.newaxis],
                )
                right_edges = np.minimum(
                    frame_edges[1:][np.newaxis, :],
                    (phi0_b + half_width)[:, np.newaxis],
                )
                profile = np.maximum(0.0, right_edges - left_edges) / np.where(
                    width_b < 1e-12, 1.0, width_b
                )[:, np.newaxis]

            total_weight = np.sum(profile, axis=1)
            frame_idx = np.clip(
                np.floor(centers_b).astype(int), 0, num_images - 1
            )
            degenerate = (width_b < 1e-12) | (total_weight <= 1e-15)
            normal = ~degenerate
            block_weights = np.zeros((m, num_images), dtype=float)
            if np.any(normal):
                block_weights[normal] = (
                    profile[normal] / total_weight[normal, np.newaxis]
                )
            if np.any(degenerate):
                block_weights[degenerate, frame_idx[degenerate]] = 1.0
            yield bs, be, block_weights

    def predict_with_intensity(self, apply_lp=None, distribution=None,
                               compute_frame_distribution=True):
        """
        执行完整预测并附加强度计算与帧分配。

        本方法先调用原有的 predict() 完成几何预测，
        再基于 cctbx 的结构因子强度计算 LP 校正后的总积分强度，
        并按 rocking curve 分配到各帧。

        参数:
            apply_lp (bool, optional): 是否应用 LP 校正，
                默认使用 config 中的 apply_lp_correction
            distribution (str, optional): 强度分布类型，"gaussian" 或 "uniform"，
                默认使用 config 中的 intensity_distribution

        返回:
            dict: 包含 predict() 原有字段，以及：
                - predicted_intensity_lp: LP 校正后的总积分强度 (N,)
                - per_frame_total_intensity: 每帧总强度 (num_images,)
                - per_spot_frame_assignment: 每个反射主归属帧 (N,)
                - per_spot_frame_weights: 每个反射分配到各帧的权重 (N, num_images)
                - rocking_widths_deg: 每个反射的 rocking width (N,)
                - d_spacings: 每个反射的 d 间距 (N,)
        """
        if apply_lp is None:
            apply_lp = self.apply_lp_correction
        if distribution is not None:
            self.intensity_distribution = str(distribution)

        # 1. 调用原有 predict() 完成几何预测
        results = self.predict()

        n = len(self.predicted_px)
        if n == 0:
            results.update({
                "predicted_intensity_lp": np.zeros(0, dtype=float),
                "per_frame_total_intensity": np.zeros(self.num_images, dtype=float),
                "per_spot_frame_assignment": np.zeros(0, dtype=int),
                "per_spot_frame_weights": np.zeros((0, self.num_images), dtype=float),
                "rocking_widths_deg": np.zeros(0, dtype=float),
                "d_spacings": np.zeros(0, dtype=float),
            })
            return results

        # 2. 获取 |F|² 强度
        # 直接使用 predict() 已经计算好的强度，避免两处查表不一致
        base_intensities = self.predicted_intensity.copy()

        # 3. 计算 d 间距：|q| = 2π / d
        q_norms = np.linalg.norm(self.predicted_q_hkls, axis=1)
        q_norms_safe = np.where(q_norms < 1e-12, 1.0, q_norms)
        d_spacings = 2.0 * np.pi / q_norms_safe

        # 4. LP 校正
        if apply_lp:
            lp_intensities = self._apply_lp_correction(
                base_intensities, d_spacings, self.predicted_q_hkls
            )
        else:
            lp_intensities = base_intensities.copy()

        # 5. 计算 rocking width
        rocking_widths = self._compute_rocking_widths(d_spacings)

        # 6. 按需分配到帧。积分斑点批处理不消费这些矩阵，跳过可避免
        # O(N_reflections * N_frames) 的时间和内存；默认 True 保持 API 兼容。
        if compute_frame_distribution:
            distribution_result = self._distribute_intensity_to_frames(
                self.predicted_px, lp_intensities, rocking_widths
            )
        else:
            distribution_result = {
                "per_frame_total_intensity": None,
                "per_spot_frame_assignment": None,
                "per_spot_frame_weights": None,
            }

        # 7. 扩展结果字典（不覆盖 predict() 原有字段）
        results.update({
            "predicted_intensity_lp": lp_intensities,
            "per_frame_total_intensity": distribution_result["per_frame_total_intensity"],
            "per_spot_frame_assignment": distribution_result["per_spot_frame_assignment"],
            "per_spot_frame_weights": distribution_result["per_spot_frame_weights"],
            "rocking_widths_deg": rocking_widths,
            "d_spacings": d_spacings,
        })

        return results

    def get_cctbx_miller_info(self):
        """
        返回 cctbx 从 CIF/晶胞计算出的全部 hkl 及其强度信息。

        返回:
            dict，包含:
                - hkls: ndarray, 形状 (N, 3)，全部 Miller 指数
                - intensities: ndarray, 形状 (N,)，|F|² 强度
                - d_spacings: ndarray, 形状 (N,)，d 间距（Å）
                - n_reflections: int，反射总数
                - intensity_available: bool，强度是否可用
                - message: str，状态说明
        """
        if self._crystal_model is None or not self._intensity_available:
            return {
                "hkls": np.zeros((0, 3), dtype=int),
                "intensities": np.zeros(0, dtype=float),
                "d_spacings": np.zeros(0, dtype=float),
                "n_reflections": 0,
                "intensity_available": False,
                "message": "未加载 CIF 或强度不可用",
            }

        miller_set = self._crystal_model._miller_set
        if miller_set is None:
            return {
                "hkls": np.zeros((0, 3), dtype=int),
                "intensities": np.zeros(0, dtype=float),
                "d_spacings": np.zeros(0, dtype=float),
                "n_reflections": 0,
                "intensity_available": False,
                "message": "CrystalModel 中未生成 miller_set",
            }

        hkls = np.array(miller_set.indices(), dtype=int)
        intensities = self.intensities  # 与 RotationPredictor 中保存的强度一致
        d_spacings = miller_set.d_spacings().data().as_numpy_array()

        return {
            "hkls": hkls,
            "intensities": intensities,
            "d_spacings": d_spacings,
            "n_reflections": len(hkls),
            "intensity_available": True,
            "message": "OK",
        }

    def compare_cctbx_dials_hkls(self):
        """
        对比 cctbx 计算的全部 hkl 与 DIALS 预测的 hkl 的匹配情况。

        返回:
            dict，包含:
                - n_cctbx: cctbx hkl 数量
                - n_dials: DIALS hkl 数量
                - n_matched: 匹配数量
                - n_unmatched_dials: DIALS 中未在 cctbx 中找到的数量
                - unmatched_dials_hkls: 未匹配的 DIALS hkl 示例（前 20 个）
                - match_ratio: 匹配比例
                - message: 状态说明
        """
        cctbx_info = self.get_cctbx_miller_info()
        if not cctbx_info["intensity_available"]:
            return {
                "n_cctbx": 0,
                "n_dials": 0,
                "n_matched": 0,
                "n_unmatched_dials": 0,
                "unmatched_dials_hkls": np.zeros((0, 3), dtype=int),
                "match_ratio": 0.0,
                "message": cctbx_info["message"],
            }

        if self.predicted_hkls is None:
            return {
                "n_cctbx": cctbx_info["n_reflections"],
                "n_dials": 0,
                "n_matched": 0,
                "n_unmatched_dials": 0,
                "unmatched_dials_hkls": np.zeros((0, 3), dtype=int),
                "match_ratio": 0.0,
                "message": "尚未调用 predict() 或 predict_with_intensity()",
            }

        cctbx_set = set(map(tuple, cctbx_info["hkls"]))
        dials_hkls = self.predicted_hkls.T
        n_dials = len(dials_hkls)

        matched = 0
        unmatched = []
        for hkl in dials_hkls:
            key = tuple(hkl)
            if key in cctbx_set:
                matched += 1
            else:
                unmatched.append(hkl)

        unmatched_array = (
            np.array(unmatched[:20], dtype=int)
            if unmatched else np.zeros((0, 3), dtype=int)
        )

        return {
            "n_cctbx": len(cctbx_set),
            "n_dials": n_dials,
            "n_matched": matched,
            "n_unmatched_dials": len(unmatched),
            "unmatched_dials_hkls": unmatched_array,
            "match_ratio": matched / n_dials if n_dials > 0 else 0.0,
            "message": "OK",
        }

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------
    def _validate_config(self):
        """检查配置字典是否包含所有必需键，并验证各类输入。"""
        required = ["d_min", "d_max", "num_images"]
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"配置缺少必需键: {missing}")

        # 晶体输入二选一
        has_cif = "cif_path" in self.config and self.config["cif_path"] is not None
        has_manual_cell = (
            "unit_cell_parameters" in self.config and
            "space_group_symbol" in self.config and
            self.config["unit_cell_parameters"] is not None and
            self.config["space_group_symbol"] is not None
        )
        if not (has_cif or has_manual_cell):
            raise ValueError(
                "config 中必须提供 cif_path 或 unit_cell_parameters + space_group_symbol"
            )

        # 探测器输入三选一
        has_poni_path = "poni_path" in self.config and self.config["poni_path"] is not None
        has_poni_dict = "poni" in self.config and self.config["poni"] is not None
        has_fit2d = "fit2d" in self.config and self.config["fit2d"] is not None
        if not (has_poni_path or has_poni_dict or has_fit2d):
            raise ValueError(
                "config 中必须提供 poni_path、poni 或 fit2d"
            )

        # 取向输入三选一
        has_angle_initial = (
            "angle1_deg" in self.config and
            "angle2_deg" in self.config and
            self.config["angle1_deg"] is not None and
            self.config["angle2_deg"] is not None
        )
        has_orientation_matrix = (
            "orientation_matrix" in self.config and
            self.config["orientation_matrix"] is not None
        )
        has_euler_angles = (
            "euler_angles" in self.config and
            self.config["euler_angles"] is not None
        )
        if not (has_angle_initial or has_orientation_matrix or has_euler_angles):
            raise ValueError(
                "config 中必须提供 angle1_deg + angle2_deg、orientation_matrix 或 euler_angles"
            )

    def report_status(self):
        """打印当前预测器的状态参数。"""
        print("=" * 60)
        print("RotationPredictor 状态报告")
        print("=" * 60)
        print(f"晶体输入模式: {self._crystal_input_mode}")
        print(f"  - CIF 已加载: {self._cif_loaded}")
        print(f"探测器输入模式: {self._detector_input_mode}")
        print(f"波长来源: {self._wavelength_source}")
        print(f"取向输入模式: {self._orientation_input_mode}")
        print(f"结构因子强度可用: {self._intensity_available}")
        print(f"晶胞参数: {self.unit_cell.parameters() if self.unit_cell else None}")
        print(f"空间群: {self.space_group_symbol}")
        print(f"波长: {self.wavelength_angstrom} Å")
        end_angle = self.scan_start_deg + self.oscillation_range
        print(f"扫描范围: {self.scan_start_deg}° ~ {end_angle}°")
        print(f"扫描帧数: {self.num_images}")
        if self.predicted is not None:
            print(f"预测反射总数: {len(self.predicted)}")
        print("=" * 60)

    def _resolve_path(self, path):
        """将相对路径解析为绝对路径。"""
        if os.path.isabs(path):
            return path
        # 相对于项目根目录
        return os.path.abspath(os.path.join(_project_root, path))

    @staticmethod
    def _resolve_static_path(path):
        """解析静态路径（用于 from_yaml）。"""
        if os.path.isabs(path):
            return path
        return os.path.abspath(path)
