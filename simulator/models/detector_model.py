"""
detector_model.py

负责所有 pyFAI 相关的探测器几何计算：
- 从 PONI 字典或文件创建探测器
- 计算每个像素的 3D 实验室坐标
- 计算每个像素的散射矢量 q

模块间数据传递统一使用 dict。
"""
import os

import numpy as np
import pyFAI
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from dxtbx.model.detector import DetectorFactory
from scitbx import matrix


class DetectorModel:
    """
    探测器模型，封装 pyFAI 的 AzimuthalIntegrator 和 Detector。
    """

    def __init__(self, config):
        """
        初始化探测器模型。

        参数:
            config (dict): 配置字典，支持以下键（按优先级依次使用）：
                - 'poni' (dict): PONI 参数字典
                - 'poni_path' (str): PONI 文件路径
                - 'fit2d' (dict): Fit2D 风格参数字典
        """
        self.config = config
        self.ai = None
        self.detector = None
        self.poni_dict = None
        self.fit2d_dict = None
        self._input_mode = None      # "poni_file" | "poni_dict" | "fit2d"

        if 'poni' in config and config['poni'] is not None:
            self._build_from_poni_dict(config['poni'])
        elif 'poni_path' in config and config['poni_path'] is not None:
            self._build_from_poni_file(config['poni_path'])
        elif 'fit2d' in config and config['fit2d'] is not None:
            self._build_from_fit2d_dict(config['fit2d'])
        else:
            raise ValueError("config 中必须提供 'poni'、'poni_path' 或 'fit2d'")

    def _build_from_poni_dict(self, poni_dict):
        """根据 PONI 字典构造 AzimuthalIntegrator 和 Detector。"""
        self.poni_dict = dict(poni_dict)

        required_keys = ['dist', 'poni1', 'poni2', 'pixel1', 'pixel2', 'max_shape', 'wavelength']
        for key in required_keys:
            if key not in poni_dict:
                raise ValueError(f"PONI 字典缺少必需键: {key}")

        detector = Detector(
            pixel1=poni_dict['pixel1'],
            pixel2=poni_dict['pixel2'],
            max_shape=poni_dict['max_shape'],
        )

        ai = AzimuthalIntegrator(
            dist=poni_dict['dist'],
            poni1=poni_dict['poni1'],
            poni2=poni_dict['poni2'],
            rot1=poni_dict.get('rot1', 0.0),
            rot2=poni_dict.get('rot2', 0.0),
            rot3=poni_dict.get('rot3', 0.0),
            detector=detector,
            wavelength=poni_dict['wavelength'],
        )

        self.ai = ai
        self.detector = detector
        self._input_mode = "poni_dict"

    def _build_from_poni_file(self, poni_path):
        """根据 PONI 文件构造 AzimuthalIntegrator 和 Detector。"""
        if not os.path.exists(poni_path):
            raise FileNotFoundError(f"PONI 文件不存在: {poni_path}")

        self.ai = pyFAI.load(poni_path)
        self.detector = self.ai.detector
        self.poni_dict = self.ai.get_config()
        self._input_mode = "poni_file"

    @classmethod
    def from_poni_dict(cls, poni_dict):
        """
        从 PONI 参数字典创建 DetectorModel。

        参数:
            poni_dict (dict): PONI 参数字典，键包括 dist、poni1、poni2、rot1、rot2、rot3、pixel1、pixel2、max_shape、wavelength 等。

        返回:
            DetectorModel: 探测器模型实例。
        """
        config = {'poni': poni_dict}
        return cls(config)

    @classmethod
    def from_poni_file(cls, poni_path):
        """
        从 PONI 文件创建 DetectorModel。

        参数:
            poni_path (str): PONI 文件路径。

        返回:
            DetectorModel: 探测器模型实例。
        """
        config = {'poni_path': poni_path}
        return cls(config)

    def _build_from_fit2d_dict(self, fit2d_dict):
        """
        根据 Fit2D 风格参数字典构造 AzimuthalIntegrator 和 Detector。

        Fit2D 参数说明：
            - direct_dist / directDist: 样品到探测器距离，单位 mm
            - center_x / centerX, center_y / centerY: 直通光斑中心像素坐标
            - pixel_x / pixelX, pixel_y / pixelY: 像素尺寸，单位 micron
            - tilt: 探测器倾斜角，单位 degree（可选，默认 0）
            - tilt_plan_rotation / tiltPlanRotation: 倾斜面旋转角，单位 degree（可选，默认 0）
            - wavelength: X 射线波长，单位米（可选；缺失时尝试使用 config 中的 wavelength_angstrom）
            - max_shape: 探测器形状 (ny, nx)（可选；缺失时根据中心坐标自动估算）

        注意：pyFAI 的 setFit2D 要求 wavelength 单位为 Å，因此内部会进行换算。
        """
        self.fit2d_dict = dict(fit2d_dict)

        # 统一键名：支持 camelCase 与 snake_case
        key_map = {
            'directDist': 'direct_dist',
            'centerX': 'center_x',
            'centerY': 'center_y',
            'pixelX': 'pixel_x',
            'pixelY': 'pixel_y',
            'tiltPlanRotation': 'tilt_plan_rotation',
        }
        normalized = {}
        for key, value in fit2d_dict.items():
            normalized[key_map.get(key, key)] = value

        required_keys = ['direct_dist', 'center_x', 'center_y', 'pixel_x', 'pixel_y']
        for key in required_keys:
            if key not in normalized:
                raise ValueError(f"fit2d 字典缺少必需键: {key}")

        # 波长：优先使用 fit2d 内的 wavelength（米），否则尝试 config 中的 wavelength_angstrom
        if 'wavelength' in normalized and normalized['wavelength'] is not None:
            wavelength_m = float(normalized['wavelength'])
        elif 'wavelength_angstrom' in self.config and self.config['wavelength_angstrom'] is not None:
            wavelength_m = float(self.config['wavelength_angstrom']) * 1e-10
        else:
            raise ValueError(
                "fit2d 字典中必须提供 wavelength（单位米），"
                "或在 config 顶层提供 wavelength_angstrom（单位 Å）"
            )

        # 探测器形状：优先使用 fit2d 内的 max_shape，否则根据中心坐标估算
        if 'max_shape' in normalized and normalized['max_shape'] is not None:
            max_shape = tuple(normalized['max_shape'])
        else:
            # 根据光束中心位置估算探测器尺寸（取 2 倍中心并向上取整）
            nx = int(2 * np.ceil(normalized['center_x']))
            ny = int(2 * np.ceil(normalized['center_y']))
            max_shape = (ny, nx)
            print(
                f"[DetectorModel] fit2d 未提供 max_shape，"
                f"根据中心坐标估算为 {max_shape}"
            )

        # Fit2D 使用 micron，Detector 使用 meter
        pixel1 = normalized['pixel_y'] * 1e-6
        pixel2 = normalized['pixel_x'] * 1e-6

        detector = Detector(
            pixel1=pixel1,
            pixel2=pixel2,
            max_shape=max_shape,
        )

        # pyFAI.setFit2D 要求波长单位为 Å，因此从米换算
        wavelength_angstrom = wavelength_m * 1e10

        ai = AzimuthalIntegrator()
        ai.setFit2D(
            directDist=normalized['direct_dist'],
            centerX=normalized['center_x'],
            centerY=normalized['center_y'],
            tilt=normalized.get('tilt', 0.0),
            tiltPlanRotation=normalized.get('tilt_plan_rotation', 0.0),
            pixelX=normalized['pixel_x'],
            pixelY=normalized['pixel_y'],
            detector=detector,
            wavelength=wavelength_angstrom,
        )

        self.ai = ai
        self.detector = detector
        self._input_mode = "fit2d"

    @classmethod
    def from_fit2d_dict(cls, fit2d_dict):
        """
        从 Fit2D 风格参数字典创建 DetectorModel。

        参数:
            fit2d_dict (dict): Fit2D 参数字典，键包括：
                direct_dist, center_x, center_y, pixel_x, pixel_y,
                tilt, tilt_plan_rotation, wavelength, max_shape。

        返回:
            DetectorModel: 探测器模型实例。
        """
        config = {'fit2d': fit2d_dict}
        return cls(config)

    def get_pixel_positions(self):
        """
        获取探测器每个像素的 3D 实验室坐标。

        返回:
            dict: 包含 'positions'（形状为 (ny, nx, 3) 的 ndarray）和 'shape' 的字典。
                  positions 最后一维顺序为 [axis1, axis2, axis3] = [y, x, z]，单位米。
        """
        if self.ai is None:
            raise RuntimeError("DetectorModel 尚未初始化，无法获取像素坐标")

        # pyFAI 的 position_array() 返回最后一维顺序为 [z, y, x]，
        # 项目内部统一使用 [y, x, z]，因此需要重新排列。
        positions_raw = self.ai.position_array()
        positions = np.stack([
            positions_raw[..., 1],  # axis1 = y
            positions_raw[..., 2],  # axis2 = x
            positions_raw[..., 0],  # axis3 = z
        ], axis=-1)

        shape = (positions.shape[0], positions.shape[1])

        return {
            'positions': positions,
            'shape': shape,
        }

    def get_pixel_q_vectors(self):
        """
        获取探测器每个像素的散射矢量 q（含 2π 因子，单位 Å⁻¹）。

        返回:
            dict: 包含 'q_vectors'（形状为 (ny, nx, 3) 的 ndarray）和 'shape' 的字典。
                  q_vectors 最后一维顺序为项目约定的 [qy, qx, qz]，
                  对应 pyFAI 的 [axis1, axis2, axis3] = [y, x, z]。
        """
        if self.ai is None:
            raise RuntimeError("DetectorModel 尚未初始化，无法获取 q 矢量")

        # 直接使用 pyFAI 内置单位获取实验室坐标系下的散射矢量分量，单位 Å⁻¹。
        # pyFAI 约定：
        #   qxgi -> axis2 = x = 水平方向
        #   qygi -> axis1 = y = 竖直方向
        #   qzgi -> axis3 = z = 光束方向
        # 项目内部约定顺序为 [qy, qx, qz]，因此只需重排，无需额外变号。
        qxgi = self.ai.array_from_unit(unit="qxgi_A^-1")
        qygi = self.ai.array_from_unit(unit="qygi_A^-1")
        qzgi = self.ai.array_from_unit(unit="qzgi_A^-1")

        q_vectors = np.stack([qygi, qxgi, qzgi], axis=-1)

        return {
            'q_vectors': q_vectors,
            'shape': (q_vectors.shape[0], q_vectors.shape[1]),
        }

    def get_pixel_q_magnitudes(self):
        """
        获取探测器每个像素的 q 模长（含 2π 因子，单位 Å⁻¹）。

        返回:
            dict: 包含 'q_magnitudes'（形状为 (ny, nx) 的 ndarray）的字典。
        """
        if self.ai is None:
            raise RuntimeError("DetectorModel 尚未初始化，无法获取 q 模长")

        # 复用 PT3 的 q 矢量结果
        q_vectors = self.get_pixel_q_vectors()['q_vectors']
        q_magnitudes = np.linalg.norm(q_vectors, axis=-1)

        return {
            'q_magnitudes': q_magnitudes,
        }

    def get_wavelength(self):
        """
        获取当前 X 射线波长（米）。

        返回:
            dict: 包含 'wavelength' 的字典。
        """
        if self.ai is None:
            raise RuntimeError("DetectorModel 尚未初始化，无法获取波长")
        return {
            'wavelength': self.ai.wavelength,
        }

    def get_input_mode(self):
        """返回探测器输入模式：'poni_file'、'poni_dict' 或 'fit2d'。"""
        return self._input_mode

    def get_wavelength_angstrom(self):
        """
        获取当前 X 射线波长（Å）。

        返回:
            dict: 包含 'wavelength_angstrom' 的字典。
        """
        if self.ai is None:
            raise RuntimeError("DetectorModel 尚未初始化，无法获取波长")
        return {
            'wavelength_angstrom': self.ai.get_wavelength() * 1e10,
        }

    def get_image_size(self):
        """
        获取探测器图像尺寸（像素）。

        返回:
            dict: 包含 'image_size' 的字典，格式为 (nx, ny)，即 (fast, slow)。
        """
        if self.ai is None:
            raise RuntimeError("DetectorModel 尚未初始化，无法获取图像尺寸")
        ny, nx = self.ai.detector.max_shape
        return {
            'image_size': (nx, ny),
        }

    def get_pixel_size_mm(self):
        """
        获取探测器像素尺寸（毫米）。

        返回:
            dict: 包含 'pixel_size' 的字典，格式为 (fast_size_mm, slow_size_mm)。
        """
        if self.ai is None:
            raise RuntimeError("DetectorModel 尚未初始化，无法获取像素尺寸")
        return {
            'pixel_size': (self.ai.pixel2 * 1e3, self.ai.pixel1 * 1e3),
        }

    def get_rotation_matrix_scitbx(self):
        """
        获取 pyFAI 旋转矩阵的 scitbx.matrix.sqr 形式。

        pyFAI 的 rotation_matrix() 返回 numpy.ndarray (3,3)，
        本方法将其展平为 list 后传入 scitbx.matrix.sqr 并转置，
        使矩阵表示探测器局部 [row, col, normal] -> 项目实验室 [x, y, z]。

        返回:
            dict: 包含 'rotation_matrix'（scitbx.matrix.sqr）的字典。
        """
        if self.ai is None:
            raise RuntimeError("DetectorModel 尚未初始化，无法获取旋转矩阵")
        R = matrix.sqr(self.ai.rotation_matrix().flatten().tolist()).transpose()
        return {
            'rotation_matrix': R,
        }

    def get_dials_detector(self):
        """
        构造并返回 dxtbx.model.detector 对象，供 DIALS 使用。

        使用与项目一致的坐标系（入射光沿 +Z），无需额外的 Mz 反射转换。

        返回:
            dict: 包含 'detector'（dxtbx.model.detector 对象）的字典。
        """
        if self.ai is None:
            raise RuntimeError("DetectorModel 尚未初始化，无法构造 DIALS 探测器")

        R = self.get_rotation_matrix_scitbx()['rotation_matrix']

        fast_axis = tuple(np.asarray(R * matrix.col((1, 0, 0))).flatten())
        slow_axis = tuple(np.asarray(R * matrix.col((0, 1, 0))).flatten())
        origin = tuple(np.asarray(
            matrix.col((0.0, 0.0, self.ai.dist)) +
            R * matrix.col((-self.ai.poni2, -self.ai.poni1, 0.0))
        ).flatten() * 1e3)

        image_size = self.get_image_size()['image_size']
        pixel_size = self.get_pixel_size_mm()['pixel_size']

        detector = DetectorFactory.make_detector(
            stype="SENSOR_PAD",
            fast_axis=fast_axis,
            slow_axis=slow_axis,
            origin=origin,
            pixel_size=pixel_size,
            image_size=image_size,
            trusted_range=(-1, 1e6),
            name="Panel0",
        )

        return {
            'detector': detector,
        }

    def q_to_pixel(self, q_lab):
        """
        将实验室坐标系下的散射矢量解析反演为探测器像素坐标（通用平面探测器版本）。

        该方法直接利用 pyFAI PONI 几何参数（dist、poni1、poni2、rot1、rot2、rot3、
        pixel1、pixel2、wavelength）求解射线与探测器平面的交点，再转回像素坐标。
        适用于任意旋转/倾斜的平面探测器，无需离散网格插值或优化。

        参数:
            q_lab (ndarray): 散射矢量，形状 (3,)、(3, N) 或 (N, 3)。
                             顺序为标准笛卡尔 [qx, qy, qz]，单位 Å⁻¹，含 2π。

        返回:
            dict: 包含 'pixel_coordinates' 的字典：
                  - 输入 (3,)   -> 输出 (2,)，[row, col]
                  - 输入 (3, N) -> 输出 (2, N)，每列 [row, col]
                  - 输入 (N, 3) -> 输出 (N, 2)，每行 [row, col]

        异常:
            ValueError: 当射线与探测器平面平行（无交点）时抛出。
        """
        if self.ai is None:
            raise RuntimeError("DetectorModel 尚未初始化，无法执行 q 到像素反演")

        q_lab = np.asarray(q_lab, dtype=float)

        # 记录原始布局并统一为 (3, N)
        if q_lab.ndim == 1:
            if q_lab.shape[0] != 3:
                raise ValueError(f"1D q_lab 长度必须为 3，当前: {q_lab.shape}")
            q_lab = q_lab.reshape(3, 1)
            layout = 'single'
        elif q_lab.ndim == 2:
            if q_lab.shape[0] == 3:
                layout = '3N'
            elif q_lab.shape[1] == 3:
                q_lab = q_lab.T
                layout = 'N3'
            else:
                raise ValueError(
                    f"2D q_lab 必须有一个维度为 3，当前形状: {q_lab.shape}"
                )
        else:
            raise ValueError(f"q_lab 必须是 1D 或 2D 数组，当前维度: {q_lab.ndim}")

        # 几何参数
        dist = self.ai.dist
        poni1 = self.ai.poni1
        poni2 = self.ai.poni2
        pixel1 = self.ai.pixel1
        pixel2 = self.ai.pixel2
        lam = self.ai.wavelength * 1e10  # 米 -> Å
        R = self.ai.rotation_matrix()    # 探测器局部坐标系 -> 实验室坐标系

        # 入射光束方向沿实验室 +z
        s0 = np.array([0.0, 0.0, 1.0])[:, None]
        # 出射方向：s1 = s0 + q/k，其中 k = 2π/λ
        s1 = s0 + q_lab * lam / (2.0 * np.pi)

        # 探测器平面法向（实验室系）和过点
        n = R[:, 2]
        P0 = np.array([0.0, 0.0, dist])

        # 射线 X(t) = t*s1 与平面 (X-P0)·n = 0 相交
        denom = np.dot(n, s1)
        if np.any(np.abs(denom) < 1e-15):
            raise ValueError("散射矢量对应的射线与探测器平面平行，无法求交点")

        t = np.dot(P0, n) / denom
        X = t[None, :] * s1

        # 交点转回探测器局部坐标系
        p_local = R.T @ (X - P0[:, None])

        # 局部坐标 (axis1, axis2) = (y, x) -> 像素 (row, col)
        rows = (p_local[0] + poni1) / pixel1
        cols = (p_local[1] + poni2) / pixel2

        pixel_coordinates = np.vstack([rows, cols])

        # 恢复原始布局
        if layout == 'single':
            pixel_coordinates = pixel_coordinates[:, 0]
        elif layout == 'N3':
            pixel_coordinates = pixel_coordinates.T

        return {
            'pixel_coordinates': pixel_coordinates,
        }
