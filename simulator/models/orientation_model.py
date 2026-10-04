"""
orientation_model.py

负责晶体取向相关的计算：
- 由用户给定的 U 矩阵、欧拉角或旋转轴+角度构造 U 矩阵
- 将晶体倒空间矢量 q_crystal 转到实验室坐标系 q_lab
- 组合 B 矩阵得到 UB 矩阵

模块间数据传递统一使用 dict。
"""
import warnings
import numpy as np
from scipy.spatial.transform import Rotation

class OrientationModel:
    """
    取向模型，封装晶体取向旋转矩阵 U 和 UB 矩阵构造。
    """

    def __init__(self, config):
        """
        初始化取向模型。

        参数:
            config (dict): 配置字典，支持以下键（按优先级依次使用）：
                - 'orientation_matrix' (ndarray): 直接给定 3×3 U 矩阵
                - 'euler_angles' (list/tuple): [omega, chi, phi]，单位度
                - 'rotation_axis' + 'rotation_angle': 绕指定轴旋转
                以上皆无时，默认使用 3×3 恒等矩阵（对应第 0 帧）。
        """
        if config is None:
            config = {}
        self.config = dict(config)
        self._input_mode = None      # "orientation_matrix" | "euler_angles" | "rotation_axis" | "rotation_predictor_angles" | "default"

        if 'orientation_matrix' in self.config and self.config['orientation_matrix'] is not None:
            self._set_u_matrix(self.config['orientation_matrix'])
            self._input_mode = "orientation_matrix"
        elif 'euler_angles' in self.config and self.config['euler_angles'] is not None:
            self._build_from_euler_angles(self.config['euler_angles'])
            self._input_mode = "euler_angles"
        elif self._has_rotation_predictor_angles():
            self._build_from_rotation_predictor_angles()
            self._input_mode = "rotation_predictor_angles"
        elif 'rotation_axis' in self.config and self.config['rotation_axis'] is not None:
            angle = self.config.get('rotation_angle', 0.0)
            self._build_from_rotation_axis(self.config['rotation_axis'], angle)
            self._input_mode = "rotation_axis"
        else:
            self.u_matrix = np.eye(3)
            self._input_mode = "default"

    def _set_u_matrix(self, u_matrix):
        """
        设置并校验 U 矩阵。

        参数:
            u_matrix (ndarray): 3×3 矩阵。

        异常:
            ValueError: 矩阵不是 3×3 或不满足正交性。
        """
        u_matrix = np.asarray(u_matrix, dtype=float)
        if u_matrix.shape != (3, 3):
            raise ValueError(f"orientation_matrix 必须是 3×3 矩阵，当前形状: {u_matrix.shape}")

        # 正交性校验：U^T · U ≈ I
        identity_check = u_matrix.T @ u_matrix
        if not np.allclose(identity_check, np.eye(3), atol=1e-6):
            raise ValueError("orientation_matrix 不是正交矩阵，U^T · U 与单位矩阵差异过大")

        # 行列式应为 +1（正常旋转），允许微小误差
        if np.linalg.det(u_matrix) < 0:
            warnings.warn("orientation_matrix 行列式为负，可能是反射而非纯旋转")

        self.u_matrix = u_matrix

    def _build_from_euler_angles(self, euler_angles):
        """
        根据欧拉角 [omega, chi, phi] 构造 U 矩阵。

        参数:
            euler_angles (list/tuple): [omega, chi, phi]，单位度。

        异常:
            ValueError: 输入不是长度为 3 的序列。
        """
        euler_angles = np.asarray(euler_angles, dtype=float)
        if euler_angles.shape != (3,):
            raise ValueError(f"euler_angles 必须是长度为 3 的序列，当前形状: {euler_angles.shape}")

        # 通用公式：YXZ 主动旋转
        # omega: 绕 Y 轴（竖直方向）
        # chi:   绕 X 轴（水平方向）
        # phi:   绕 Z 轴（光束方向）
        rotation = Rotation.from_euler('YXZ', euler_angles, degrees=True)
        self.u_matrix = rotation.as_matrix()

    def _build_from_rotation_axis(self, axis, angle_deg):
        """
        根据旋转轴和旋转角度构造 U 矩阵。

        参数:
            axis (list/ndarray): 旋转轴方向，不必单位化，例如 [0, 1, 0]。
            angle_deg (float): 旋转角度，单位度。右手定则。

        异常:
            ValueError: 旋转轴为零向量。
        """
        axis = np.asarray(axis, dtype=float)
        if axis.shape != (3,):
            raise ValueError(f"rotation_axis 必须是长度为 3 的向量，当前形状: {axis.shape}")

        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("rotation_axis 不能为零向量")

        axis_normalized = axis / norm
        angle_rad = np.radians(angle_deg)

        # 通用公式：旋转向量 = 角度 × 单位轴
        rotation = Rotation.from_rotvec(angle_rad * axis_normalized)
        self.u_matrix = rotation.as_matrix()

    def _has_rotation_predictor_angles(self):
        """
        判断 config 中是否包含 rotation_predictor 风格的初始取向参数。

        必需键：'angle1_deg'、'angle2_deg'。
        可选键：'rotation_angle_deg'、'rotation_axis'（缺失时默认 0° 绕 Y 轴）。
        """
        return (
            'angle1_deg' in self.config and self.config['angle1_deg'] is not None and
            'angle2_deg' in self.config and self.config['angle2_deg'] is not None
        )

    def _build_from_rotation_predictor_angles(self):
        """
        根据 rotation_predictor 风格的参数构造 U 矩阵。

        约定：
            - angle1_deg: 初始绕 X 轴旋转角
            - angle2_deg: 初始绕 Z 轴旋转角
            - rotation_angle_deg: 扫描起始前绕 rotation_axis 的额外旋转角（可选，默认 0）
            - rotation_axis: 额外旋转轴（可选，默认 [0, 1, 0]）

        初始取向采用主动旋转、固定轴（外禀）顺序：先绕 X，再绕 Z。
        随后叠加绕 rotation_axis 的额外旋转。
        """
        angle1_deg = float(self.config['angle1_deg'])
        angle2_deg = float(self.config['angle2_deg'])
        rotation_angle_deg = float(self.config.get('rotation_angle_deg', 0.0))
        rotation_axis = self.config.get('rotation_axis', [0.0, 1.0, 0.0])

        R_x = Rotation.from_euler("X", angle1_deg, degrees=True)
        R_z = Rotation.from_euler("Z", angle2_deg, degrees=True)
        U_initial = (R_z * R_x).as_matrix()

        axis = np.asarray(rotation_axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("rotation_axis 不能为零向量")
        axis_normalized = axis / norm

        rotation = Rotation.from_rotvec(
            np.radians(rotation_angle_deg) * axis_normalized
        )
        self.u_matrix = rotation.as_matrix() @ U_initial

    @classmethod
    def identity(cls):
        """
        创建恒等取向模型（第 0 帧默认取向）。

        返回:
            OrientationModel: u_matrix 为单位矩阵的取向模型实例。
        """
        return cls({})

    @classmethod
    def from_u_matrix(cls, u_matrix):
        """
        直接由 U 矩阵创建 OrientationModel。

        参数:
            u_matrix (ndarray): 3×3 旋转矩阵。

        返回:
            OrientationModel: 取向模型实例。
        """
        config = {'orientation_matrix': u_matrix}
        return cls(config)

    @classmethod
    def from_euler_angles(cls, euler_angles):
        """
        由欧拉角 [omega, chi, phi] 创建 OrientationModel。

        参数:
            euler_angles (list/tuple): [omega, chi, phi]，单位度。
                omega: 绕 Y 轴（竖直方向）
                chi:   绕 X 轴（水平方向）
                phi:   绕 Z 轴（光束方向）

        返回:
            OrientationModel: 取向模型实例。
        """
        config = {'euler_angles': euler_angles}
        return cls(config)

    @classmethod
    def from_rotation_axis(cls, axis, angle_deg):
        """
        由旋转轴和旋转角度创建 OrientationModel。

        参数:
            axis (list or ndarray): 旋转轴方向，例如 [0, 1, 0]。
            angle_deg (float): 旋转角度（度）。

        返回:
            OrientationModel: 取向模型实例。
        """
        config = {
            'rotation_axis': axis,
            'rotation_angle': angle_deg,
        }
        return cls(config)

    def get_u_matrix(self):
        """
        获取当前 U 矩阵。

        返回:
            dict: 包含 'u_matrix'（3×3 ndarray）的字典。
        """
        return {
            'u_matrix': self.u_matrix.copy(),
        }

    def apply_to_q_crystal(self, q_crystal):
        """
        将晶体倒空间矢量 q_crystal 转到实验室坐标系 q_lab。

        参数:
            q_crystal (ndarray): 形状 (3, N) 或 (N, 3)，单位 Å⁻¹，含 2π。

        返回:
            dict: 包含 'q_lab'（形状 (3, N)）的字典。
                  顺序与输入相同，为 [qy, qx, qz]。
        """
        q_crystal = np.asarray(q_crystal, dtype=float)

        if q_crystal.ndim != 2:
            raise ValueError(f"q_crystal 必须是 2D 数组，当前维度: {q_crystal.ndim}")

        # 统一转为 (3, N)
        if q_crystal.shape[0] == 3:
            q_crystal_3n = q_crystal
        elif q_crystal.shape[1] == 3:
            q_crystal_3n = q_crystal.T
        else:
            raise ValueError(
                f"q_crystal 必须有一个维度为 3，当前形状: {q_crystal.shape}"
            )

        q_lab = self.u_matrix @ q_crystal_3n

        return {
            'q_lab': q_lab,
        }

    def build_ub_matrix(self, b_matrix):
        """
        由 B 矩阵和当前 U 矩阵构造 UB 矩阵。

        参数:
            b_matrix (ndarray): 3×3 B 矩阵（含 2π 因子）。

        返回:
            dict: 包含 'ub_matrix'（3×3 ndarray）的字典。
                    UB = U · B，直接作用于 hkl 得到 q_lab。
        """
        b_matrix = np.asarray(b_matrix, dtype=float)
        if b_matrix.shape != (3, 3):
            raise ValueError(f"b_matrix 必须是 3×3 矩阵，当前形状: {b_matrix.shape}")

        ub_matrix = self.u_matrix @ b_matrix

        return {
            'ub_matrix': ub_matrix,
        }

    def build_dials_a_matrix(self, b_matrix_no_2pi):
        """
        由当前 U 矩阵和不含 2π 的 B 矩阵构造 DIALS 的 A 矩阵（UB）。

        DIALS 内部 Ewald 条件使用 1/λ，因此其 B 矩阵不含 2π 因子。
        本方法直接返回 U · B_no_2pi，作为 dxtbx.model.Crystal 的 setting matrix。

        参数:
            b_matrix_no_2pi (ndarray): 3×3 B 矩阵，不含 2π 因子。

        返回:
            dict: 包含 'a_matrix'（3×3 ndarray）的字典。
        """
        b_matrix_no_2pi = np.asarray(b_matrix_no_2pi, dtype=float)
        if b_matrix_no_2pi.shape != (3, 3):
            raise ValueError(
                f"b_matrix_no_2pi 必须是 3×3 矩阵，当前形状: {b_matrix_no_2pi.shape}"
            )

        a_matrix = self.u_matrix @ b_matrix_no_2pi
        return {
            'a_matrix': a_matrix,
        }

    def get_input_mode(self):
        """返回取向输入模式。"""
        return self._input_mode

    # ----------------------------------------------------------------------
    # T1-FZ：基本区投影（薄封装 orix.map_into_symmetry_reduced_zone）
    # ----------------------------------------------------------------------

    @staticmethod
    def _get_orix_symmetry(point_group="m-3m", include_inversion=True):
        """
        将 H-M 点群符号映射到 orix 的 Symmetry 对象。

        参数:
            point_group (str): 点群 H-M 符号，如 'm-3m'、'432'、'4/mmm'。
                              必须与 orix Phase(point_group=...) 接受的格式一致。
            include_inversion (bool):
                - True  : 完整点群（含反演），如 m-3m -> 48 个操作
                - False : 仅 proper rotation 子群，如 m-3m -> 432 (24 个操作)

        返回:
            orix.quaternion.Symmetry: 对应的 orix Symmetry 对象。
        """
        from orix.crystal_map import Phase
        # cctbx 返回的 H-M 短符号（如 "m"、"2"）不被 orix 接受，
        # 必须映射到 orix 完整位置符号。单斜的三个方向 (m11/1m1/11m, 211/121/112)
        # 对 FZ 投影等价；选 1m1 / 121 作为 c 轴惯例。
        orix_pg = {
            # 单斜（orix 区分三个方向，任意一个即可）
            "2":   "211",
            "m":   "1m1",
            "2/m": "2/m",
            # 六方 D3h 的 -62m 设定（cctbx 保留空间群设定返回，
            # 如 P-62m No.189）；orix 只接受 -6m2 设定。
            # 两者为同一点群的两种 H-M 设定，proper rotation 子群同为 622，FZ 等价。
            "-62m": "-6m2",
        }.get(point_group, point_group)
        full_sym = Phase(point_group=orix_pg).point_group
        if include_inversion:
            return full_sym
        # 仅取 proper rotation 子群（去掉反演）
        return full_sym.proper_subgroup

    def project_to_fundamental_zone(self, point_group="m-3m"):
        """
        将 self.u_matrix 投影到点群的基本区。

        实现思路：
            晶体对称等价关系：U ~ U·P（P 是晶体点群操作，右乘）。
            基本区投影：枚举所有 proper rotation P_i，找使 U·P_i 最接近
            单位矩阵 I 的代表，即 min ||U·P_i - I||_F。

        参数:
            point_group (str): 点群 H-M 符号。默认 'm-3m'。

        返回:
            dict: 包含以下键：
                - 'u_matrix': 投影后的 3x3 ndarray
                - 'point_group': 使用的点群符号
                - 'was_in_fundamental_zone': 投影前是否已在基本区
                  （通过比较投影前后 U 矩阵判定）
        """
        sym = self._get_orix_symmetry(point_group, include_inversion=False)
        P_list = sym.to_matrix()  # (|P_proper|, 3, 3)
        u_in = self.u_matrix.copy()

        best_frob = float('inf')
        u_out = None
        for P in P_list:
            U_candidate = u_in @ P          # 右乘：U_new = U·P（晶体对称操作）
            frob = np.linalg.norm(U_candidate - np.eye(3), 'fro')
            if frob < best_frob:
                best_frob = frob
                u_out = U_candidate.copy()

        # 确保 det = +1（纯旋转）
        if np.linalg.det(u_out) < 0:
            u_out = -u_out

        # 投影前是否已在基本区
        was_in_fz = bool(np.allclose(u_out, u_in, atol=1e-10))

        # 校验
        self._set_u_matrix(u_out)
        self._input_mode = "fundamental_zone_projected"

        return {
            "u_matrix": self.u_matrix.copy(),
            "point_group": point_group,
            "was_in_fundamental_zone": was_in_fz,
        }

    # ----------------------------------------------------------------------
    # T2-RO：随机取向（薄封装 orix.Rotation.random）
    # ----------------------------------------------------------------------

    @classmethod
    def random_orientation(cls, output_format="matrix", seed=None, point_group=None):
        """
        在 SO(3) 上按 Haar 测度采样一个随机取向，可选地投影到点群基本区。

        实现思路：
            1. orix.Rotation.random() 已是 SO(3) Haar 测度均匀采样
            2. 转 3×3 矩阵
            3. 可选：调用本类的 project_to_fundamental_zone() 做 FZ 投影
            4. 派生欧拉角（YXZ 约定，与项目一致）和轴角
            5. 按 output_format 组装返回值

        参数:
            output_format (str): 输出格式。可选：
                - 'matrix'      : 仅返回 U 矩阵
                - 'euler'       : 返回 U 矩阵 + 欧拉角 [omega, chi, phi]
                - 'axis_angle'  : 返回 U 矩阵 + 旋转轴 + 旋转角
            seed (int, optional): 随机种子，保证可复现
            point_group (str, optional): 若提供则自动做基本区投影，如 'm-3m'

        返回:
            dict: 包含以下键：
                - 'u_matrix': 3x3 ndarray（必有）
                - 'euler_angles': (3,) ndarray [omega, chi, phi]，度（output_format 含 'euler' 时）
                - 'rotation_axis': (3,) ndarray 单位矢量（output_format 含 'axis_angle' 时）
                - 'rotation_angle': float，度（output_format 含 'axis_angle' 时）
                - 'format': 输入的 output_format
                - 'seed': 随机种子
                - 'point_group': 使用的点群符号
        """
        from orix.quaternion import Rotation
        from scipy.spatial.transform import Rotation as R_scipy

        # 1. orix Haar 测度采样（orix 0.13.0 的 random() 内部用 np.random.uniform，
        #    种子通过 np.random.seed() 控制，不接受 random_state 参数）
        if seed is not None:
            np.random.seed(seed)
        R = Rotation.random((1,))
        u_matrix = np.squeeze(R.to_matrix())
        # 记录投影前的原始随机 U（便于对比"基本区归一化"的效果）
        u_matrix_before_fz = u_matrix.copy()

        # 2. 可选：基本区投影
        if point_group is not None:
            # 直接通过实例化投影（复用 T1 的逻辑）
            tmp = cls.from_u_matrix(u_matrix)
            proj_result = tmp.project_to_fundamental_zone(point_group)
            u_matrix = proj_result["u_matrix"]
            was_projected = (not np.allclose(u_matrix, u_matrix_before_fz, atol=1e-6))
            # 投影后重新从 3×3 构造 Rotation 用于派生格式
            R = Rotation.from_matrix(u_matrix)
        else:
            was_projected = False

        # 3. 按 output_format 派生所需数据
        result = {
            "u_matrix": u_matrix,
            "u_matrix_before_fz": u_matrix_before_fz,
            "was_projected_to_fz": was_projected,
            "format": output_format,
            "seed": seed,
            "point_group": point_group,
        }

        if output_format in ("euler", "all"):
            # 项目约定 YXZ 主动旋转：[omega, chi, phi]
            euler_yxz = R_scipy.from_matrix(u_matrix).as_euler("YXZ", degrees=True)
            result["euler_angles"] = euler_yxz  # (3,)

        if output_format in ("axis_angle", "all"):
            # 轴角：orix 0.13.0 的 R.axis / R.angle 返回 Vector3d 对象，
            #      必须用 .data 取原始 numpy 数组才能参与数学运算
            try:
                axis_raw = R.axis.data if hasattr(R.axis, "data") else np.asarray(R.axis)
                axis = np.squeeze(np.asarray(axis_raw))
                angle_raw = R.angle.data if hasattr(R.angle, "data") else np.asarray(R.angle)
                angle_deg = float(np.squeeze(np.asarray(angle_raw))) * (180.0 / np.pi)
            except (AttributeError, TypeError, ValueError):
                # 备选：scipy 派生
                rotvec = R_scipy.from_matrix(u_matrix).as_rotvec()
                angle = np.linalg.norm(rotvec)
                if angle < 1e-10:
                    axis = np.array([0.0, 0.0, 1.0])
                else:
                    axis = rotvec / angle
                angle_deg = float(angle) * (180.0 / np.pi)
            result["rotation_axis"] = axis
            result["rotation_angle"] = angle_deg

        return result

