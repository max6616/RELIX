"""
crystal_model.py

负责所有 cctbx 相关的晶体学计算：
- 读取 CIF 文件
- 生成 Miller 集合
- 计算结构因子与强度 I(hkl)
- 构造 B 矩阵

模块间数据传递统一使用 dict。
"""
import numpy as np
from cctbx import crystal, miller, sgtbx
from iotbx import cif

class CrystalModel:
    """
    晶体模型，封装 cctbx 的晶体结构、Miller 集、结构因子和 B 矩阵计算。
    """

    def __init__(self, config=None):
        """
        初始化晶体模型。

        参数:
            config (dict, optional): 配置字典，可包含 'cif_path'、'd_min'、'd_max' 等键。
        """
        self.config = config if config is not None else {}
        self._structure = None
        self._crystal_symmetry = None
        self._unit_cell = None
        self._miller_set = None

        # build_miller_set_with_intensities() 保存的全集数组，
        # 供 filter_miller_set_by_d_range() / get_intensity_dict() 复用
        self._hkl_array = None
        self._intensities = None
        self._d_spacings_full = None

        # 状态参数：记录晶体输入方式
        self._input_mode = None      # "cif" | "manual"
        self._cif_loaded = False
        self._intensity_available = False

    def load(self):
        """
        统一加载入口，根据 config 自动判断输入方式。

        优先使用 'cif_path'；若不存在则使用 'unit_cell_parameters' + 'space_group_symbol'。
        若 config 中指定了 'change_of_basis_op'，会在 load_cif() 之后自动应用基矢变换。

        返回:
            dict: 与 load_cif() 或 load_manual_cell() 返回字典一致。
        """
        cif_path = self.config.get('cif_path')
        if cif_path is not None:
            result = self.load_cif(cif_path)
            cb_op = self.config.get('change_of_basis_op')
            if cb_op is not None:
                result = self.change_basis(cb_op)
            return result

        unit_cell_parameters = self.config.get('unit_cell_parameters')
        space_group_symbol = self.config.get('space_group_symbol')
        if (unit_cell_parameters is not None and
                space_group_symbol is not None):
            return self.load_manual_cell(unit_cell_parameters, space_group_symbol)

        raise ValueError(
            "config 中必须提供 'cif_path' 或 'unit_cell_parameters + space_group_symbol'"
        )

    def load_cif(self, cif_path=None):
        """
        读取 CIF 文件并提取晶体结构。

        参数:
            cif_path (str, optional): CIF 文件路径。若未提供，则使用 self.config 中的 'cif_path'。

        返回:
            dict: 包含以下键的字典：
                - 'structure': xray.structure 对象
                - 'crystal_symmetry': crystal.symmetry 对象
                - 'unit_cell': unit_cell 对象
                - 'space_group_info': 空间群信息字符串
                - 'unit_cell_parameters': 晶胞参数元组 (a, b, c, alpha, beta, gamma)
        """
        if cif_path is None:
            cif_path = self.config.get('cif_path')

        if not cif_path:
            raise ValueError("必须提供 cif_path 参数或在 config 中设置 'cif_path'")

        cif_structures = cif.reader(file_path=cif_path).build_crystal_structures()
        if not cif_structures:
            raise ValueError(f"CIF 文件未包含晶体结构: {cif_path}")

        structure = list(cif_structures.values())[0]
        crystal_symmetry = structure.crystal_symmetry()
        unit_cell = crystal_symmetry.unit_cell()
        space_group_info = crystal_symmetry.space_group_info()
        unit_cell_parameters = unit_cell.parameters()

        self._structure = structure
        self._crystal_symmetry = crystal_symmetry
        self._unit_cell = unit_cell
        self._input_mode = "cif"
        self._cif_loaded = True

        return {
            'structure': structure,
            'crystal_symmetry': crystal_symmetry,
            'unit_cell': unit_cell,
            'space_group_info': str(space_group_info),
            'unit_cell_parameters': unit_cell_parameters,
        }

    def change_basis(self, cb_op):
        """
        对已加载的晶体应用基矢变换（change of basis）。

        必须在 load_cif() 之后、build_miller_set() 之前调用。
        用于将 CIF 的标准设定（如单斜唯一轴b）变换到实验设定（如XDS唯一轴a）。

        参数:
            cb_op: 可以是字符串（如 "b,a,c"）或 sgtbx.change_of_basis_op 对象。

        返回:
            dict: 与 load_cif() 返回格式相同，包含变换后的晶体信息。
        """
        if not self._cif_loaded:
            raise ValueError("change_basis() 只能在 load_cif() 之后调用")

        if isinstance(cb_op, str):
            cb_op = sgtbx.change_of_basis_op(cb_op)

        # 变换晶体结构（原子坐标也会随之变换）
        self._structure = self._structure.change_basis(cb_op)
        self._crystal_symmetry = self._structure.crystal_symmetry()
        self._unit_cell = self._crystal_symmetry.unit_cell()
        self._miller_set = None  # 重置，需要重新生成

        return {
            'structure': self._structure,
            'crystal_symmetry': self._crystal_symmetry,
            'unit_cell': self._unit_cell,
            'space_group_info': str(self._crystal_symmetry.space_group_info()),
            'unit_cell_parameters': self._unit_cell.parameters(),
        }

    def load_manual_cell(self, unit_cell_parameters=None, space_group_symbol=None):
        """
        根据手动输入的晶胞参数和空间群构造晶体对称性（无原子结构）。

        参数:
            unit_cell_parameters (list/tuple, optional): 晶胞参数 (a, b, c, alpha, beta, gamma)，单位 Å/度。
                若未提供，则使用 self.config 中的 'unit_cell_parameters'。
            space_group_symbol (str, optional): 空间群符号，例如 "P 1 2 1"。
                若未提供，则使用 self.config 中的 'space_group_symbol'。

        返回:
            dict: 包含以下键的字典：
                - 'structure': None（手动输入无原子结构）
                - 'crystal_symmetry': crystal.symmetry 对象
                - 'unit_cell': unit_cell 对象
                - 'space_group_info': 空间群信息字符串
                - 'unit_cell_parameters': 晶胞参数元组
        """
        if unit_cell_parameters is None:
            unit_cell_parameters = self.config.get('unit_cell_parameters')
        if space_group_symbol is None:
            space_group_symbol = self.config.get('space_group_symbol')

        if unit_cell_parameters is None or space_group_symbol is None:
            raise ValueError(
                "必须提供 unit_cell_parameters 和 space_group_symbol，"
                "或在 config 中设置对应键"
            )

        unit_cell_parameters = tuple(float(x) for x in unit_cell_parameters)
        space_group_symbol = str(space_group_symbol)

        crystal_symmetry = crystal.symmetry(
            unit_cell=unit_cell_parameters,
            space_group_symbol=space_group_symbol,
        )
        unit_cell = crystal_symmetry.unit_cell()
        space_group_info = crystal_symmetry.space_group_info()

        self._structure = None
        self._crystal_symmetry = crystal_symmetry
        self._unit_cell = unit_cell
        self._input_mode = "manual"
        self._cif_loaded = False

        return {
            'structure': None,
            'crystal_symmetry': crystal_symmetry,
            'unit_cell': unit_cell,
            'space_group_info': str(space_group_info),
            'unit_cell_parameters': unit_cell.parameters(),
        }

    def build_miller_set(self, d_min, d_max):
        """
        基于当前晶体结构生成 Miller 集合，并按 d 间距范围过滤。

        参数:
            d_min (float): 最小 d 间距（Å），对应最大分辨率。
            d_max (float): 最大 d 间距（Å），对应最小分辨率。

        返回:
            dict: 包含以下键的字典：
                - 'miller_set': cctbx miller.set 对象（已按 d_max 过滤）
                - 'hkl_array': Miller 指数数组，形状 (N, 3)，类型 int
                - 'd_spacings': d 间距数组，形状 (N,)，单位 Å
                - 'd_min': 输入的最小 d 间距
                - 'd_max': 输入的最大 d 间距
        """
        if self._crystal_symmetry is None:
            raise ValueError("必须先调用 load_cif() 加载晶体结构")

        if d_min <= 0 or d_max <= 0:
            raise ValueError("d_min 和 d_max 必须为正数")

        if d_min >= d_max:
            raise ValueError("d_min 必须小于 d_max")

        # 1. 生成所有 Miller 指数（到 d_min）
        full_miller_set = miller.build_set(
            crystal_symmetry=self._crystal_symmetry,
            anomalous_flag=False,
            d_min=d_min,
        )

        # 2. 按 d_max 过滤（保留 d <= d_max 的反射）
        d_spacings = full_miller_set.d_spacings().data()
        keep_mask = d_spacings <= d_max
        filtered_indices = full_miller_set.indices().select(keep_mask)

        filtered_miller_set = miller.set(
            crystal_symmetry=self._crystal_symmetry,
            indices=filtered_indices,
            anomalous_flag=False,
        )

        # 3. 转换为 numpy 数组
        hkl_array = np.array(
            [(h, k, l) for h, k, l in filtered_indices], dtype=int
        )

        d_spacings_filtered = d_spacings.select(keep_mask).as_numpy_array()

        self._miller_set = filtered_miller_set

        return {
            'miller_set': filtered_miller_set,
            'hkl_array': hkl_array,
            'd_spacings': d_spacings_filtered,
            'd_min': d_min,
            'd_max': d_max,
        }

    def compute_intensities(self):
        """
        计算当前 Miller 集合的结构因子 F(hkl) 和强度 I(hkl) = |F|²。

        返回:
            dict: 包含以下键的字典：
                - 'hkl_array': Miller 指数数组，形状 (N, 3)，类型 int
                - 'intensities': 强度数组 I(hkl)，形状 (N,)
                - 'f_calc': cctbx miller.array 对象（复结构因子）
                - 'miller_array': cctbx miller.array 对象（强度）
        """
        if self._structure is None:
            raise ValueError("必须先调用 load_cif() 加载晶体结构")
        if self._miller_set is None:
            raise ValueError("必须先调用 build_miller_set() 生成 Miller 集合")

        # 1. 计算复结构因子 F(hkl)
        f_calc = self._miller_set.structure_factors_from_scatterers(
            xray_structure=self._structure,
            algorithm="direct",
        ).f_calc()

        # 2. 转换为强度 I(hkl) = |F|²
        intensity_array = f_calc.as_intensity_array()

        # 3. 提取 numpy 数组
        hkl_array = np.array(
            [(h, k, l) for h, k, l in intensity_array.indices()], dtype=int
        )
        intensities = intensity_array.data().as_numpy_array()

        self._intensity_available = True

        return {
            'hkl_array': hkl_array,
            'intensities': intensities,
            'f_calc': f_calc,
            'miller_array': intensity_array,
        }

    def build_miller_set_with_intensities(self, d_min, d_max):
        """
        生成 Miller 集合并尝试计算强度，返回统一结果字典。

        若已加载 CIF（有原子结构），则计算真实结构因子强度；
        否则强度统一为 1.0，并标记强度不可用。

        参数:
            d_min (float): 最小 d 间距（Å）。
            d_max (float): 最大 d 间距（Å）。

        返回:
            dict: 包含以下键的字典：
                - 'hkl_array': Miller 指数数组，形状 (N, 3)
                - 'd_spacings': d 间距数组，形状 (N,)
                - 'intensities': 强度数组，形状 (N,)
                - 'intensity_available': 是否使用了真实结构因子强度
        """
        miller_result = self.build_miller_set(d_min=d_min, d_max=d_max)

        if self._structure is not None:
            intensity_result = self.compute_intensities()
            miller_array = intensity_result['miller_array']

            # 展开到 P1 空间群，生成所有对称等价反射
            miller_array_p1 = miller_array.expand_to_p1()

            # 显式添加 Friedel 对 (-h, -k, -l)，确保 DIALS 预测的负 hkl 也能匹配
            # 非反常散射近似下，Friedel 对强度相同
            p1_hkls = np.array(miller_array_p1.indices(), dtype=int)
            p1_intensities = miller_array_p1.data().as_numpy_array()

            friedel_hkls = -p1_hkls
            all_hkls = np.vstack([p1_hkls, friedel_hkls])
            all_intensities = np.concatenate([p1_intensities, p1_intensities])

            # 去重，保留 Friedel 对和 P1 展开结果
            unique_hkls, unique_indices = np.unique(
                all_hkls, axis=0, return_index=True
            )
            unique_intensities = all_intensities[unique_indices]

            # 同步更新内部保存的 miller_set，保证其他接口一致
            from cctbx.array_family import flex
            self._miller_set = miller.set(
                crystal_symmetry=self._crystal_symmetry,
                indices=flex.miller_index(
                    [(int(h), int(k), int(l)) for h, k, l in unique_hkls]
                ),
                anomalous_flag=True,
            )

            hkl_array = unique_hkls
            d_spacings = self._miller_set.d_spacings().data().as_numpy_array()
            intensities = unique_intensities
        else:
            intensities = np.ones(len(miller_result['hkl_array']))
            self._intensity_available = False
            hkl_array = miller_result['hkl_array']
            d_spacings = miller_result['d_spacings']

        # 保存全集数组，供子集过滤与查表字典复用
        self._hkl_array = hkl_array
        self._intensities = intensities
        self._d_spacings_full = d_spacings

        return {
            'hkl_array': hkl_array,
            'd_spacings': d_spacings,
            'intensities': intensities,
            'intensity_available': self._intensity_available,
        }

    def filter_miller_set_by_d_range(self, d_min, d_max):
        """
        在已构建的全集基础上按 d 间距范围过滤子集。

        纯 numpy 掩码操作，不重新计算结构因子。
        用于批量生成场景中同一晶体多次取向预测时的强度查表复用。

        必须先调用 build_miller_set_with_intensities() 构建全集。

        参数:
            d_min (float): 最小 d 间距（Å）。
            d_max (float): 最大 d 间距（Å）。

        返回:
            dict: 与 build_miller_set_with_intensities() 相同的键：
                - 'hkl_array', 'd_spacings', 'intensities', 'intensity_available'
        """
        if self._hkl_array is None:
            raise ValueError("必须先调用 build_miller_set_with_intensities() 构建全集")

        mask = (self._d_spacings_full >= d_min) & (self._d_spacings_full <= d_max)

        return {
            'hkl_array': self._hkl_array[mask],
            'd_spacings': self._d_spacings_full[mask],
            'intensities': self._intensities[mask],
            'intensity_available': self._intensity_available,
        }

    def get_intensity_dict(self):
        """
        返回全集 {hkl: intensity} 查表字典。

        用于批量预测场景：字典构建一次，多个取向的 RotationPredictor 共享，
        避免每个样本重复构建大字典。

        必须先调用 build_miller_set_with_intensities() 构建全集。

        返回:
            dict: 键为 (h, k, l) 元组，值为强度 float。
        """
        if self._hkl_array is None or self._intensities is None:
            raise ValueError("必须先调用 build_miller_set_with_intensities() 构建全集")

        return {
            tuple(hkl): float(intensity)
            for hkl, intensity in zip(self._hkl_array, self._intensities)
        }

    def get_crystal_info(self):
        """
        汇总当前晶体的结构信息（来自 CIF），用于批量生成 CSV 头注释。

        化学式按 Hill 系统排序（有 C 时 C、H 在前，其余字母序），
        原子计数 = occupancy × 位点重复度（含对称展开到整个晶胞）。

        返回:
            dict: 包含以下键：
                - chemical_formula (str or None): Hill 化学式；manual 模式为 None
                - n_atoms_cell (int or None): 晶胞内原子总数；manual 模式为 None
                - space_group_symbol (str): H-M 空间群符号
                - space_group_number (int): 空间群编号
                - hall_symbol (str): Hall 符号
                - crystal_system (str): 晶系（如 "Monoclinic"）
                - lattice_centring (str): 点阵型式（P/I/F/A/B/C/R）
                - unit_cell (tuple): (a, b, c, alpha, beta, gamma)，Å/度
                - unit_cell_volume (float): 晶胞体积，Å³
                - reciprocal_cell (tuple): 倒易晶胞参数，Å⁻¹/度
        """
        if self._crystal_symmetry is None:
            raise ValueError("尚未加载晶体，请先调用 load() 或 load_cif()/load_manual_cell()")

        sg = self._crystal_symmetry.space_group()

        chemical_formula = None
        n_atoms_cell = None
        if self._structure is not None:
            from collections import Counter
            counts = Counter()
            total = 0
            for sc in self._structure.scatterers():
                mult = int(round(sc.occupancy * sc.multiplicity()))
                counts[sc.element_symbol()] += mult
                total += mult
            n_atoms_cell = total

            has_carbon = "C" in counts

            def hill_sort_key(elem):
                if has_carbon:
                    if elem == "C":
                        return (0, "")
                    if elem == "H":
                        return (1, "")
                return (2, elem)

            parts = []
            for elem in sorted(counts, key=hill_sort_key):
                n = counts[elem]
                parts.append(elem if n == 1 else f"{elem}{n}")
            chemical_formula = " ".join(parts)

        return {
            "chemical_formula": chemical_formula,
            "n_atoms_cell": n_atoms_cell,
            "space_group_symbol": str(self._crystal_symmetry.space_group_info()),
            "space_group_number": sg.type().number(),
            "hall_symbol": sg.type().hall_symbol(),
            "crystal_system": sg.crystal_system(),
            "lattice_centring": sg.conventional_centring_type_symbol(),
            "unit_cell": self._unit_cell.parameters(),
            "unit_cell_volume": self._unit_cell.volume(),
            "reciprocal_cell": self._unit_cell.reciprocal().parameters(),
        }

    def get_structure(self):
        """返回 cctbx xray.structure 对象；手动输入时返回 None。"""
        return self._structure

    def get_crystal_symmetry(self):
        """返回 cctbx crystal.symmetry 对象。"""
        if self._crystal_symmetry is None:
            raise ValueError("尚未加载晶体，请先调用 load() 或 load_cif()/load_manual_cell()")
        return self._crystal_symmetry

    def get_unit_cell(self):
        """返回 cctbx unit_cell 对象。"""
        if self._unit_cell is None:
            raise ValueError("尚未加载晶体，请先调用 load() 或 load_cif()/load_manual_cell()")
        return self._unit_cell

    def get_space_group_symbol(self):
        """返回空间群符号字符串。"""
        if self._crystal_symmetry is None:
            raise ValueError("尚未加载晶体，请先调用 load() 或 load_cif()/load_manual_cell()")
        return str(self._crystal_symmetry.space_group_info())

    def get_point_group_symbol(self):
        """
        从 cctbx 空间群中提取点群符号（orix 兼容格式）。

        cctbx 的 `space_group.point_group_type()` 剥离所有平移分量
        （centering + screw/glide translations），返回纯点群 H-M 符号，
        格式与 orix 的 `Phase(point_group=...)` 兼容。

        示例:
            'P m -3 m'  -> 'm-3m'
            'P 4/m m m' -> '4/mmm'
            'P 1'       -> '1'
            'F d -3 m'  -> 'm-3m'

        返回:
            str: 点群 H-M 符号。

        异常:
            ValueError: 尚未加载晶体。
        """
        if self._crystal_symmetry is None:
            raise ValueError("尚未加载晶体，请先调用 load() 或 load_cif()/load_manual_cell()")
        sg = self._crystal_symmetry.space_group()
        return str(sg.point_group_type())

    def get_input_mode(self):
        """返回晶体输入模式：'cif' 或 'manual'。"""
        return self._input_mode

    def is_cif_loaded(self):
        """返回是否已加载 CIF 文件。"""
        return self._cif_loaded

    def is_intensity_available(self):
        """返回是否可计算真实结构因子强度。"""
        return self._intensity_available

    def get_b_matrix(self):
        """
        获取 PDB 正交化约定下的 B 矩阵。

        三个矩阵的换算关系：

        1. O 矩阵（实空间正交化矩阵）
           由 `unit_cell.orthogonalization_matrix()` 得到，将分数坐标 [u, v, w]
           转换为笛卡尔坐标（单位 Å）：
               x_cartesian = O · [u, v, w]^T

        2. B_no_2pi 矩阵（倒空间 B 矩阵，不含 2π）
           倒空间基矢是实空间基矢的逆转置：
               B_no_2pi = (O⁻¹)^T
           满足：
               |B_no_2pi · (h, k, l)^T| = 1 / d_hkl

        3. B 矩阵（倒空间 B 矩阵，含 2π）
           散射矢量 q 的定义通常包含 2π 因子：
               q = 2π · B_no_2pi · (h, k, l)^T
               = B · (h, k, l)^T
           因此：
               B = 2π · B_no_2pi
           满足：
               |B · (h, k, l)^T| = 2π / d_hkl

        返回:
            dict: 包含以下键的字典：
                - 'orthogonalization_matrix': 实空间正交化矩阵 O（3×3 ndarray）
                - 'b_matrix_no_2pi': 不含 2π 的倒空间 B 矩阵（3×3 ndarray）
                - 'b_matrix': 含 2π 的倒空间 B 矩阵（3×3 ndarray）
        """
        if self._unit_cell is None:
            raise ValueError("必须先调用 load_cif() 加载晶体结构")

        # 1. 获取实空间正交化矩阵 O
        orthogonalization_matrix = np.array(
            self._unit_cell.orthogonalization_matrix()
        ).reshape(3, 3)

        # 2. 计算倒空间 B_no_2pi = (O⁻¹)ᵀ
        b_matrix_no_2pi = np.linalg.inv(orthogonalization_matrix).T

        # 3. 含 2π 因子的 B 矩阵
        b_matrix = 2 * np.pi * b_matrix_no_2pi

        return {
            'orthogonalization_matrix': orthogonalization_matrix,
            'b_matrix_no_2pi': b_matrix_no_2pi,
            'b_matrix': b_matrix,
        }

    def get_unit_cell_parameters(self):
        """
        获取晶胞参数与晶胞体积。

        返回:
            dict: 包含以下键的字典：
                - 'a': a 轴长度（Å）
                - 'b': b 轴长度（Å）
                - 'c': c 轴长度（Å）
                - 'alpha': α 角（度）
                - 'beta': β 角（度）
                - 'gamma': γ 角（度）
                - 'volume': 晶胞体积（Å³）
        """
        if self._unit_cell is None:
            raise ValueError("必须先调用 load_cif() 加载晶体结构")

        a, b, c, alpha, beta, gamma = self._unit_cell.parameters()
        volume = self._unit_cell.volume()

        return {
            'a': a,
            'b': b,
            'c': c,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'volume': volume,
        }

    def check_systematic_absences(self, hkl_array=None):
        """
        检查给定 Miller 指数中哪些属于系统消光（systematic absences）。

        系统消光由空间群的对称元素（如体心/面心、滑移面、螺旋轴）决定。
        cctbx 的 `space_group.is_sys_absent(hkl)` 可直接判断某 hkl 是否被系统消光。

        参数:
            hkl_array (ndarray, optional): 待检查的 Miller 指数数组，形状 (N, 3)。
                若未提供，则使用当前已生成的 `_miller_set` 中的 Miller 指数。

        返回:
            dict: 包含以下键的字典：
                - 'space_group_symbol': 空间群符号字符串
                - 'hkl_array': 输入的 Miller 指数数组
                - 'absence_mask': 系统消光掩码，形状 (N,)，True 表示该 hkl 被消光
                - 'absent_count': 被消光的反射数目
                - 'absent_hkl_array': 被消光的 Miller 指数数组
                - 'allowed_count': 允许的反射数目
        """
        if self._crystal_symmetry is None:
            raise ValueError("必须先调用 load_cif() 加载晶体结构")

        if hkl_array is None:
            if self._miller_set is None:
                raise ValueError("未提供 hkl_array 且未生成 Miller 集合")
            hkl_array = np.array(
                [(h, k, l) for h, k, l in self._miller_set.indices()], dtype=int
            )
        else:
            hkl_array = np.asarray(hkl_array, dtype=int)
            if hkl_array.ndim != 2 or hkl_array.shape[1] != 3:
                raise ValueError("hkl_array 形状必须为 (N, 3)")

        space_group = self._crystal_symmetry.space_group()
        space_group_symbol = str(self._crystal_symmetry.space_group_info())

        absence_mask = np.array(
            [space_group.is_sys_absent(tuple(int(x) for x in hkl)) for hkl in hkl_array],
            dtype=bool,
        )

        absent_count = int(np.sum(absence_mask))
        allowed_count = len(hkl_array) - absent_count
        absent_hkl_array = hkl_array[absence_mask] if absent_count > 0 else np.array([], dtype=int).reshape(0, 3)

        return {
            'space_group_symbol': space_group_symbol,
            'hkl_array': hkl_array,
            'absence_mask': absence_mask,
            'absent_count': absent_count,
            'absent_hkl_array': absent_hkl_array,
            'allowed_count': allowed_count,
        }
