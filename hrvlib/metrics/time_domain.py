import numpy as np
import warnings
from typing import List, Tuple, Union, Optional
from scipy.interpolate import CubicSpline


class HRVTimeDomainAnalysis:
    """
    高精度HRV时域分析工具
    实现方法符合临床和科研标准(HRV Guidelines: Task Force of ESC/ASPE)
    """

    def __init__(self, rr_intervals: List[Union[int, float]],
                 sampling_frequency: Optional[float] = None,
                 units: str = 'ms',
                 outlier_threshold: float = 3.0,
                 ectopic_correction: bool = True):
        """
        初始化HRV分析器

        参数:
        rr_intervals: RR间期列表(单位毫秒或秒)
        sampling_frequency: 信号采样频率(Hz)
        units: 输入单位('ms' 或 's')
        outlier_threshold: 离群值检测的Z-score阈值
        ectopic_correction: 是否进行异位搏动校正
        """
        self.raw_rr = np.array(rr_intervals, dtype=np.float64)

        if units not in ['ms', 's']:
            raise ValueError("单位必须是'ms'或's'")

        # 转换为毫秒
        self.rr_ms = self.raw_rr * 1000 if units == 's' else self.raw_rr

        if len(self.rr_ms) < 2:
            raise ValueError("至少需要2个RR间期进行计算")

        if np.any(self.rr_ms <= 0):
            raise ValueError("RR间期必须为正值")

        self.sampling_frequency = sampling_frequency
        self.outlier_threshold = outlier_threshold
        self.ectopic_correction = ectopic_correction

        # 数据预处理
        self.clean_rr = self._preprocess_data()

    def _preprocess_data(self) -> np.ndarray:
        """
        数据预处理流程:
        1. 离群值检测和处理
        2. 异位搏动校正(可选)
        3. 重采样(如果提供采样频率)
        """
        # 离群值处理
        cleaned = self._handle_outliers()

        # 异位搏动校正
        if self.ectopic_correction:
            cleaned = self._correct_ectopic_beats(cleaned)

        # 重采样
        if self.sampling_frequency:
            cleaned = self._resample_rr(cleaned)

        return cleaned

    def _handle_outliers(self) -> np.ndarray:
        """使用移动中位数和Z-score检测离群值"""
        rr = self.rr_ms.copy()
        median = np.median(rr)
        abs_dev = np.abs(rr - median)
        mad = np.median(abs_dev)

        # 如果所有值相同，MAD=0，无需处理
        if mad < 1e-6:
            return rr

        # 计算Z-score
        z_scores = 0.6745 * (rr - median) / mad

        # 替换离群值
        outlier_idx = np.where(np.abs(z_scores) > self.outlier_threshold)[0]

        # 使用整个序列的中位数替换离群值
        for idx in outlier_idx:
            # 找到最近的非离群值
            left_val = None
            right_val = None

            # 向左查找非离群值
            left = idx - 1
            while left >= 0:
                if left not in outlier_idx:
                    left_val = rr[left]
                    break
                left -= 1

            # 向右查找非离群值
            right = idx + 1
            while right < len(rr):
                if right not in outlier_idx:
                    right_val = rr[right]
                    break
                right += 1

            # 使用找到的值替换离群值
            if left_val is not None and right_val is not None:
                rr[idx] = (left_val + right_val) / 2
            elif left_val is not None:
                rr[idx] = left_val
            elif right_val is not None:
                rr[idx] = right_val
            else:
                # 如果找不到非离群值，使用整个序列的中位数
                rr[idx] = median

        return rr

    def _correct_ectopic_beats(self, rr: np.ndarray) -> np.ndarray:
        """使用自适应阈值校正异位搏动"""
        if len(rr) < 3:
            return rr  # 需要至少3个点进行校正

        diff = np.abs(np.diff(rr))
        median_diff = np.median(diff)

        # 识别异常差异
        threshold = 4.0 * median_diff
        ectopic_idx = np.where(diff > threshold)[0]

        # 校正异位搏动
        for idx in ectopic_idx:
            if idx > 0 and idx < len(rr) - 1:
                # 替换为前后两个点的平均值
                rr[idx] = (rr[idx - 1] + rr[idx + 1]) / 2

        return rr

    def _resample_rr(self, rr: np.ndarray) -> np.ndarray:
        """使用样条插值进行重采样"""
        # 计算时间点
        time_points = np.cumsum(rr) / 1000.0  # 转换为秒
        total_time = time_points[-1]

        # 创建插值函数
        cs = CubicSpline(time_points, rr)

        # 创建均匀时间网格
        fs = self.sampling_frequency
        num_samples = int(np.ceil(total_time * fs)) + 1
        new_time = np.linspace(0, total_time, num_samples)

        # 插值获取新RR序列
        resampled_rr = cs(new_time)
        return resampled_rr

    def sdnn(self) -> float:
        """
        计算SDNN: RR间期的标准差
        反映总体HRV
        """
        return float(np.std(self.clean_rr, ddof=1))

    def rmssd(self) -> float:
        """
        计算RMSSD: 连续RR间期差值的均方根
        反映短期HRV
        """
        if len(self.clean_rr) < 2:
            return 0.0

        diff = np.diff(self.clean_rr)
        squared_diff = diff ** 2
        mean_squared_diff = np.mean(squared_diff)
        return float(np.sqrt(mean_squared_diff))

    def pnn50(self) -> float:
        """
        计算pNN50: 相邻RR间期差值超过50ms的百分比
        反映迷走神经活性
        """
        if len(self.clean_rr) < 2:
            return 0.0

        diff = np.abs(np.diff(self.clean_rr))
        nn50 = np.sum(diff > 50)
        total = len(diff)
        return (nn50 / total) * 100 if total > 0 else 0.0

    def nn50(self) -> int:
        """相邻RR间期差值超过50ms的数量"""
        if len(self.clean_rr) < 2:
            return 0

        diff = np.abs(np.diff(self.clean_rr))
        return int(np.sum(diff > 50))

    def mean_rr(self) -> float:
        """平均RR间期"""
        return float(np.mean(self.clean_rr))

    def median_rr(self) -> float:
        """RR间期中位数"""
        return float(np.median(self.clean_rr))

    def hr_triangular_index(self) -> float:
        """RR间期直方图的高度"""
        if len(self.clean_rr) == 0:
            return 0.0

        hist, bin_edges = np.histogram(self.clean_rr, bins='auto')
        max_index = np.argmax(hist)
        return hist[max_index] / len(self.clean_rr)

    def tinn(self) -> float:
        """
        三角插值NN间期直方图(TINN)
        测量直方图分布的宽度
        """
        if len(self.clean_rr) < 3:
            return 0.0

        hist, bin_edges = np.histogram(self.clean_rr, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 找到直方图峰值
        max_idx = np.argmax(hist)

        # 找到左右边界
        left_idx = max_idx
        while left_idx > 0 and hist[left_idx] > hist[max_idx] / 2:
            left_idx -= 1

        right_idx = max_idx
        while right_idx < len(hist) - 1 and hist[right_idx] > hist[max_idx] / 2:
            right_idx += 1

        # 计算TINN
        return bin_edges[right_idx] - bin_edges[left_idx]

    def full_analysis(self) -> dict:
        """执行完整的时域分析"""
        return {
            'sdnn': self.sdnn(),
            'rmssd': self.rmssd(),
            'pnn50': self.pnn50(),
            'nn50': self.nn50(),
            'mean_rr': self.mean_rr(),
            'median_rr': self.median_rr(),
            'hr_triangular_index': self.hr_triangular_index(),
            'tinn': self.tinn()
        }


def validate_rr_intervals(rr_intervals: List[Union[int, float]]) -> None:
    """验证RR间期输入"""
    if not isinstance(rr_intervals, list):
        raise TypeError("RR间期必须为列表")

    if len(rr_intervals) < 2:
        raise ValueError("至少需要2个RR间期")

    if any(rr <= 0 for rr in rr_intervals):
        raise ValueError("RR间期必须为正值")

    if not all(isinstance(rr, (int, float)) for rr in rr_intervals):
        raise TypeError("RR间期必须为数值类型")