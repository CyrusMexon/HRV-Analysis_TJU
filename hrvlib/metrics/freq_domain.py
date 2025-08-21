import numpy as np
from scipy import signal, interpolate
from typing import Tuple, Dict, Optional
import warnings


class HRVFreqDomainAnalysis:
    """
    高精度HRV频域分析工具，采用Welch方法计算功率谱密度
    实现功能：
    1. 自适应信号预处理
    2. 多参数Welch谱估计
    3. 频带功率精确积分
    4. 多频段分析（ULF, VLF, LF, HF）
    5. 异常处理和参数验证

    参数：
    rr_intervals : RR间期序列（单位：秒）
    sampling_rate : 重采样频率（Hz，默认4Hz）
    detrend_method : 去趋势方法（'linear', 'constant', None）
    window_type : 窗函数类型（默认'hann'）
    segment_length : 分段长度（秒）
    overlap_ratio : 分段重叠比例（0-1）
    """

    VALID_WINDOWS = ['hann', 'hamming', 'blackman', 'bartlett', 'flattop', 'parzen', 'bohman', 'nuttall']
    VALID_DETRENDS = ['linear', 'constant', None]
    DEFAULT_FREQ_BANDS = {
        'ulf': (0.0, 0.003),
        'vlf': (0.003, 0.04),
        'lf': (0.04, 0.15),
        'hf': (0.15, 0.4),
        'lf_hf_ratio': (0.04, 0.4)
    }

    def __init__(self,
                 rr_intervals: np.ndarray,
                 sampling_rate: float = 4.0,
                 detrend_method: Optional[str] = 'linear',
                 window_type: str = 'hann',
                 segment_length: float = 120.0,
                 overlap_ratio: float = 0.75):

        self._validate_input(rr_intervals, sampling_rate, detrend_method, window_type, segment_length, overlap_ratio)

        self.rr_intervals = rr_intervals
        self.sampling_rate = sampling_rate
        self.detrend_method = detrend_method
        self.window_type = window_type
        self.segment_length = segment_length
        self.overlap_ratio = overlap_ratio
        self.time_domain = self._create_time_domain_signal()
        self.freqs, self.psd = self._compute_welch_psd()
        self.spectral_metrics = self._compute_spectral_metrics()

    def _validate_input(self,
                        rr_intervals: np.ndarray,
                        sampling_rate: float,
                        detrend_method: Optional[str],
                        window_type: str,
                        segment_length: float,
                        overlap_ratio: float) -> None:
        """严格验证所有输入参数"""
        if not isinstance(rr_intervals, np.ndarray):
            raise TypeError("RR间期必须为NumPy数组")
        if rr_intervals.ndim != 1:
            raise ValueError("RR间期必须是一维数组")
        if len(rr_intervals) == 0:
            raise ValueError("RR间期数组不能为空")
        if np.any(rr_intervals <= 0):
            raise ValueError("RR间期必须为正值")
        if sampling_rate <= 0:
            raise ValueError("采样率必须为正数")
        if detrend_method not in self.VALID_DETRENDS:
            raise ValueError(f"去趋势方法必须是: {self.VALID_DETRENDS}")
        if window_type not in self.VALID_WINDOWS:
            raise ValueError(f"无效窗函数: {window_type}. 有效选项: {self.VALID_WINDOWS}")
        if segment_length <= 0:
            raise ValueError("分段长度必须为正数")
        if not (0 <= overlap_ratio < 1):
            raise ValueError("重叠比例必须在[0,1)范围内")

    def _create_time_domain_signal(self) -> np.ndarray:
        """创建等间隔时间序列信号（带异常值处理）"""
        # 空数组检查
        if len(self.rr_intervals) == 0:
            return np.array([])

        # 计算累积时间
        time_points = np.cumsum(self.rr_intervals)
        time_points -= time_points[0]  # 从零开始

        # 使用三次样条插值进行重采样
        interp_func = interpolate.CubicSpline(time_points, self.rr_intervals)

        # 创建等间隔时间轴
        duration = time_points[-1]
        num_samples = int(duration * self.sampling_rate)
        new_time_axis = np.linspace(0, duration, num_samples)

        # 应用插值
        return interp_func(new_time_axis)

    def _compute_welch_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        """使用Welch方法计算功率谱密度（带自适应参数）"""
        # 空数组检查
        if len(self.time_domain) == 0:
            return np.array([]), np.array([])

        # 计算分段参数
        nperseg = int(self.segment_length * self.sampling_rate)

        # 确保nperseg不超过信号长度的一半
        max_nperseg = len(self.time_domain) // 2
        if nperseg > max_nperseg:
            warnings.warn(f"请求的分段长度({nperseg})超过信号长度({len(self.time_domain)})的一半，已调整为{max_nperseg}")
            nperseg = max_nperseg

        # 确保nperseg至少为4
        if nperseg < 4:
            warnings.warn(f"计算的分段长度({nperseg})太小，无法进行Welch计算")
            return np.array([]), np.array([])

        noverlap = int(nperseg * self.overlap_ratio)

        # 获取窗函数
        window = self._get_window(nperseg)

        # 计算PSD
        freqs, psd = signal.welch(
            x=self.time_domain,
            fs=self.sampling_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=self.detrend_method,
            scaling='density',
            average='mean'
        )

        return freqs, psd

    def _get_window(self, nperseg: int) -> np.ndarray:
        """生成指定类型的窗函数"""
        if self.window_type == 'hann':
            return signal.windows.hann(nperseg)
        elif self.window_type == 'hamming':
            return signal.windows.hamming(nperseg)
        elif self.window_type == 'blackman':
            return signal.windows.blackman(nperseg)
        elif self.window_type == 'bartlett':
            return signal.windows.bartlett(nperseg)
        elif self.window_type == 'flattop':
            return signal.windows.flattop(nperseg)
        elif self.window_type == 'parzen':
            return signal.windows.parzen(nperseg)
        elif self.window_type == 'bohman':
            return signal.windows.bohman(nperseg)
        elif self.window_type == 'nuttall':
            return signal.windows.nuttall(nperseg)
        else:
            raise ValueError(f"未知窗函数类型: {self.window_type}")

    def _compute_spectral_metrics(self) -> Dict[str, float]:
        """计算所有频域指标（带自适应积分）"""
        # 空PSD检查
        if len(self.psd) == 0 or len(self.freqs) == 0:
            warnings.warn("PSD计算结果为空，返回默认值")
            return {
                'ulf_power': 0.0, 'ulf_power_nu': 0.0,
                'vlf_power': 0.0, 'vlf_power_nu': 0.0,
                'lf_power': 0.0, 'lf_power_nu': 0.0,
                'hf_power': 0.0, 'hf_power_nu': 0.0,
                'lf_hf_ratio': float('nan'),
                'total_power': 0.0,
                'peak_freq': float('nan')
            }

        results = {}

        # 计算总功率 - 使用np.trapz替代
        total_power = np.trapz(self.psd, self.freqs)

        # 计算各频段功率 - 同样使用np.trapz
        for band, (low, high) in self.DEFAULT_FREQ_BANDS.items():
            if band == 'lf_hf_ratio':
                continue

            mask = (self.freqs >= low) & (self.freqs <= high)
            if not np.any(mask):
                warnings.warn(f"频段[{low}, {high}] Hz内无数据点，功率设为0")
                band_power = 0.0
            else:
                band_power = np.trapz(self.psd[mask], self.freqs[mask])

            # 计算归一化功率（占总功率百分比）
            norm_power = (band_power / total_power) * 100 if total_power > 0 else 0.0

            results[f'{band}_power'] = band_power
            results[f'{band}_power_nu'] = norm_power

        # 计算LF/HF比率
        lf_power = results.get('lf_power', 0.0)
        hf_power = results.get('hf_power', 0.0)

        if hf_power > 0:
            results['lf_hf_ratio'] = lf_power / hf_power
        else:
            results['lf_hf_ratio'] = float('inf') if lf_power > 0 else float('nan')

        # 添加总功率和峰值频率
        results['total_power'] = total_power
        results['peak_freq'] = self._find_peak_frequency()

        return results

    def _find_peak_frequency(self) -> float:
        """在HF频段内寻找主峰频率"""
        # 空数组检查
        if len(self.freqs) == 0 or len(self.psd) == 0:
            return float('nan')

        hf_mask = (self.freqs >= self.DEFAULT_FREQ_BANDS['hf'][0]) & \
                  (self.freqs <= self.DEFAULT_FREQ_BANDS['hf'][1])

        if not np.any(hf_mask):
            return float('nan')

        hf_psd = self.psd[hf_mask]
        hf_freqs = self.freqs[hf_mask]

        # 寻找最高峰
        peak_idx = np.argmax(hf_psd)
        return hf_freqs[peak_idx]

    def get_results(self) -> Dict[str, float]:
        """返回所有频域指标"""
        return self.spectral_metrics

    def get_psd(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回频率和PSD数组"""
        return self.freqs, self.psd