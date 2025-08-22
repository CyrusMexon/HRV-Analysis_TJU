import numpy as np
import scipy.stats
import scipy.spatial
import scipy.special
from typing import Tuple, Optional, List, Dict, Union


class NonlinearHRVAnalysis:
    """
    非线性HRV分析工具集
    实现Poincaré分析、样本熵、多尺度熵、DFA和RQA
    符合科研和企业级精度要求
    """

    @staticmethod
    def preprocess_rr(rr_intervals: np.ndarray,
                      outlier_threshold: float = 3.0,
                      interpolation_method: str = 'cubic',
                      max_correction_percentage: float = 5.0) -> Tuple[np.ndarray, float]:
        """
        RR间期数据预处理：异常值检测和校正

        参数:
        rr_intervals: RR间期序列(毫秒)
        outlier_threshold: 异常值检测阈值(标准差倍数)
        interpolation_method: 插值方法 ('linear'或'cubic')
        max_correction_percentage: 最大允许校正百分比

        返回:
        cleaned_rr: 清理后的RR间期序列
        corrected_percentage: 校正的心搏百分比
        """
        # 输入验证
        if not isinstance(rr_intervals, np.ndarray):
            raise TypeError("输入必须是numpy数组")
        if rr_intervals.ndim != 1:
            raise ValueError("输入必须是一维数组")
        if len(rr_intervals) < 10:
            raise ValueError("RR间期序列至少需要10个点")
        if interpolation_method not in ['linear', 'cubic']:
            raise ValueError("插值方法必须是'linear'或'cubic'")

        n = len(rr_intervals)
        original_rr = rr_intervals.copy()
        corrected_flags = np.zeros(n, dtype=bool)  # 标记校正点
        corrected_count = 0

        # 初始异常值检测
        median = np.median(rr_intervals)
        mad = scipy.stats.median_abs_deviation(rr_intervals, scale='normal')

        # 计算动态阈值
        lower_bound = median - outlier_threshold * mad
        upper_bound = median + outlier_threshold * mad

        # 创建处理后的数组
        processed = rr_intervals.copy()

        # 迭代校正异常值
        for i in range(len(processed)):
            if processed[i] < lower_bound or processed[i] > upper_bound:
                corrected_flags[i] = True
                corrected_count += 1

                # 基于相邻点动态插值
                prev_val = processed[i - 1] if i > 0 else median
                next_val = processed[i + 1] if i < len(processed) - 1 else median

                if interpolation_method == 'linear':
                    # 线性插值
                    processed[i] = (prev_val + next_val) / 2
                else:  # 三次样条插值
                    # 使用加权平均作为简化的三次样条近似
                    processed[i] = (prev_val * 0.4 + next_val * 0.6)

        # 二次检查确保所有值在合理范围内
        q1, q3 = np.percentile(processed, [25, 75])
        iqr = q3 - q1
        final_lower = q1 - 1.5 * iqr
        final_upper = q3 + 1.5 * iqr

        # 替换极端值
        extreme_mask = (processed < final_lower) | (processed > final_upper)
        if np.any(extreme_mask):
            median_val = np.median(processed)
            processed[extreme_mask] = median_val
            # 更新校正计数
            new_corrections = np.sum(~corrected_flags & extreme_mask)
            corrected_count += new_corrections
            corrected_flags |= extreme_mask

        # 计算校正百分比
        corrected_percentage = (corrected_count / n) * 100

        # 如果校正百分比超过阈值，发出警告
        if corrected_percentage > max_correction_percentage:
            print(f"警告：校正心搏百分比({corrected_percentage:.2f}%)超过阈值({max_correction_percentage}%)")

        return processed, corrected_percentage

    @staticmethod
    def poincare_analysis(rr_intervals: np.ndarray,
                          preprocess: bool = True,
                          interpolation_method: str = 'cubic',
                          outlier_threshold: float = 3.0) -> Tuple[float, float, np.ndarray, float]:
        """
        执行Poincaré分析并计算SD1和SD2指标

        参数:
        rr_intervals: RR间期序列(毫秒)
        preprocess: 是否进行预处理
        interpolation_method: 插值方法 ('linear'或'cubic')
        outlier_threshold: 异常值检测阈值(标准差倍数)

        返回:
        sd1: 垂直于恒等线的标准差
        sd2: 沿着恒等线的标准差
        cleaned_rr: 清理后的RR间期序列
        corrected_percentage: 校正的心搏百分比
        """
        corrected_percentage = 0.0

        # 数据预处理
        if preprocess:
            rr_clean, corrected_percentage = NonlinearHRVAnalysis.preprocess_rr(
                rr_intervals, outlier_threshold, interpolation_method
            )
        else:
            rr_clean = rr_intervals.copy()

        # 创建Poincaré图数据点
        x = rr_clean[:-1]
        y = rr_clean[1:]

        # 计算差值
        diff = y - x

        # 计算SD1 (正确公式)
        sd1 = np.sqrt(np.var(diff, ddof=1) / 2)

        # 计算SD2 (正确公式)
        # SD2 = sqrt(2 * SDNN² - 0.5 * SD1²)
        sdnn = np.std(rr_clean, ddof=1)
        sd2 = np.sqrt(2 * sdnn ** 2 - 0.5 * sd1 ** 2)

        # 添加数值稳定性检查
        sd1 = max(sd1, 1e-10)
        sd2 = max(sd2, 1e-10)

        return sd1, sd2, rr_clean, corrected_percentage

    @staticmethod
    def sample_entropy(rr_intervals: np.ndarray,
                       m: int = 2,
                       r: float = 0.2,
                       normalize: bool = True,
                       preprocess: bool = True,
                       interpolation_method: str = 'cubic') -> float:
        """
        计算RR间期序列的样本熵

        参数:
        rr_intervals: RR间期序列(毫秒)
        m: 模板长度(通常为2)
        r: 容差系数(通常为0.2)
        normalize: 是否标准化数据
        preprocess: 是否进行预处理
        interpolation_method: 插值方法 ('linear'或'cubic')

        返回:
        sampen: 样本熵值
        """
        # 数据预处理
        if preprocess:
            rr_clean, _ = NonlinearHRVAnalysis.preprocess_rr(
                rr_intervals, interpolation_method=interpolation_method
            )
        else:
            rr_clean = rr_intervals.copy()

        # 输入验证
        if len(rr_clean) < 100:
            raise ValueError("样本熵计算至少需要100个数据点")
        if m < 1:
            raise ValueError("模板长度m必须至少为1")
        if r <= 0:
            raise ValueError("容差r必须为正数")

        # 标准化数据
        if normalize:
            std = np.std(rr_clean, ddof=1)
            if std < 1e-10:  # 防止除零错误
                return 0.0
            rr_norm = (rr_clean - np.mean(rr_clean)) / std
        else:
            rr_norm = rr_clean.copy()

        n = len(rr_norm)

        # 样本熵的标准计算方法
        def _phi(m):
            # 创建模板向量
            x = np.array([rr_norm[i:i + m] for i in range(n - m)])
            # 计算模板间的距离
            dist = scipy.spatial.distance.cdist(x, x, metric='chebyshev')
            # 计算匹配数（排除自匹配）
            return np.sum(dist <= r) - n + m  # 减去自匹配

        # 计算样本熵
        b = _phi(m)
        a = _phi(m + 1)

        # 避免除零错误
        if b == 0:
            return 0.0
        if a == 0:
            return float('inf')

        return -np.log(a / b)

    @staticmethod
    def multiscale_entropy(rr_intervals: np.ndarray,
                           scale_max: int = 10,
                           m: int = 2,
                           r: float = 0.15,
                           preprocess: bool = True,
                           interpolation_method: str = 'cubic') -> np.ndarray:
        """
        计算多尺度样本熵

        参数:
        rr_intervals: RR间期序列(毫秒)
        scale_max: 最大尺度
        m: 模板长度
        r: 容差系数
        preprocess: 是否进行预处理
        interpolation_method: 插值方法 ('linear'或'cubic')

        返回:
        mse: 各尺度的样本熵值
        """
        # 数据预处理
        if preprocess:
            rr_clean, _ = NonlinearHRVAnalysis.preprocess_rr(
                rr_intervals, interpolation_method=interpolation_method
            )
        else:
            rr_clean = rr_intervals.copy()

        n = len(rr_clean)
        if n < scale_max * 100:
            raise ValueError(f"多尺度熵计算需要至少{scale_max * 100}个数据点")

        mse = np.zeros(scale_max)

        for scale in range(1, scale_max + 1):
            # 创建粗粒度序列
            coarse_grained = []
            for i in range(0, n - scale + 1, scale):
                coarse_grained.append(np.mean(rr_clean[i:i + scale]))

            # 计算样本熵
            try:
                sampen = NonlinearHRVAnalysis.sample_entropy(
                    np.array(coarse_grained), m, r, normalize=True, preprocess=False
                )
                mse[scale - 1] = sampen
            except Exception as e:
                print(f"尺度 {scale} 计算失败: {str(e)}")
                mse[scale - 1] = np.nan

        return mse

    @staticmethod
    def detrended_fluctuation_analysis(rr_intervals: np.ndarray,
                                       preprocess: bool = True,
                                       interpolation_method: str = 'cubic') -> Tuple[float, float, np.ndarray]:
        """
        计算去趋势波动分析(DFA)指标

        参数:
        rr_intervals: RR间期序列(毫秒)
        preprocess: 是否进行预处理
        interpolation_method: 插值方法 ('linear'或'cubic')

        返回:
        alpha1: 短期标度指数(4-11点)
        alpha2: 长期标度指数(>11点)
        fluctuations: 各尺度的波动值
        """
        # 数据预处理
        if preprocess:
            rr_clean, _ = NonlinearHRVAnalysis.preprocess_rr(
                rr_intervals, interpolation_method=interpolation_method
            )
        else:
            rr_clean = rr_intervals.copy()

        n = len(rr_clean)
        if n < 100:
            raise ValueError("DFA计算至少需要100个数据点")

        # 累积和
        y = np.cumsum(rr_clean - np.mean(rr_clean))

        # 创建盒子大小 (使用对数分布)
        min_size = 4
        max_size = min(n // 4, 100)  # 最大盒子大小为100
        box_sizes = np.unique(np.logspace(np.log10(min_size), np.log10(max_size), num=10, dtype=int))

        fluctuations = []

        for size in box_sizes:
            # 计算分段数量
            n_segments = n // size
            f2 = 0

            for i in range(n_segments):
                # 获取当前段
                segment = y[i * size:(i + 1) * size]

                # 线性拟合去趋势
                x = np.arange(size)
                slope, intercept = np.polyfit(x, segment, 1)
                trend = slope * x + intercept

                # 计算波动
                detrended = segment - trend
                f2 += np.sum(detrended ** 2)

            # 平均波动
            f2 = np.sqrt(f2 / (n_segments * size))
            fluctuations.append(f2)

        # 对数变换
        log_box = np.log(box_sizes)
        log_fluct = np.log(fluctuations)

        # 计算短期和长期标度指数
        # 短期: 4-11点
        short_term_mask = (box_sizes >= 4) & (box_sizes <= 11)
        # 长期: 12-100点
        long_term_mask = (box_sizes > 11) & (box_sizes <= 100)

        # 计算标度指数
        if np.sum(short_term_mask) > 1:
            slope1, _ = np.polyfit(log_box[short_term_mask], log_fluct[short_term_mask], 1)
        else:
            slope1 = np.nan

        if np.sum(long_term_mask) > 1:
            slope2, _ = np.polyfit(log_box[long_term_mask], log_fluct[long_term_mask], 1)
        else:
            slope2 = np.nan

        return slope1, slope2, np.array(fluctuations)

    @staticmethod
    def recurrence_quantification_analysis(rr_intervals: np.ndarray,
                                           threshold: float = 0.1,
                                           embedding_dim: int = 3,
                                           delay: int = 1,
                                           preprocess: bool = True,
                                           interpolation_method: str = 'cubic') -> dict:
        """
        递归量化分析(RQA)

        参数:
        rr_intervals: RR间期序列(毫秒)
        threshold: 递归阈值
        embedding_dim: 嵌入维度
        delay: 延迟时间
        preprocess: 是否进行预处理
        interpolation_method: 插值方法 ('linear'或'cubic')

        返回:
        rqa_metrics: RQA指标字典
        """
        # 数据预处理
        if preprocess:
            rr_clean, _ = NonlinearHRVAnalysis.preprocess_rr(
                rr_intervals, interpolation_method=interpolation_method
            )
        else:
            rr_clean = rr_intervals.copy()

        n = len(rr_clean)
        if n < 100:
            raise ValueError("RQA计算至少需要100个数据点")

        # 相空间重构
        embedded = []
        for i in range(n - (embedding_dim - 1) * delay):
            embedded.append(rr_clean[i:i + embedding_dim * delay:delay])
        embedded = np.array(embedded)

        # 创建递归图
        dist_matrix = scipy.spatial.distance_matrix(embedded, embedded)
        recurrence_matrix = dist_matrix < threshold

        # 计算RQA指标
        n_points = recurrence_matrix.shape[0]

        # 递归率
        recurrence_rate = np.sum(recurrence_matrix) / (n_points ** 2)

        # 确定性 (对角线结构)
        diagonal_lines = []
        for i in range(n_points):
            for j in range(n_points):
                if recurrence_matrix[i, j]:
                    k = 0
                    while (i + k < n_points and j + k < n_points and
                           recurrence_matrix[i + k, j + k]):
                        k += 1
                    if k > 1:  # 最小线长为2
                        diagonal_lines.append(k)
                        # 清除已计数的线
                        for d in range(k):
                            recurrence_matrix[i + d, j + d] = False

        if diagonal_lines:
            determinism = np.sum(diagonal_lines) / np.sum(recurrence_matrix)
            avg_diag = np.mean(diagonal_lines)
            max_diag = np.max(diagonal_lines)
        else:
            determinism = 0
            avg_diag = 0
            max_diag = 0

        # 层流性 (垂直线结构)
        vertical_lines = []
        for j in range(n_points):
            i = 0
            while i < n_points:
                if recurrence_matrix[i, j]:
                    k = 0
                    while i + k < n_points and recurrence_matrix[i + k, j]:
                        k += 1
                    if k > 1:  # 最小线长为2
                        vertical_lines.append(k)
                        i += k  # 跳过已计数的点
                    else:
                        i += 1
                else:
                    i += 1

        if vertical_lines:
            laminarity = np.sum(vertical_lines) / np.sum(recurrence_matrix)
            avg_vert = np.mean(vertical_lines)
            max_vert = np.max(vertical_lines)
        else:
            laminarity = 0
            avg_vert = 0
            max_vert = 0

        return {
            'recurrence_rate': recurrence_rate,
            'determinism': determinism,
            'avg_diagonal': avg_diag,
            'max_diagonal': max_diag,
            'laminarity': laminarity,
            'avg_vertical': avg_vert,
            'max_vertical': max_vert
        }