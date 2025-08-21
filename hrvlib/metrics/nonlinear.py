import numpy as np
import scipy.stats
import scipy.spatial
import scipy.special
from typing import Tuple, Optional, List, Dict


class NonlinearHRVAnalysis:
    """
    非线性HRV分析工具集
    实现Poincaré分析和样本熵计算
    符合科研和企业级精度要求
    """

    @staticmethod
    def poincare_analysis(rr_intervals: np.ndarray,
                          correction_factor: float = 0.02,
                          outlier_threshold: float = 3.0,
                          robust_statistics: bool = True) -> Tuple[float, float, np.ndarray]:
        """
        执行Poincaré分析并计算SD1和SD2指标

        参数:
        rr_intervals: RR间期序列(毫秒)
        correction_factor: 异常值校正因子(标准差倍数)
        outlier_threshold: 异常值检测阈值(标准差倍数)
        robust_statistics: 是否使用稳健统计方法(MAD)

        返回:
        sd1: 垂直于恒等线的标准差
        sd2: 沿着恒等线的标准差
        cleaned_rr: 清理后的RR间期序列
        """
        # 输入验证
        if not isinstance(rr_intervals, np.ndarray):
            raise TypeError("输入必须是numpy数组")
        if rr_intervals.ndim != 1:
            raise ValueError("输入必须是一维数组")
        if len(rr_intervals) < 10:
            raise ValueError("RR间期序列至少需要10个点")

        # 数据预处理
        rr_clean = NonlinearHRVAnalysis._preprocess_rr(rr_intervals,
                                                       correction_factor,
                                                       outlier_threshold)

        # 创建Poincaré图数据点
        x = rr_clean[:-1]
        y = rr_clean[1:]

        # 计算差值
        diff = y - x

        # 使用稳健统计方法计算标准差
        if robust_statistics:
            mad = scipy.stats.median_abs_deviation(diff, scale='normal')
            std_diff = mad if mad > 0 else np.std(diff, ddof=1)
        else:
            std_diff = np.std(diff, ddof=1)

        # 计算SD1和SD2
        sd1 = np.sqrt(0.5 * std_diff ** 2)

        # 使用协方差矩阵计算SD2
        cov_matrix = np.cov(x, y, ddof=1)  # 修复：使用ddof=1进行无偏估计
        var_sum = cov_matrix[0, 0] + cov_matrix[1, 1]
        cov_xy = cov_matrix[0, 1]
        sd2 = np.sqrt(var_sum - 2 * cov_xy - 0.5 * std_diff ** 2)

        # 添加数值稳定性检查
        sd1 = max(sd1, 1e-10)
        sd2 = max(sd2, 1e-10)

        return sd1, sd2, rr_clean

    @staticmethod
    def _preprocess_rr(rr_intervals: np.ndarray,
                       correction_factor: float,
                       outlier_threshold: float) -> np.ndarray:
        """RR间期数据预处理：异常值检测和校正"""
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
                # 基于相邻点动态插值
                prev_val = processed[i - 1] if i > 0 else median
                next_val = processed[i + 1] if i < len(processed) - 1 else median

                if i > 0 and i < len(processed) - 1:
                    # 使用加权平均
                    processed[i] = (prev_val * 0.4 + next_val * 0.6)
                elif i == 0:
                    processed[i] = next_val
                else:
                    processed[i] = prev_val

        # 二次检查确保所有值在合理范围内
        q1, q3 = np.percentile(processed, [25, 75])
        iqr = q3 - q1
        final_lower = q1 - 1.5 * iqr
        final_upper = q3 + 1.5 * iqr

        # 替换极端值
        processed[(processed < final_lower) | (processed > final_upper)] = np.median(processed)

        return processed

    @staticmethod
    def sample_entropy(rr_intervals: np.ndarray,
                       m: int = 2,
                       r: float = 0.2,
                       tolerance: float = 0.0001,
                       max_iter: int = 100,
                       normalize: bool = True) -> float:
        """
        计算RR间期序列的样本熵

        参数:
        rr_intervals: RR间期序列(毫秒)
        m: 模板长度(通常为2)
        r: 容差系数(通常为0.2)
        tolerance: 收敛容差
        max_iter: 最大迭代次数
        normalize: 是否标准化数据

        返回:
        sampen: 样本熵值
        """
        # 输入验证
        if len(rr_intervals) < 100:
            raise ValueError("样本熵计算至少需要100个数据点")
        if m < 1:
            raise ValueError("模板长度m必须至少为1")
        if r <= 0:
            raise ValueError("容差r必须为正数")

        # 标准化数据
        if normalize:
            std = np.std(rr_intervals, ddof=1)
            if std < 1e-10:  # 防止除零错误
                return 0.0
            rr_norm = (rr_intervals - np.mean(rr_intervals)) / std
        else:
            rr_norm = rr_intervals.copy()

        n = len(rr_norm)

        # 定义距离函数
        def _maxdist(x, y):
            return np.max(np.abs(x - y))

        # 创建模板向量
        def _get_templates(dim: int):
            templates = []
            for i in range(n - dim + 1):
                templates.append(rr_norm[i:i + dim])
            return np.array(templates)

        # 计算匹配数
        def _count_matches(templates: np.ndarray, dim: int) -> int:
            count = 0
            n_templates = len(templates)

            # 使用KDTree加速搜索
            if n_templates > 1000:
                tree = scipy.spatial.cKDTree(templates)
                pairs = tree.query_pairs(r, output_type='ndarray')
                count = len(pairs)
            else:
                # 小数据集直接计算
                for i in range(n_templates):
                    for j in range(i + 1, n_templates):
                        if _maxdist(templates[i], templates[j]) < r:
                            count += 1
            return count

        # 迭代计算直到收敛
        prev_sampen = None
        for iteration in range(max_iter):
            # 计算m维匹配
            templates_m = _get_templates(m)
            b = _count_matches(templates_m, m)

            # 计算m+1维匹配
            templates_m1 = _get_templates(m + 1)
            a = _count_matches(templates_m1, m + 1)

            # 避免除零错误
            if b == 0:
                # 没有匹配，序列高度规则，返回0
                return 0.0
            if a == 0:
                # 在m+1维没有匹配，返回无穷大
                return float('inf')

            # 计算样本熵
            ratio = a / b
            sampen = -np.log(ratio)

            # 检查收敛
            if prev_sampen is not None and abs(sampen - prev_sampen) < tolerance:
                return sampen
            prev_sampen = sampen

        return sampen

    @staticmethod
    def multiscale_entropy(rr_intervals: np.ndarray,
                           scale_max: int = 10,
                           m: int = 2,
                           r: float = 0.15) -> np.ndarray:
        """
        计算多尺度样本熵

        参数:
        rr_intervals: RR间期序列(毫秒)
        scale_max: 最大尺度
        m: 模板长度
        r: 容差系数

        返回:
        mse: 各尺度的样本熵值
        """
        n = len(rr_intervals)
        if n < scale_max * 100:
            raise ValueError(f"多尺度熵计算需要至少{scale_max * 100}个数据点")

        mse = np.zeros(scale_max)

        for scale in range(1, scale_max + 1):
            # 创建粗粒度序列
            coarse_grained = []
            for i in range(0, n - scale + 1, scale):
                coarse_grained.append(np.mean(rr_intervals[i:i + scale]))

            # 计算样本熵
            try:
                sampen = NonlinearHRVAnalysis.sample_entropy(
                    np.array(coarse_grained), m, r, normalize=True
                )
                mse[scale - 1] = sampen
            except Exception as e:
                print(f"尺度 {scale} 计算失败: {str(e)}")
                mse[scale - 1] = np.nan

        return mse

    @staticmethod
    def detrended_fluctuation_analysis(rr_intervals: np.ndarray,
                                       n_boxes: int = 16) -> Tuple[float, np.ndarray]:
        """
        计算去趋势波动分析(DFA)指标

        参数:
        rr_intervals: RR间期序列(毫秒)
        n_boxes: 盒子数量

        返回:
        alpha: DFA指数
        fluctuations: 各尺度的波动值
        """
        n = len(rr_intervals)
        if n < 100:
            raise ValueError("DFA计算至少需要100个数据点")

        # 累积和
        y = np.cumsum(rr_intervals - np.mean(rr_intervals))

        # 创建盒子大小
        box_sizes = np.logspace(np.log10(4), np.log10(n // 4), n_boxes, dtype=int)
        box_sizes = np.unique(box_sizes)  # 确保唯一值
        box_sizes = box_sizes[box_sizes < n // 4]  # 确保有效

        fluctuations = []

        for size in box_sizes:
            # 分段
            n_boxes = n // size
            f2 = 0

            for i in range(n_boxes):
                # 获取当前段
                segment = y[i * size:(i + 1) * size]

                # 线性拟合去趋势
                x = np.arange(size)
                slope, intercept = np.polyfit(x, segment, 1)
                trend = slope * x + intercept

                # 计算波动
                f2 += np.sum((segment - trend) ** 2)

            # 平均波动
            f2 = np.sqrt(f2 / (n_boxes * size))
            fluctuations.append(f2)

        # 对数变换后线性拟合
        log_box = np.log(box_sizes)
        log_fluct = np.log(fluctuations)

        # 使用稳健线性回归
        slope, intercept = np.polyfit(log_box, log_fluct, 1)

        return slope, np.array(fluctuations)

    @staticmethod
    def recurrence_quantification_analysis(rr_intervals: np.ndarray,
                                           threshold: float = 0.1,
                                           embedding_dim: int = 3,
                                           delay: int = 1) -> dict:
        """
        递归量化分析(RQA)

        参数:
        rr_intervals: RR间期序列(毫秒)
        threshold: 递归阈值
        embedding_dim: 嵌入维度
        delay: 延迟时间

        返回:
        rqa_metrics: RQA指标字典
        """
        n = len(rr_intervals)
        if n < 100:
            raise ValueError("RQA计算至少需要100个数据点")

        # 相空间重构
        embedded = []
        for i in range(n - (embedding_dim - 1) * delay):
            embedded.append(rr_intervals[i:i + embedding_dim * delay:delay])
        embedded = np.array(embedded)

        # 创建递归图
        dist_matrix = scipy.spatial.distance_matrix(embedded, embedded)
        recurrence_matrix = dist_matrix < threshold

        # 计算RQA指标
        n_points = recurrence_matrix.shape[0]

        # 递归率
        recurrence_rate = np.sum(recurrence_matrix) / (n_points ** 2)

        # 确定性
        diagonal_lines = []
        for i in range(n_points):
            j = i
            line_length = 0
            while j < n_points and recurrence_matrix[i, j]:
                line_length += 1
                j += 1
            if line_length > 1:
                diagonal_lines.append(line_length)

        if diagonal_lines:
            determinism = np.sum(diagonal_lines) / np.sum(recurrence_matrix)
            avg_diag = np.mean(diagonal_lines)
            max_diag = np.max(diagonal_lines)
        else:
            determinism = 0
            avg_diag = 0
            max_diag = 0

        # 层流性
        vertical_lines = []
        for j in range(n_points):
            i = 0
            while i < n_points and recurrence_matrix[i, j]:
                i += 1
            line_length = i
            if line_length > 1:
                vertical_lines.append(line_length)

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