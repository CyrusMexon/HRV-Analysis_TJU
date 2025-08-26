import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .nonlinear import NonlinearHRVAnalysis
import numpy as np
import pytest
import random


def generate_realistic_hrv(n_points=500):
    """生成更真实的HRV测试数据"""
    # 基础心率
    base_hr = np.random.normal(800, 50, n_points)

    # 添加呼吸性窦性心律不齐
    respiratory_effect = 20 * np.sin(np.linspace(0, 10 * np.pi, n_points))

    # 添加随机变异
    random_variation = np.random.normal(0, 10, n_points)

    # 组合所有成分
    rr_intervals = base_hr + respiratory_effect + random_variation

    # 添加一些异常值
    outlier_indices = np.random.choice(n_points, size=int(n_points * 0.05), replace=False)
    rr_intervals[outlier_indices] = np.random.choice([300, 2000], size=len(outlier_indices))

    return rr_intervals.astype(int)


# 使用真实数据
RR_INTERVALS = generate_realistic_hrv(500)
RR_WITH_OUTLIERS = generate_realistic_hrv(500)


def test_preprocess_rr():
    """测试RR间期预处理功能"""
    cleaned_rr, corrected_percentage = NonlinearHRVAnalysis.preprocess_rr(
        RR_WITH_OUTLIERS, outlier_threshold=3.0, interpolation_method='cubic'
    )

    # 验证校正百分比计算
    assert 3 <= corrected_percentage <= 10

    # 验证异常值已被校正
    assert np.all(cleaned_rr >= 700) and np.all(cleaned_rr <= 900)

    # 测试线性插值
    cleaned_rr_linear, _ = NonlinearHRVAnalysis.preprocess_rr(
        RR_WITH_OUTLIERS, interpolation_method='linear'
    )
    assert np.all(cleaned_rr_linear >= 700) and np.all(cleaned_rr_linear <= 900)


def test_poincare_analysis():
    """测试Poincaré分析功能"""
    # 测试无预处理
    sd1, sd2, rr_clean, corrected_percentage = NonlinearHRVAnalysis.poincare_analysis(
        RR_INTERVALS, preprocess=False
    )
    assert corrected_percentage == 0
    # 扩大预期范围以适应真实数据
    assert 5 < sd1 < 200  # 正常HRV的SD1通常在5-200ms之间
    assert 10 < sd2 < 300  # SD2通常比SD1大

    # 测试有预处理
    sd1, sd2, rr_clean, corrected_percentage = NonlinearHRVAnalysis.poincare_analysis(
        RR_WITH_OUTLIERS, preprocess=True
    )
    assert corrected_percentage > 0
    # 预处理后范围应该更小
    assert 5 < sd1 < 100
    assert 10 < sd2 < 200


def test_sample_entropy():
    """测试样本熵计算功能"""
    # 测试无预处理
    sampen = NonlinearHRVAnalysis.sample_entropy(RR_INTERVALS, preprocess=False)
    assert 0.1 < sampen < 3.0

    # 测试有预处理
    sampen_preprocessed = NonlinearHRVAnalysis.sample_entropy(
        RR_WITH_OUTLIERS, preprocess=True
    )
    assert 0.1 < sampen_preprocessed < 3.0


def test_multiscale_entropy():
    """测试多尺度熵计算功能"""
    mse = NonlinearHRVAnalysis.multiscale_entropy(RR_INTERVALS, scale_max=5)
    assert len(mse) == 5
    assert np.all(~np.isnan(mse))


def test_detrended_fluctuation_analysis():
    """测试去趋势波动分析功能"""
    alpha1, alpha2, fluctuations = NonlinearHRVAnalysis.detrended_fluctuation_analysis(RR_INTERVALS)

    # 验证返回了短期和长期标度指数
    assert 0.5 < alpha1 < 1.5
    # 长期指数可能为NaN（短序列无长期指数）
    if not np.isnan(alpha2):
        assert 0.5 < alpha2 < 1.5

    # 验证波动值
    assert len(fluctuations) > 0


def test_recurrence_quantification_analysis():
    """测试递归量化分析功能"""
    rqa_metrics = NonlinearHRVAnalysis.recurrence_quantification_analysis(RR_INTERVALS)

    # 验证返回了所有RQA指标
    assert 'recurrence_rate' in rqa_metrics
    assert 'determinism' in rqa_metrics
    assert 'laminarity' in rqa_metrics
    assert 0 <= rqa_metrics['recurrence_rate'] <= 1


def test_error_handling():
    """测试错误处理功能"""
    # 测试无效输入类型
    with pytest.raises(TypeError):
        NonlinearHRVAnalysis.poincare_analysis([800, 810, 790])

    # 测试序列过短
    with pytest.raises(ValueError):
        NonlinearHRVAnalysis.poincare_analysis(np.array([800, 810]))

    # 测试无效插值方法
    with pytest.raises(ValueError):
        NonlinearHRVAnalysis.preprocess_rr(RR_INTERVALS, interpolation_method='invalid')

    # 测试样本熵数据不足
    with pytest.raises(ValueError):
        NonlinearHRVAnalysis.sample_entropy(np.ones(50))


if __name__ == "__main__":
    pytest.main([__file__])