import unittest
import numpy as np
from .time_domain import HRVTimeDomainAnalysis, validate_rr_intervals


class TestHRVTimeDomainAnalysis(unittest.TestCase):

    def setUp(self):
        # 生成模拟RR间期数据(5分钟记录，平均心率60bpm)
        np.random.seed(42)
        self.rr_ms = np.random.normal(loc=1000, scale=50, size=300)

        # 添加一些异常值
        self.rr_ms[10] = 2000  # 长间期
        self.rr_ms[20] = 500  # 短间期
        self.rr_ms[30] = 3000  # 极端值

        # 创建异位搏动
        self.rr_ms[40] = 500
        self.rr_ms[41] = 1500

        # 有效数据点
        self.valid_rr = [800, 900, 1000, 1100, 900, 950, 1050]

    def test_validation(self):
        # 测试无效输入
        with self.assertRaises(ValueError):
            validate_rr_intervals([100, 0, 800])

        with self.assertRaises(ValueError):
            validate_rr_intervals([100])

        with self.assertRaises(TypeError):
            validate_rr_intervals("not a list")

        with self.assertRaises(TypeError):
            validate_rr_intervals([100, 'a', 800])

    def test_sdnn_calculation(self):
        # 测试SDNN计算
        hrv = HRVTimeDomainAnalysis(self.valid_rr)
        manual_std = np.std(self.valid_rr, ddof=1)
        self.assertAlmostEqual(hrv.sdnn(), manual_std, places=5)

        # 测试预处理效果
        hrv_raw = HRVTimeDomainAnalysis(self.rr_ms, ectopic_correction=False)
        hrv_clean = HRVTimeDomainAnalysis(self.rr_ms)
        self.assertLess(hrv_clean.sdnn(), hrv_raw.sdnn())

    def test_rmssd_calculation(self):
        # 测试RMSSD计算
        hrv = HRVTimeDomainAnalysis(self.valid_rr)
        diff = np.diff(self.valid_rr)
        rms = np.sqrt(np.mean(diff ** 2))
        self.assertAlmostEqual(hrv.rmssd(), rms, places=5)

        # 测试异位搏动校正
        rr_with_ectopic = [800, 800, 800, 500, 800, 800, 800]  # 包含异位搏动
        hrv = HRVTimeDomainAnalysis(rr_with_ectopic)
        self.assertLess(hrv.rmssd(), 200)  # 校正后RMSSD应降低

    def test_pnn50_calculation(self):
        # 创建具有特定差异的序列
        rr = [1000, 1050, 950, 1000, 1100, 900]  # 差异: 50, 100, 50, 100, 200
        hrv = HRVTimeDomainAnalysis(rr)
        # 差异超过50ms的数量: 3 (1050-950=100, 1000-1100=100, 1100-900=200)
        # 总差异数: 5
        self.assertAlmostEqual(hrv.pnn50(), 60.0, places=5)  # 3/5 * 100=60

        # 测试边界条件
        hrv = HRVTimeDomainAnalysis([1000, 1000])
        self.assertEqual(hrv.pnn50(), 0.0)

    def test_full_analysis(self):
        # 测试完整分析输出
        hrv = HRVTimeDomainAnalysis(self.valid_rr)
        results = hrv.full_analysis()

        self.assertIn('sdnn', results)
        self.assertIn('rmssd', results)
        self.assertIn('pnn50', results)
        self.assertIn('nn50', results)
        self.assertIn('mean_rr', results)
        self.assertIn('median_rr', results)
        self.assertIn('hr_triangular_index', results)
        self.assertIn('tinn', results)

        # 验证值的一致性
        self.assertAlmostEqual(results['sdnn'], hrv.sdnn())
        self.assertAlmostEqual(results['rmssd'], hrv.rmssd())

    def test_resampling(self):
        # 测试重采样功能
        hrv = HRVTimeDomainAnalysis(self.valid_rr, sampling_frequency=4)

        # 计算预期点数
        total_time = np.sum(self.valid_rr) / 1000.0  # 总时间(秒)
        expected_samples = int(np.ceil(total_time * 4)) + 1
        self.assertEqual(len(hrv.clean_rr), expected_samples)

    def test_outlier_handling(self):
        # 测试离群值处理
        rr_with_outliers = [1000, 2000, 500, 1000, 3000, 1000]
        hrv = HRVTimeDomainAnalysis(rr_with_outliers)

        # 离群值应被修正
        self.assertTrue(np.all(hrv.clean_rr > 500))
        self.assertTrue(np.all(hrv.clean_rr < 2000))

        # 统计值应合理
        self.assertLess(hrv.sdnn(), 300)

    def test_edge_cases(self):
        # 测试小样本
        with self.assertRaises(ValueError):
            HRVTimeDomainAnalysis([1000])

        # 测试恒定心率
        hrv = HRVTimeDomainAnalysis([1000, 1000, 1000, 1000])
        self.assertEqual(hrv.sdnn(), 0.0)
        self.assertEqual(hrv.rmssd(), 0.0)
        self.assertEqual(hrv.pnn50(), 0.0)

        # 测试单位转换
        hrv_ms = HRVTimeDomainAnalysis([1.0, 1.0], units='s')
        hrv_s = HRVTimeDomainAnalysis([1000, 1000], units='ms')
        self.assertEqual(hrv_ms.mean_rr(), hrv_s.mean_rr())

        # 测试空输入
        with self.assertRaises(ValueError):
            HRVTimeDomainAnalysis([])

        # 测试单个点
        with self.assertRaises(ValueError):
            HRVTimeDomainAnalysis([1000])

        # 测试负值
        with self.assertRaises(ValueError):
            HRVTimeDomainAnalysis([1000, -500, 800])

    def test_ectopic_correction(self):
        # 测试异位搏动校正
        rr = [800, 800, 800, 500, 800, 800, 800]
        hrv_no_correction = HRVTimeDomainAnalysis(rr, ectopic_correction=False)
        hrv_with_correction = HRVTimeDomainAnalysis(rr, ectopic_correction=True)

        # 校正后RMSSD应降低
        self.assertLess(hrv_with_correction.rmssd(), hrv_no_correction.rmssd())

        # 校正后序列应更平滑
        self.assertLess(np.std(np.diff(hrv_with_correction.clean_rr)),
                        np.std(np.diff(hrv_no_correction.clean_rr)))

    def test_constant_heart_rate(self):
        # 测试恒定心率
        rr = [1000] * 100
        hrv = HRVTimeDomainAnalysis(rr)

        self.assertEqual(hrv.sdnn(), 0.0)
        self.assertEqual(hrv.rmssd(), 0.0)
        self.assertEqual(hrv.pnn50(), 0.0)
        self.assertEqual(hrv.nn50(), 0)
        self.assertEqual(hrv.mean_rr(), 1000.0)
        self.assertEqual(hrv.median_rr(), 1000.0)

    def test_all_values_same(self):
        # 测试所有值相同的情况
        rr = [1000, 1000, 1000, 1000, 1000]
        hrv = HRVTimeDomainAnalysis(rr)

        # 确保不会出现除零错误
        self.assertEqual(hrv.sdnn(), 0.0)
        self.assertEqual(hrv.rmssd(), 0.0)
        self.assertEqual(hrv.pnn50(), 0.0)

        # 测试离群值处理
        self.assertTrue(np.all(hrv.clean_rr == 1000))


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)