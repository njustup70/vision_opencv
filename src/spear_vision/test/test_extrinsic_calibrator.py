import numpy as np

from spear_vision.core.extrinsic_calibrator import ExtrinsicCalibrator
from spear_vision.utils.tf_utils import Rt, rpy_deg_to_rvec


def _make_rt(tx, ty, tz, roll, pitch, yaw) -> Rt:
    return Rt(
        rvec=rpy_deg_to_rvec(roll, pitch, yaw),
        tvec=np.array([[tx], [ty], [tz]], dtype=np.float64),
    )


def test_extrinsic_calibrator_outlier_rejection():
    cal = ExtrinsicCalibrator()

    # 五个“正常样本”
    for i in range(5):
        rt = _make_rt(0.1 + 0.0001 * i, 0.0, 0.0, 0.0, 0.0, 0.0)
        cal.add_sample(rt, conf_primary=0.9, conf_secondary=0.9, frame_index=i, stride=1, min_conf_p=0.5, min_conf_s=0.5)

    # 一个明显离群样本（平移/旋转都偏大）
    outlier = _make_rt(0.2, 0.0, 0.0, 0.0, 0.0, 30.0)
    cal.add_sample(outlier, conf_primary=0.9, conf_secondary=0.9, frame_index=99, stride=1, min_conf_p=0.5, min_conf_s=0.5)

    finalized, mean_rt, stats = cal.maybe_finalize(
        required_samples=5, outlier_translation_m=0.01, outlier_rotation_deg=5.0
    )

    assert finalized is True
    assert mean_rt is not None
    assert stats.num_kept == 5
    assert np.allclose(mean_rt.tvec.reshape(3), np.array([0.1002, 0.0, 0.0]), atol=1e-3)
