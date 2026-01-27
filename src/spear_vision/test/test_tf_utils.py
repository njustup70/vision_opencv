import numpy as np

from spear_vision.utils.tf_utils import (
    Rt,
    compose_rt,
    invert_rt,
    rmat_to_rpy_deg,
    rpy_deg_to_rvec,
    rodrigues_to_matrix,
)


def test_compose_invert_roundtrip():
    rt = Rt(
        rvec=rpy_deg_to_rvec(10.0, -5.0, 20.0),
        tvec=np.array([[0.12], [-0.03], [0.45]], dtype=np.float64),
    )
    inv = invert_rt(rt)
    ident = compose_rt(rt, inv)

    rmat = rodrigues_to_matrix(ident.rvec)
    assert np.allclose(rmat, np.eye(3), atol=1e-6)
    assert np.allclose(ident.tvec, np.zeros((3, 1)), atol=1e-6)


def test_rpy_roundtrip():
    roll, pitch, yaw = 15.0, -7.0, 30.0
    rvec = rpy_deg_to_rvec(roll, pitch, yaw)
    rmat = rodrigues_to_matrix(rvec)
    roll2, pitch2, yaw2 = rmat_to_rpy_deg(rmat)
    assert np.allclose([roll2, pitch2, yaw2], [roll, pitch, yaw], atol=1e-3)
