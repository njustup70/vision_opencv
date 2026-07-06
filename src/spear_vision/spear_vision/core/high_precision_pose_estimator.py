#!/usr/bin/env python3
"""
ChArUco 高精度位姿估计器 (从 spear/arucopnp.py 迁入 spear_vision 包)
"""
import cv2
import numpy as np


class HighPrecisionPoseEstimator:
    def __init__(self, K=None, D=None):
        cv_version = cv2.__version__.split('.')
        cv_major = int(cv_version[0])
        cv_minor = int(cv_version[1])

        assert cv_major > 4 or (cv_major == 4 and cv_minor >= 10), \
            f"OpenCV >= 4.10.0 required, got {cv2.__version__}"

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
        self.board = cv2.aruco.CharucoBoard((3, 3), 0.03, 0.022, self.dictionary)

        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.cornerRefinementWinSize = 4
        self.detector_params.cornerRefinementMaxIterations = 20
        self.detector_params.cornerRefinementMinAccuracy = 0.02

        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.board,
            detectorParams=self.detector_params,
        )

        if K is not None:
            self.K = K
        else:
            self.K = np.array([[800.0, 0, 320.0],
                               [0, 800.0, 240.0],
                               [0, 0, 1.0]], dtype=np.float32)

        if D is not None:
            self.D = D
        else:
            self.D = np.zeros((5, 1), dtype=np.float32)

        self.last_rvec = None
        self.last_tvec = None

    def on_image(self, frame):
        if frame is None:
            return None, None

        c_corners, c_ids, m_corners, m_ids = self.charuco_detector.detectBoard(frame)

        obj_points, img_points = None, None
        best_rvec, best_tvec = None, None

        if c_ids is not None and len(c_ids) >= 4:
            obj_points, img_points = self.board.matchImagePoints(c_corners, c_ids)
        elif m_ids is not None and len(m_ids) >= 1:
            obj_list = []
            img_list = []
            board_ids = self.board.getIds()
            board_obj_points = self.board.getObjPoints()
            for i, m_id in enumerate(m_ids.flatten()):
                idx = np.where(board_ids == m_id)[0]
                if len(idx) > 0:
                    obj_list.append(board_obj_points[idx[0]])
                    img_list.append(m_corners[i].reshape(4, 2))
            obj_points = np.concatenate(obj_list, axis=0) if obj_list else None
            img_points = np.concatenate(img_list, axis=0) if img_list else None
        else:
            return None, None

        if obj_points is not None and len(obj_points) >= 4:
            assert isinstance(img_points, np.ndarray) and isinstance(obj_points, np.ndarray)
            retval, rvecs, tvecs = 0, None, None
            if retval <= 0:
                try:
                    retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                        obj_points, img_points, self.K, self.D,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                except cv2.error:
                    retval = 0
            if retval > 0:
                best_rvec, best_tvec = self._select_best_pose(rvecs, tvecs)

        self._draw_result(frame, c_corners, c_ids, m_corners, m_ids, best_rvec, best_tvec)
        return best_rvec, best_tvec

    def _select_best_pose(self, rvecs, tvecs):
        if len(rvecs) == 1:
            return rvecs[0], tvecs[0]
        if self.last_rvec is not None:
            dist0 = np.linalg.norm(rvecs[0] - self.last_rvec)
            dist1 = np.linalg.norm(rvecs[1] - self.last_rvec)
            idx = 0 if dist0 < dist1 else 1
        else:
            idx = 0
        self.last_rvec = rvecs[idx]
        self.last_tvec = tvecs[idx]
        return rvecs[idx], tvecs[idx]

    def _draw_result(self, frame, corners, ids, m_corners, m_ids, rvec, tvec):
        if m_corners is not None and m_ids is not None and len(m_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, m_corners, None)
        if corners is not None and ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, corners, None)
        if rvec is not None and tvec is not None:
            cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, 0.1)
        return frame
