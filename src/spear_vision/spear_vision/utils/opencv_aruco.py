"""
OpenCV ArUco/ChArUco 辅助函数（spear_vision）

为什么要封装：
- OpenCV 的 ArUco API 在不同版本/发行版上存在细微差异（尤其是返回值/类型）；
- 本包需要统一实现：detectMarkers → refineDetectedMarkers → interpolateCornersCharuco → subpix 细化；
- 支持“两块板同时出现”的场景：通过 ids_start 错开 ID 区间，避免两块板互相误匹配。

坐标/物理单位：
- Board 物理尺寸用米（m），这样 PnP 输出的 tvec 也是米。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def _require_aruco() -> None:
    # 风险提示（对应你总结的风险 B）：
    # - ArUco/ChArUco 历史上属于 opencv_contrib 模块；
    # - 在某些 pip wheel / 精简版 OpenCV 构建中可能没有 cv2.aruco；
    # - 本工程强依赖 ArUco/ChArUco，因此这里给出明确的报错与安装建议，避免“神秘 AttributeError”。
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV build has no 'cv2.aruco' module. "
            "Please install OpenCV with aruco/charuco support (e.g. Ubuntu apt python3-opencv, "
            "or opencv-contrib-python for pure pip environments)."
        )


def resolve_dictionary(name: str) -> "cv2.aruco_Dictionary":
    # 允许用户配置 "6X6_250" 或 "DICT_6X6_250"
    _require_aruco()
    if not name:
        raise ValueError("dictionary is empty")
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown dictionary '{name}'")
    dict_id = getattr(cv2.aruco, name)
    # OpenCV 不同版本可能是 getPredefinedDictionary 或 Dictionary_get
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "Dictionary_get"):
        return cv2.aruco.Dictionary_get(dict_id)
    raise RuntimeError("OpenCV aruco module does not provide dictionary getter API.")


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary: "cv2.aruco_Dictionary",
    ids_start: int = 0,
) -> "cv2.aruco_CharucoBoard":
    # 风险提示（对应你总结的风险 A）：
    # - OpenCV 4.7+ 起 aruco 的 Python API 有过破坏性变化；
    # - 旧式 CharucoBoard_create/DetectorParameters_create/detectMarkers 等接口可能不存在；
    # - 因此这里做“新旧 API 兼容”：
    #   * 旧版：cv2.aruco.CharucoBoard_create(...)
    #   * 新版：cv2.aruco.CharucoBoard(...)（类构造）或其它绑定形式
    _require_aruco()
    board = None
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        # OpenCV 4.5.x / 4.6.x 常见
        board = cv2.aruco.CharucoBoard_create(
            int(squares_x),
            int(squares_y),
            float(square_length_m),
            float(marker_length_m),
            dictionary,
        )
    elif hasattr(cv2.aruco, "CharucoBoard"):
        # OpenCV 4.7+ 常见：类绑定
        # 注意：不同发行版的 Python 绑定构造签名可能略有差异，这里做多种尝试。
        try:
            board = cv2.aruco.CharucoBoard(
                (int(squares_x), int(squares_y)),
                float(square_length_m),
                float(marker_length_m),
                dictionary,
            )
        except Exception:
            try:
                board = cv2.aruco.CharucoBoard(
                    int(squares_x),
                    int(squares_y),
                    float(square_length_m),
                    float(marker_length_m),
                    dictionary,
                )
            except Exception as exc:
                raise RuntimeError("Failed to construct CharucoBoard with this OpenCV build.") from exc
    else:
        raise RuntimeError("OpenCV aruco module does not provide CharucoBoard creation API.")

    if ids_start:
        # 默认情况下 OpenCV 会从 0..N-1 自动分配 marker id。
        # 对“两块板同时检测”的需求，必须把 ID 区间错开（例如大板 0.., 小板 100..）。
        ids = get_board_ids(board)
        num_markers = int(ids.size)
        if num_markers <= 0:
            # 极端情况下某些绑定可能不暴露 ids，但仍暴露 objPoints；我们用它推断 marker 数量
            obj_pts = get_board_obj_points(board)
            num_markers = int(len(obj_pts))
        if num_markers <= 0:
            raise RuntimeError(
                "Failed to determine number of markers for CharucoBoard (cannot apply ids_start offset). "
                "This OpenCV build may not expose board ids/objPoints in Python bindings."
            )
        desired_ids = (np.arange(num_markers, dtype=np.int32) + int(ids_start)).reshape(-1, 1)
        # 某些版本的 board.ids 可能是 (N,) 或 (N,1)；这里尽量写入兼容形状
        # OpenCV 4.7+ 的绑定大概率仍允许写 board.ids；但我们做多种尝试，给出明确错误提示。
        if hasattr(board, "ids"):
            try:
                board.ids = desired_ids
            except Exception:
                try:
                    board.ids = desired_ids.reshape(-1)
                except Exception as exc:
                    # 部分绑定可能提供 setIds()/set_ids()
                    _set_board_ids(board, desired_ids)
        else:
            _set_board_ids(board, desired_ids)

    return board


def _set_board_ids(board, desired_ids: np.ndarray) -> None:
    # 兼容：有些 OpenCV Python 绑定可能不允许直接赋值 board.ids，而是提供 setter。
    if hasattr(board, "setIds"):
        try:
            board.setIds(desired_ids)
            return
        except Exception:
            board.setIds(np.array(desired_ids, dtype=np.int32).reshape(-1).tolist())
            return
    if hasattr(board, "set_ids"):
        try:
            board.set_ids(desired_ids)
            return
        except Exception:
            board.set_ids(np.array(desired_ids, dtype=np.int32).reshape(-1).tolist())
            return
    raise RuntimeError(
        "This OpenCV CharucoBoard binding does not allow setting custom marker ids (ids_start). "
        "Workaround: use different dictionaries for different boards so IDs can start from 0 for both."
    )


def get_board_ids(board) -> np.ndarray:
    # 兼容：OpenCV 版本/绑定不同，board ids 的访问方式可能不同：
    # - 旧版：board.ids
    # - 新版：board.getIds()
    ids = getattr(board, "ids", None)
    if ids is None and hasattr(board, "getIds"):
        try:
            ids = board.getIds()
        except Exception:
            ids = None
    if ids is None and hasattr(board, "get_ids"):
        try:
            ids = board.get_ids()
        except Exception:
            ids = None
    if ids is None:
        return np.array([], dtype=np.int32)
    return np.array(ids, dtype=np.int32).reshape(-1)


def get_board_obj_points(board) -> list[np.ndarray]:
    # 兼容：旧版是 board.objPoints；新版可能是 board.getObjPoints()
    obj = getattr(board, "objPoints", None)
    if obj is None and hasattr(board, "getObjPoints"):
        try:
            obj = board.getObjPoints()
        except Exception:
            obj = None
    if obj is None and hasattr(board, "get_obj_points"):
        try:
            obj = board.get_obj_points()
        except Exception:
            obj = None
    if obj is None:
        return []
    # 一般是 tuple(list) of (4,3) 点；统一转成 list，便于后续索引
    return list(obj)


def get_charuco_chessboard_corners(board) -> np.ndarray:
    # 兼容：旧版是 board.chessboardCorners；新版可能是 board.getChessboardCorners()
    corners = getattr(board, "chessboardCorners", None)
    if corners is None and hasattr(board, "getChessboardCorners"):
        try:
            corners = board.getChessboardCorners()
        except Exception:
            corners = None
    if corners is None and hasattr(board, "get_chessboard_corners"):
        try:
            corners = board.get_chessboard_corners()
        except Exception:
            corners = None
    if corners is None:
        raise RuntimeError("OpenCV CharucoBoard has no chessboard corners API (chessboardCorners/getChessboardCorners).")
    return np.array(corners, dtype=np.float64).reshape(-1, 3)


def board_marker_id_set(board) -> set[int]:
    # 将 board.ids 转成 python set，便于快速判断某个 marker 是否属于该 board
    ids = get_board_ids(board)
    if ids.size == 0:
        return set()
    return set(int(x) for x in ids.reshape(-1).tolist())


def filter_markers_for_board(
    corners: list[np.ndarray],
    ids: Optional[np.ndarray],
    board,
) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
    # detectMarkers 可能会检测出其它物体上的 marker，
    # 这里按 board.ids 过滤，只保留属于该 board 的 marker。
    if ids is None or len(ids) == 0:
        return [], None
    keep = board_marker_id_set(board)
    indices = [i for i, mid in enumerate(ids.reshape(-1).tolist()) if int(mid) in keep]
    if not indices:
        return [], None
    filtered_corners = [corners[i] for i in indices]
    filtered_ids = ids[indices]
    return filtered_corners, filtered_ids


def create_detector_parameters() -> "cv2.aruco_DetectorParameters":
    # 这些参数对“弱光/反光/轻微模糊/曝光漂移”的鲁棒性影响很大：
    # - 阈值窗口范围影响二值化质量（过小会噪声多，过大对局部阴影敏感）
    # - cornerRefinement* 影响角点精度与稳定性（直接影响 PnP）
    _require_aruco()
    # OpenCV 4.7+ 可能是 DetectorParameters()；旧版是 DetectorParameters_create()
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
    elif hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        raise RuntimeError("OpenCV aruco module does not provide DetectorParameters API.")

    # 兼顾弱光/反光/轻微模糊：适当放宽阈值窗口与角点细化
    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 3
    if hasattr(params, "adaptiveThreshWinSizeMax"):
        params.adaptiveThreshWinSizeMax = 23
    if hasattr(params, "adaptiveThreshWinSizeStep"):
        params.adaptiveThreshWinSizeStep = 10

    if hasattr(params, "cornerRefinementMethod"):
        # 某些版本常量名可能不存在；不存在时就不设置（仍可运行）
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(params, "cornerRefinementWinSize"):
        params.cornerRefinementWinSize = 5
    if hasattr(params, "cornerRefinementMaxIterations"):
        params.cornerRefinementMaxIterations = 30
    if hasattr(params, "cornerRefinementMinAccuracy"):
        params.cornerRefinementMinAccuracy = 0.1

    return params


@dataclass(frozen=True)
class MarkerDetection:
    corners: list[np.ndarray]
    ids: Optional[np.ndarray]
    rejected: list[np.ndarray]


# ArucoDetector（OpenCV 4.7+）缓存：
# - 新 API 推荐创建 detector 对象再 detectMarkers；
# - 我们保持函数式接口不变，但内部做缓存避免每帧重复构造对象。
_ARUCO_DETECTOR_CACHE: dict[tuple[int, int], object] = {}


def _get_aruco_detector(dictionary: "cv2.aruco_Dictionary", detector_params: "cv2.aruco_DetectorParameters"):
    if not hasattr(cv2.aruco, "ArucoDetector"):
        return None
    key = (id(dictionary), id(detector_params))
    det = _ARUCO_DETECTOR_CACHE.get(key)
    if det is None:
        det = cv2.aruco.ArucoDetector(dictionary, detector_params)
        _ARUCO_DETECTOR_CACHE[key] = det
    return det


def detect_markers(
    gray: np.ndarray,
    dictionary: "cv2.aruco_Dictionary",
    detector_params: "cv2.aruco_DetectorParameters",
) -> MarkerDetection:
    # gray 必须是单通道图像（mono8）
    _require_aruco()
    # 兼容 OpenCV 4.7+（ArucoDetector）与旧版（cv2.aruco.detectMarkers）
    det = _get_aruco_detector(dictionary, detector_params)
    if det is not None and hasattr(det, "detectMarkers"):
        corners, ids, rejected = det.detectMarkers(gray)
    else:
        # 旧版 API
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)
    return MarkerDetection(corners=corners, ids=ids, rejected=rejected)


def refine_markers(
    gray: np.ndarray,
    board,
    detection: MarkerDetection,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
) -> MarkerDetection:
    # refineDetectedMarkers 会利用 board 的几何布局 + rejectedCandidates 尝试“找回漏检”
    # 若提供相机内参，则 refine 会更可靠（但内参不是必需）
    _require_aruco()
    # refineDetectedMarkers：旧版是模块函数；新版有的版本仍保留该函数。
    if hasattr(cv2.aruco, "refineDetectedMarkers"):
        corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
            gray,
            board,
            detection.corners,
            detection.ids,
            detection.rejected,
            camera_matrix,
            dist_coeffs,
        )
        return MarkerDetection(corners=corners, ids=ids, rejected=rejected)

    # 若当前 OpenCV 版本没有 refineDetectedMarkers，则退化为“不 refine”，保证不报错可运行
    return detection


def interpolate_charuco_corners(
    gray: np.ndarray,
    board,
    corners: list[np.ndarray],
    ids: Optional[np.ndarray],
) -> tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    # 由已识别的 marker 推断出棋盘“内部角点”（ChArUco corners）。
    # ChArUco 角点是亚像素级可优化点，通常比 marker 四角更稳定，用于高精度 PnP/标定。
    _require_aruco()
    if ids is None or len(ids) == 0:
        return 0, None, None

    def _legacy_interp(b, corners_in: list[np.ndarray], ids_in: np.ndarray):
        if not hasattr(cv2.aruco, "interpolateCornersCharuco"):
            return 0, None, None
        try:
            num, cc, ci = cv2.aruco.interpolateCornersCharuco(corners_in, ids_in, gray, b)
        except Exception:
            return 0, None, None
        if cc is None or ci is None:
            return 0, None, None
        return int(num), cc, ci

    # 1) 优先走旧式 interpolateCornersCharuco（在 4.5.x/4.6.x/部分 4.7+ 仍可用）
    num, charuco_corners, charuco_ids = _legacy_interp(board, corners, ids)
    if int(num) > 0 and charuco_corners is not None and charuco_ids is not None:
        return int(num), charuco_corners, charuco_ids

    # 2) 风险 C 规避：部分 OpenCV 版本在 board.ids 不从 0 开始时，可能导致插值失败。
    #    这里做一个“ID 归一化”的兼容补丁：把 marker id 映射到 [0..N-1] 再插值。
    try:
        board_ids = get_board_ids(board)
        if board_ids.size > 0 and int(board_ids.min()) != 0:
            id_to_index = {int(mid): i for i, mid in enumerate(board_ids.tolist())}
            mapped_ids_list = []
            mapped_corners: list[np.ndarray] = []
            ids_flat = np.array(ids, dtype=np.int32).reshape(-1).tolist()
            for c, mid in zip(corners, ids_flat):
                if int(mid) in id_to_index:
                    mapped_corners.append(c)
                    mapped_ids_list.append(int(id_to_index[int(mid)]))
            if mapped_ids_list and mapped_corners:
                mapped_ids = np.array(mapped_ids_list, dtype=np.int32).reshape(-1, 1)

                # 用 board 自身的几何参数创建一个 ids 从 0 开始的“等价 board”
                if hasattr(board, "getChessboardSize") and hasattr(board, "getSquareLength") and hasattr(board, "getMarkerLength"):
                    sx, sy = board.getChessboardSize()  # (squaresX, squaresY)
                    # 不同 OpenCV 版本对 dictionary 的暴露方式不同，这里尽量兼容读取
                    dictionary = getattr(board, "dictionary", None)
                    if dictionary is None and hasattr(board, "getDictionary"):
                        try:
                            dictionary = board.getDictionary()
                        except Exception:
                            dictionary = None
                    if dictionary is None:
                        dictionary = getattr(board, "Dictionary", None)
                    if dictionary is not None:
                        norm_board = create_charuco_board(
                            int(sx),
                            int(sy),
                            float(board.getSquareLength()),
                            float(board.getMarkerLength()),
                            dictionary,
                            ids_start=0,
                        )
                        num2, cc2, ci2 = _legacy_interp(norm_board, mapped_corners, mapped_ids)
                        if int(num2) > 0 and cc2 is not None and ci2 is not None:
                            return int(num2), cc2, ci2
    except Exception:
        pass

    # 3) OpenCV 4.7+ 新式 CharucoDetector 兼容路径（当旧接口不存在时）
    if hasattr(cv2.aruco, "CharucoDetector"):
        try:
            det = cv2.aruco.CharucoDetector(board)
            # 不同版本返回值可能不同；尽量用“前两个”作为 (corners, ids)
            out = det.detectBoard(gray)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                cc = out[0]
                ci = out[1]
                if cc is not None and ci is not None and len(ci) > 0:
                    return int(len(ci)), cc, ci
        except Exception:
            pass

    return 0, None, None


def refine_charuco_corners_subpix(
    gray: np.ndarray,
    charuco_corners: np.ndarray,
    win_size: int = 3,
    max_iters: int = 30,
    eps: float = 0.01,
) -> np.ndarray:
    # 再做一次 cornerSubPix 细化，进一步提升角点精度与稳定性
    # （OpenCV 内部也可能细化，但这里显式做一次，便于参数化与统一行为）
    refined = np.array(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(max_iters), float(eps))
    cv2.cornerSubPix(gray, refined, (int(win_size), int(win_size)), (-1, -1), criteria)
    return refined


def draw_marker_ids(
    image: np.ndarray,
    corners: list[np.ndarray],
    ids: Optional[np.ndarray],
    color: tuple[int, int, int] = (0, 255, 0),
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    # 在图像上标注每个 ArUco marker 的 ID（满足你“看到 ID 号”的需求）
    if ids is None or len(ids) == 0:
        return
    ids_flat = np.array(ids, dtype=np.int32).reshape(-1).tolist()
    for i, mid in enumerate(ids_flat):
        if i >= len(corners):
            break
        pts = np.array(corners[i], dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            continue
        x, y = np.mean(pts, axis=0)
        cv2.putText(
            image,
            f"ID:{int(mid)}",
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(scale),
            color,
            int(thickness),
            lineType=cv2.LINE_AA,
        )


def draw_marker_corners(
    image: np.ndarray,
    corners: list[np.ndarray],
    color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 3,
    thickness: int = 1,
) -> None:
    # 标出每个 marker 的四个角点（便于你肉眼判断角点质量）
    for c in corners:
        pts = np.array(c, dtype=np.float32).reshape(-1, 2)
        for p in pts:
            cv2.circle(image, (int(p[0]), int(p[1])), int(radius), color, int(thickness))
