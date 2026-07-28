"""G1 head-camera extrinsics for decoding EgoVLA's camera-frame EE poses.

EgoVLA is H1-calibrated; its decoder emits wrist poses in the head-camera optical
frame. To turn those into G1 pelvis-frame poses (what the bridge/IK need) we use
the G1 D435 optical frame in the pelvis frame:

    T_pelvis_optical = FK(d435_link) @ T_d435link_optical

The Isaac-Lab sim mounts `front_cam` on `d435_link` with pos_offset=0,
rot_offset=(0.5,-0.5,0.5,-0.5) wxyz (ROS convention). d435_link is fixed off
torso_link, so at the tabletop (waist=0) config this is effectively constant.

EgoVLA's `cam_frame_poses_to_world_frame(p) = main_cam @ inv(CAM_AXIS) @ p` with
`main_cam = T_pelvis_optical @ CAM_AXIS`, so it simplifies to `T_pelvis_optical @ p`
— that is exactly what `EgoVLADecoder` applies.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

# Isaac-Lab g1_front_camera preset: prim ".../d435_link/front_cam",
# pos_offset=(0,0,0), rot_offset=(0.5,-0.5,0.5,-0.5) wxyz, convention="ros".
_CAM_ROT_OFFSET_WXYZ = (0.5, -0.5, 0.5, -0.5)
_CAM_POS_OFFSET = (0.0, 0.0, 0.0)
_D435_FRAME = "d435_link"

# EgoVLA's fixed optical-axis convention (robot-independent).
CAM_AXIS_TRANSFORM = np.array(
    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)


def _R_from_quat_wxyz(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return pin.Quaternion(float(w), float(x), float(y), float(z)).normalized().matrix()


def d435_optical_offset() -> np.ndarray:
    """T_d435link_optical (4x4): the ROS optical frame relative to d435_link."""
    T = np.eye(4)
    T[:3, :3] = _R_from_quat_wxyz(_CAM_ROT_OFFSET_WXYZ)
    T[:3, 3] = _CAM_POS_OFFSET
    return T


def T_pelvis_optical(kin, q_full: np.ndarray | None = None) -> np.ndarray:
    """Camera optical frame in the G1 pelvis (base) frame, via FK of d435_link."""
    if q_full is None:
        q_full = pin.neutral(kin.model)
    pos, R = kin.fk_frame_pose(q_full, _D435_FRAME)
    T_pd = np.eye(4)
    T_pd[:3, :3] = R
    T_pd[:3, 3] = pos
    return T_pd @ d435_optical_offset()
