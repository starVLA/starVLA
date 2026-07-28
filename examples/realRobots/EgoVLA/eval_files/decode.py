"""Decode EgoVLA's raw 48-dim action into G1 pelvis-frame EE poses + Inspire hand qpos.

Faithful port of the decode half of EgoVLA's ``ik_eval_single_step`` (the model
forward is replaced by the starVLA EgoVLA framework). Per step, the 48-dim vector
is ``[wrist_trans 2x3 | MANO hand 2x15 | wrist rot6d 2x6]`` (camera frame). We:

  1. rot6d -> rotation matrix (HMR2 convention);
  2. MANO hand -> denorm -> ``mano_forward_retarget`` (MANO FK) -> fingertips ->
     ``HandActuationNet`` -> Inspire-12 qpos per hand  (capability-complete hand:
     real per-finger articulation, so a commanded grasp physically closes);
  3. wrist (rotmat, trans) -> pelvis-frame EE pose via the retarget-axis + pelvis
     offset + G1 camera transform (``T_pelvis_optical``).

Only ``mano_forward_retarget`` (the MANO kinematic model) is imported from EgoVLA
— it needs the license-gated MANO models. Everything else is self-contained here.
Set ``EGOVLA_RELEASE`` (env) to the EgoVLA_Release checkout; the MANO ``.pkl`` and
the hand-retarget ``.pth`` weights must be present there (see README).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

# --- EgoVLA MANO FK (license-gated model) -----------------------------------
_EGOVLA = os.environ.get("EGOVLA_RELEASE", "/home/dhy/Projects/EgoVLA_Release")


def _install_numpy_aliases():
    """chumpy (pulled in when smplx unpickles the MANO .pkl) uses numpy aliases
    removed in numpy>=1.24; restore them so the MANO model loads under numpy 1.26."""
    for _n, _t in {"bool": bool, "int": int, "float": float, "complex": complex,
                   "object": object, "str": str, "unicode": str}.items():
        if not hasattr(np, _n):
            setattr(np, _n, _t)
    for _n, _v in {"nan": float("nan"), "inf": float("inf")}.items():
        if not hasattr(np, _n):
            setattr(np, _n, _v)


def _load_mano_forward_retarget():
    _install_numpy_aliases()
    if _EGOVLA not in sys.path:
        sys.path.insert(0, _EGOVLA)
    # EgoVLA's mano/model.py loads the MANO .pkl via a path relative to CWD
    # ("mano_v1_2/models/..."), so import it with CWD at the EgoVLA checkout.
    _cwd = os.getcwd()
    try:
        os.chdir(_EGOVLA)
        from human_plan.utils.mano.forward import mano_forward_retarget  # noqa: E402
    finally:
        os.chdir(_cwd)
    return mano_forward_retarget


# --- Constants (extracted from EgoVLA; see README) --------------------------
# denorm range for the 15 MANO hand-pose dims
_MANO_MIN = np.array([-1, 1.5, -2, -3, -1.5, -1] + [-4] * 9, dtype=np.float64)
_MANO_MAX = np.array([2.2, 3.5, 1, 0.5, 4, 5] + [4] * 9, dtype=np.float64)
_MANO_RANGE = _MANO_MAX - _MANO_MIN

# MANO -> retarget axis (ISAACLAB) per hand, and MANO pelvis offset per hand
_LEFT_AXIS = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
_RIGHT_AXIS = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64)
_LEFT_PELVIS = np.array([-0.09567, 0.006383, 0.006186], dtype=np.float64)
_RIGHT_PELVIS = np.array([0.09567, 0.006383, 0.006186], dtype=np.float64)

# fingertip picks from the MANO joint set (wrist + 5 fingertips); [1:] drops wrist
_MANO_TO_INSPIRE = [0, 4, 8, 12, 16, 20]


def denorm_hand_dof(hand_dof: np.ndarray) -> np.ndarray:
    return hand_dof * _MANO_RANGE + _MANO_MIN


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """6D rotation -> 3x3 (HMR2 convention; matches EgoVLA rotation_convert)."""
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = torch.nn.functional.normalize(a1)
    b2 = torch.nn.functional.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


class HandActuationNet(nn.Module):
    """MANO fingertips (2x15) -> Inspire qpos (2x12). Matches EgoVLA's net; loads
    ``hand_actuation_net.pth`` (input_dim=30, output_dim=24)."""

    def __init__(self, input_dim=30, output_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class EgoVLADecoder:
    """Turn EgoVLA 48-dim predictions into G1 pelvis-frame EE poses + Inspire-12."""

    def __init__(self, kin, device: str = "cuda", assets_dir: str | None = None):
        self.kin = kin
        self.device = device
        assets_dir = assets_dir or os.path.join(os.path.dirname(__file__), "assets")

        from .g1_camera import T_pelvis_optical
        self.T_po = T_pelvis_optical(kin)  # (4,4) pelvis<-optical, cam_frame->pelvis

        self._mano_fwd = _load_mano_forward_retarget()
        self.hand_net = HandActuationNet(30, 24).to(device).eval()
        self.hand_net.load_state_dict(
            torch.load(os.path.join(assets_dir, "hand_actuation_net.pth"), map_location=device)
        )

    # -- geometry -------------------------------------------------------------
    def _ee_pose(self, rotmat: np.ndarray, trans: np.ndarray, is_right: bool) -> np.ndarray:
        """(rotmat (T,3,3), trans (T,3)) camera-frame -> pelvis-frame (T,7) pos+quat wxyz."""
        axis = _RIGHT_AXIS if is_right else _LEFT_AXIS
        pelvis = _RIGHT_PELVIS if is_right else _LEFT_PELVIS
        ee_rot = rotmat @ np.linalg.inv(axis)
        global_trans = trans + pelvis
        Tn = rotmat.shape[0]
        transform = np.zeros((Tn, 4, 4))
        transform[:, 3, 3] = 1
        transform[:, :3, :3] = ee_rot
        transform[:, :3, 3] = global_trans
        # cam_frame_poses_to_world == main_cam @ inv(CAM_AXIS) @ p == T_pelvis_optical @ p
        world = self.T_po @ transform
        quat_xyzw = R.from_matrix(world[:, :3, :3]).as_quat()
        quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1)
        return np.concatenate([world[:, :3, 3], quat_wxyz], axis=-1)

    def _hand_inspire12(self, mano_hand: np.ndarray) -> np.ndarray:
        """mano_hand (T,2,15) normalized -> Inspire-12 per hand: (left(T,12), right(T,12))."""
        tips = {}
        for side, is_right in (("left", False), ("right", True)):
            dof = denorm_hand_dof(mano_hand[:, 1 if is_right else 0, :].astype(np.float64))
            dof[..., 6:] = 0.0
            dof_t = torch.tensor(dof, dtype=torch.float32, device=self.device)
            kps = self._mano_fwd(dof_t, is_right=is_right).detach()   # (T, J, 3)
            kps = kps[:, _MANO_TO_INSPIRE][:, 1:]                     # (T, 5, 3) fingertips
            tips[side] = kps.reshape(kps.shape[0], -1)                # (T, 15)
        with torch.inference_mode():
            out = self.hand_net(torch.cat([tips["left"], tips["right"]], dim=-1))  # (T,24)
        out = out.detach().float().cpu().numpy()
        return out[:, :12], out[:, 12:]

    # -- full decode ----------------------------------------------------------
    def decode(self, pred: np.ndarray) -> dict:
        """pred (T,48) -> dict of left/right EE pose (T,7 pelvis frame) + Inspire-12 (T,12)."""
        pred = np.asarray(pred, dtype=np.float64)
        pred_3d = pred[:, :6].reshape(-1, 2, 3)
        pred_hand = pred[:, 6:36].reshape(-1, 2, 15)
        pred_rot = pred[:, 36:]  # (T,12) = 2 x rot6d
        rotmat = rot6d_to_rotmat(torch.tensor(pred_rot)).view(-1, 2, 3, 3).numpy()

        left_ee = self._ee_pose(rotmat[:, 0], pred_3d[:, 0], is_right=False)
        right_ee = self._ee_pose(rotmat[:, 1], pred_3d[:, 1], is_right=True)
        left_h12, right_h12 = self._hand_inspire12(pred_hand)
        return {
            "left_ee_pose": left_ee, "right_ee_pose": right_ee,
            "left_inspire12": left_h12, "right_inspire12": right_h12,
        }
