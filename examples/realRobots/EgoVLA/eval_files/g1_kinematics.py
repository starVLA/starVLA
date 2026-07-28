"""G1 kinematics via pinocchio: wrist forward kinematics + per-arm damped-least-squares IK.

Used by the adapter to (Stage 3) turn egoVLA's predicted wrist EE poses into G1
left/right arm 7-DoF joint targets — the action format the GR00T-WBC-Bridge expects
(it passes arm joints straight through, no IK of its own). Also (Stage 2) provides
forward kinematics for building egoVLA's proprio input from a G1 joint config.

URDF: Unitree g1_29dof_with_hand.urdf (Dex3 hand), the same model the bridge's
fk_helper uses; EE frames are left/right_wrist_yaw_link.
"""
from __future__ import annotations

import os

import numpy as np
import pinocchio as pin

_DEFAULT_URDF = os.path.join(os.path.dirname(__file__), "assets", "g1_29dof_with_hand.urdf")

# Unitree canonical per-arm joint order (shoulder p/r/y, elbow, wrist r/p/y).
_ARM_JOINTS = {
    "left": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
}
_WRIST_FRAME = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}


def _quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    q = pin.Quaternion(R)  # stores (x, y, z, w)
    return np.array([q.w, q.x, q.y, q.z], dtype=np.float64)


def _R_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return pin.Quaternion(w, x, y, z).normalized().matrix()


class G1Kinematics:
    def __init__(self, urdf_path: str = _DEFAULT_URDF):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nq = self.model.nq

        # Per-arm config indices (idx_q) and velocity/Jacobian-column indices (idx_v).
        self.arm_idx_q: dict[str, np.ndarray] = {}
        self.arm_idx_v: dict[str, np.ndarray] = {}
        self.wrist_frame_id: dict[str, int] = {}
        for side in ("left", "right"):
            jids = [self.model.getJointId(n) for n in _ARM_JOINTS[side]]
            self.arm_idx_q[side] = np.array([self.model.idx_qs[j] for j in jids], dtype=int)
            self.arm_idx_v[side] = np.array([self.model.idx_vs[j] for j in jids], dtype=int)
            fid = self.model.getFrameId(_WRIST_FRAME[side])
            assert fid < self.model.nframes, f"missing frame {_WRIST_FRAME[side]}"
            self.wrist_frame_id[side] = fid

        self.q_lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        self.q_upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64)

    # -- forward kinematics ----------------------------------------------------
    def fk_wrist(self, q_full: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos(3), quat_wxyz(4)) of the wrist frame for `side` at config q_full."""
        q = np.asarray(q_full, dtype=np.float64).copy()
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        M = self.data.oMf[self.wrist_frame_id[side]]
        return np.asarray(M.translation, dtype=np.float64), _quat_wxyz_from_R(np.asarray(M.rotation))

    def fk_frame(self, q_full: np.ndarray, frame_name: str) -> np.ndarray:
        """Return base-frame position (3,) of an arbitrary link frame (e.g. a fingertip)."""
        q = np.asarray(q_full, dtype=np.float64).copy()
        fid = self.model.getFrameId(frame_name)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return np.asarray(self.data.oMf[fid].translation, dtype=np.float64)

    def fk_frame_pose(self, q_full: np.ndarray, frame_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos(3), R(3,3)) of a link frame in the base (pelvis) frame at config q_full."""
        q = np.asarray(q_full, dtype=np.float64).copy()
        fid = self.model.getFrameId(frame_name)
        assert fid < self.model.nframes, f"missing frame {frame_name}"
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        M = self.data.oMf[fid]
        return np.asarray(M.translation, dtype=np.float64), np.asarray(M.rotation, dtype=np.float64)

    # -- inverse kinematics (single arm, damped least squares) -----------------
    def ik_arm(
        self,
        side: str,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        q_init_full: np.ndarray,
        iters: int = 100,
        damping: float = 1e-4,
        tol: float = 1e-4,
        max_step: float = 0.3,
    ) -> tuple[np.ndarray, bool, float]:
        """Solve the 7 arm joints of `side` so its wrist frame reaches the target pose.

        Only the arm joints move; all other joints stay at q_init_full. Returns
        (q_arm(7) in _ARM_JOINTS order, converged, final_pos+rot error norm).
        """
        q = np.asarray(q_init_full, dtype=np.float64).copy()
        idx_q = self.arm_idx_q[side]
        idx_v = self.arm_idx_v[side]
        fid = self.wrist_frame_id[side]
        oMdes = pin.SE3(_R_from_quat_wxyz(np.asarray(target_quat_wxyz, float)),
                        np.asarray(target_pos, float))

        err_norm = np.inf
        for _ in range(iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[fid]
            iMd = oMf.actInv(oMdes)               # desired pose in current frame
            err = pin.log(iMd).vector             # 6D spatial error (local frame)
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                return q[idx_q].copy(), True, err_norm
            J = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL)
            Ja = J[:, idx_v]                       # 6 x 7
            JJt = Ja @ Ja.T + damping * np.eye(6)
            dq_arm = Ja.T @ np.linalg.solve(JJt, err)
            dq_arm = np.clip(dq_arm, -max_step, max_step)
            # integrate only the arm joints
            dq_full = np.zeros(self.model.nv)
            dq_full[idx_v] = dq_arm
            q = pin.integrate(self.model, q, dq_full)
            q[idx_q] = np.clip(q[idx_q], self.q_lower[idx_q], self.q_upper[idx_q])
        return q[idx_q].copy(), err_norm < tol, err_norm

    def set_arm(self, q_full: np.ndarray, side: str, q_arm7: np.ndarray) -> np.ndarray:
        q = np.asarray(q_full, dtype=np.float64).copy()
        q[self.arm_idx_q[side]] = np.asarray(q_arm7, dtype=np.float64)
        return q


if __name__ == "__main__":
    # FK -> IK round-trip self test: perturb arms, FK to get a reachable target,
    # then IK from a different seed and check it recovers the wrist pose.
    kin = G1Kinematics()
    rng = np.random.default_rng(0)
    q0 = pin.neutral(kin.model)
    n_ok = 0
    N = 20
    for t in range(N):
        q_target = q0.copy()
        for side in ("left", "right"):
            lo, hi = kin.q_lower[kin.arm_idx_q[side]], kin.q_upper[kin.arm_idx_q[side]]
            lo = np.where(np.isfinite(lo), lo, -1.5)
            hi = np.where(np.isfinite(hi), hi, 1.5)
            q_target[kin.arm_idx_q[side]] = rng.uniform(0.4 * lo, 0.4 * hi)
        ok_both = True
        for side in ("left", "right"):
            pos, quat = kin.fk_wrist(q_target, side)
            q_arm, conv, err = kin.ik_arm(side, pos, quat, q0, iters=200)
            q_sol = kin.set_arm(q0, side, q_arm)
            pos2, quat2 = kin.fk_wrist(q_sol, side)
            perr = np.linalg.norm(pos - pos2)
            ok_both = ok_both and (perr < 1e-3)
        n_ok += int(ok_both)
    print(f"FK-IK round trip: {n_ok}/{N} within 1mm")
