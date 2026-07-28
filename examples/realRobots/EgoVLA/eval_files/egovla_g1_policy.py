"""EgoVLAG1Policy — serve the starVLA EgoVLA model behind the GR00T get_action contract.

Pipeline (per get_action call):
  input:   bridge G1 observation (ego_view video, state{wrist_eef_9d, arms, ...},
           language)  ->  starVLA EgoVLA `example` (image, lang, camera-frame proprio)
  forward: EgoVLA framework `predict_action`  ->  48-dim chunk (T, 48)
  decode:  EgoVLADecoder  ->  pelvis-frame wrist EE poses + Inspire-12 hand qpos
  output:  EE poses -> G1 arm 7-DoF joints (pinocchio IK, warm-started from state)
           Inspire-12 -> Dex3-7 (per-finger, preserves articulation so a grasp is
           physically reachable); base_height/navigate -> neutral
           -> GR00T action dict the GR00T-WBC-Bridge consumes unchanged.

The observation/action contract mirrors the bridge's N1.7 / REAL_G1 profile.
"""
from __future__ import annotations

import os

import numpy as np
import torch


def _rot6d_to_R(rot6d: np.ndarray) -> np.ndarray:
    """6D (base-frame wrist_eef) -> 3x3 (Gram-Schmidt, column-major like the eef_9d)."""
    a1, a2 = rot6d[:3], rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2p = a2 - np.dot(b1, a2) * b1
    b2 = a2p / (np.linalg.norm(a2p) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _quat_wxyz_from_R(Rm: np.ndarray) -> np.ndarray:
    import pinocchio as pin
    q = pin.Quaternion(Rm)
    return np.array([q.w, q.x, q.y, q.z])


class EgoVLAG1Policy:
    ACTION_HORIZON = 30

    # bridge N1.7 / REAL_G1 contract
    _STATE_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist",
                   "left_wrist_eef_9d", "right_wrist_eef_9d"]
    _ACTION_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand",
                    "base_height_command", "navigate_command"]

    def __init__(self, ckpt_dir: str | None = None, device: str = "cuda",
                 fallback_instruction: str = "place the red ball in the box"):
        self.device = device
        self._instruction = fallback_instruction
        egovla = os.environ.get("EGOVLA_RELEASE", "/home/dhy/Projects/EgoVLA_Release")
        ckpt_dir = ckpt_dir or os.path.join(egovla, "checkpoints/ego_vla_checkpoint/ckpt-6720")

        # --- G1 kinematics (arm IK + camera FK) ---
        from .g1_kinematics import G1Kinematics
        self.kin = G1Kinematics()

        # --- starVLA EgoVLA framework (inference) ---
        from omegaconf import OmegaConf
        from starVLA.model.framework.base_framework import build_framework
        cfg = OmegaConf.create({
            "framework": {
                "name": "EgoVLA",
                "qwenvl": {"base_vlm": ckpt_dir, "attn_implementation": "sdpa", "use_bf16": True},
                "action_model": {
                    "action_model_type": "EgoVLATrajDecoder", "action_dim": 48,
                    "action_hidden_dim": 1536, "action_horizon": self.ACTION_HORIZON,
                    "proprio_size": 16, "use_proprio": True, "sep_proprio": True,
                    "traj_decoder_type": "transformer_split_action_v2",
                },
                "input_placeholder_end_token_idx": 151195,
                "input_placeholder_start_token_idx": 151375,
                "action_query_token_id": 151300,
            },
            "datasets": {"vla_data": {"obs_image_size": [384, 384]}},
        })
        self.model = build_framework(cfg).to(device).eval()

        # --- 48-dim decoder (EE poses + Inspire hand) ---
        from .decode import EgoVLADecoder
        self.decoder = EgoVLADecoder(self.kin, device=device)

        # camera transform for base->camera proprio (inverse of the decode transform)
        self._T_cam_pelvis = np.linalg.inv(self.decoder.T_po)

    # ---- GR00T contract ------------------------------------------------------
    def reset(self, options=None):
        return {"status": "ok"}

    def get_modality_config(self):
        return {
            "video_keys": ["ego_view"],
            "state_keys": self._STATE_KEYS,
            "action_keys": self._ACTION_KEYS,
            "action_horizon": self.ACTION_HORIZON,
        }

    def get_action(self, observation: dict, options=None):
        example = self._obs_to_example(observation)
        pred = self.model.predict_action([example])["normalized_actions"][0]  # (T,48)
        ego = self.decoder.decode(pred)
        action = self._to_g1_action(ego, observation)
        return action, {}

    # ---- input: bridge obs -> EgoVLA example ---------------------------------
    def _obs_to_example(self, obs: dict) -> dict:
        import cv2
        from PIL import Image
        video = np.asarray(obs["video"]["ego_view"])          # (1,T,H,W,3) uint8
        frame = video[0, -1]
        img = Image.fromarray(cv2.resize(frame, (384, 384)))

        # camera-frame wrist proprio from base-frame wrist_eef_9d
        def cam_wrist(key):
            eef9 = np.asarray(obs["state"][key]).reshape(-1)[:9]
            pos_b, R_b = eef9[:3], _rot6d_to_R(eef9[3:9])
            pos_c = self._T_cam_pelvis[:3, :3] @ pos_b + self._T_cam_pelvis[:3, 3]
            R_c = self._T_cam_pelvis[:3, :3] @ R_b
            from scipy.spatial.transform import Rotation as Rot
            return pos_c, Rot.from_matrix(R_c).as_rotvec()
        try:
            lp, lr = cam_wrist("left_wrist_eef_9d")
            rp, rr = cam_wrist("right_wrist_eef_9d")
            proprio_3d = np.concatenate([lp, rp]).astype(np.float32)     # (6,)
            proprio_rot = np.concatenate([lr, rr]).astype(np.float32)    # (6,)
        except Exception:
            proprio_3d = np.zeros(6, np.float32)
            proprio_rot = np.zeros(6, np.float32)

        return {
            "image": [img],
            "lang": self._instruction_from_obs(obs),
            "proprio_3d": proprio_3d,
            "proprio_rot": proprio_rot,
            "proprio_hand_finger_tip": np.zeros(30, np.float32),  # placeholder (input side)
        }

    def _instruction_from_obs(self, obs: dict) -> str:
        try:
            lang = obs["language"]["annotation.human.task_description"]
            s = lang[0][0] if isinstance(lang[0], (list, tuple)) else lang[0]
            if isinstance(s, bytes):
                s = s.decode()
            if s:
                self._instruction = s
        except Exception:
            pass
        return self._instruction

    # ---- output: EgoVLA decode -> GR00T G1 action dict -----------------------
    def _to_g1_action(self, ego: dict, obs: dict) -> dict:
        left_ee, right_ee = ego["left_ee_pose"], ego["right_ee_pose"]       # (T,7)
        left12, right12 = ego["left_inspire12"], ego["right_inspire12"]     # (T,12)
        T = left_ee.shape[0]

        q_seed = self._full_q_from_obs(obs)
        seed_l, seed_r = q_seed.copy(), q_seed.copy()
        la = np.zeros((T, 7), np.float32)
        ra = np.zeros((T, 7), np.float32)
        for t in range(T):
            qa_l, _, _ = self.kin.ik_arm("left", left_ee[t, :3], left_ee[t, 3:7], seed_l, iters=60)
            qa_r, _, _ = self.kin.ik_arm("right", right_ee[t, :3], right_ee[t, 3:7], seed_r, iters=60)
            la[t], ra[t] = qa_l, qa_r
            seed_l = self.kin.set_arm(seed_l, "left", qa_l)
            seed_r = self.kin.set_arm(seed_r, "right", qa_r)

        lh = np.stack([self._inspire12_to_dex7(left12[t]) for t in range(T)]).astype(np.float32)
        rh = np.stack([self._inspire12_to_dex7(right12[t]) for t in range(T)]).astype(np.float32)

        b = lambda x: x[None, ...].astype(np.float32)  # noqa: E731  (1,T,D)
        return {
            "left_arm": b(la), "right_arm": b(ra),
            "left_hand": b(lh), "right_hand": b(rh),
            "base_height_command": b(np.zeros((T, 1), np.float32)),
            "navigate_command": b(np.zeros((T, 3), np.float32)),
        }

    def _full_q_from_obs(self, obs: dict) -> np.ndarray:
        import pinocchio as pin
        q = pin.neutral(self.kin.model)
        try:
            q = self.kin.set_arm(q, "left", np.asarray(obs["state"]["left_arm"]).reshape(-1)[:7])
            q = self.kin.set_arm(q, "right", np.asarray(obs["state"]["right_arm"]).reshape(-1)[:7])
        except Exception:
            pass
        return q

    @staticmethod
    def _inspire12_to_dex7(h12: np.ndarray) -> np.ndarray:
        """EgoVLA Inspire-12 -> Dex3-7 preserving per-finger articulation.

        Dex3-7 = [index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2].
        Assumes EgoVLA's Inspire-12 ~ Isaac URDF-12 order [index_prox, index_inter,
        middle_prox, middle_inter, pinky*, ring*, thumb_yaw, thumb_pitch,
        thumb_inter, thumb_distal] (finger correspondence is a calibration point;
        articulation is preserved either way so a grasp is physically reachable).
        """
        h = np.asarray(h12, dtype=np.float32).reshape(-1)
        return np.array([h[0], h[1], h[2], h[3], h[8], h[9], h[10]], dtype=np.float32)
