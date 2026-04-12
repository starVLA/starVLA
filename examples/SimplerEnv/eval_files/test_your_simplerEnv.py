# Optional: print more diagnostics for debugging
# from sapien import disable_renderer
# disable_renderer()  # <-- Uncomment to skip the renderer


import logging
import os

import simpler_env
from simpler_env.utils.env.env_builder import build_maniskill2_env

logging.basicConfig(level=logging.DEBUG)

# Resolve rgb_overlay_path relative to the SimplerEnv package location
_simpler_env_root = os.path.dirname(os.path.dirname(simpler_env.__file__))
_overlay_path = os.path.join(_simpler_env_root, "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png")

env_name = "PutEggplantInBasketScene-v0"

kwargs = {
    "obs_mode": "rgbd",
    "robot": "widowx_sink_camera_setup",
    "sim_freq": 500,
    "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
    "control_freq": 5,
    "max_episode_steps": 120,
    "scene_name": "bridge_table_1_v2",
    "camera_cfgs": {"add_segmentation": True},
    "rgb_overlay_path": _overlay_path,
}

additional_env_build_kwargs = {}

print("🔧 Start building ManiSkill2 env...")
env = build_maniskill2_env(
    env_name,
    **additional_env_build_kwargs,
    **kwargs,
)
print("✅ Env built successfully:", env)

obs = env.reset()
print("📷 First observation keys:", obs.keys() if isinstance(obs, dict) else type(obs))
