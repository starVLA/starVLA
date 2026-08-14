"""UMI policy dataset registry for StarVLA.

Kept outside the core package so upstream StarVLA can be updated without
reapplying patches to mixtures.py, data_config.py, or embodiment_tags.py.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class UMIDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    def __init__(self, video_keys, state_keys, action_keys, *, action_semantics, language_keys=None):
        self.video_keys = video_keys
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.action_semantics = action_semantics
        if language_keys is not None:
            self.language_keys = language_keys

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={k: "q99" for k in self.state_keys}),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={k: "q99" for k in self.action_keys}),
        ])


def cfg(video, state, action, *, semantics, language=None):
    return UMIDataConfig(video, state, action, action_semantics=semantics, language_keys=language)


ROBOT_TYPE_CONFIG_MAP = {
    "umi_openeai_7d": cfg(
        ["video.camera_left"],
        ["state.eef_position", "state.eef_rotation", "state.gripper"],
        ["action.eef_position", "action.eef_rotation", "action.gripper"],
        semantics="absolute_eef",
    ),
    "umi_mv_delta_7d": cfg(
        ["video.camera0", "video.camera1"],
        ["state.eef_position", "state.eef_rotation", "state.gripper_width"],
        ["action.delta_eef_position", "action.delta_eef_rotation", "action.gripper_width"],
        semantics="delta_eef",
        language=["annotation.language.language_instruction"],
    ),
    "umi_fast_dual_delta_14d": cfg(
        ["video.left_camera", "video.right_camera"],
        ["state.left_position", "state.left_rotation", "state.left_gripper", "state.right_position", "state.right_rotation", "state.right_gripper"],
        ["action.left_delta_position", "action.left_delta_rotation", "action.left_gripper", "action.right_delta_position", "action.right_delta_rotation", "action.right_gripper"],
        semantics="dual_delta_eef",
    ),
    "umi_vista_dual_abs_16d": cfg(
        ["video.robot_0", "video.robot_1"],
        ["state.robot_0_position", "state.robot_0_quaternion", "state.robot_0_gripper", "state.robot_1_position", "state.robot_1_quaternion", "state.robot_1_gripper"],
        ["action.robot_0_position", "action.robot_0_quaternion", "action.robot_0_gripper", "action.robot_1_position", "action.robot_1_quaternion", "action.robot_1_gripper"],
        semantics="dual_absolute_eef",
    ),
    "umi_original_dual_abs_14d": cfg(
        ["video.camera0", "video.camera1"],
        ["state.robot0_position", "state.robot0_rotation", "state.robot0_gripper", "state.robot1_position", "state.robot1_rotation", "state.robot1_gripper"],
        ["action.robot0_position", "action.robot0_rotation", "action.robot0_gripper", "action.robot1_position", "action.robot1_rotation", "action.robot1_gripper"],
        semantics="dual_absolute_eef",
    ),
    "umi_nonhuman_joint_6d": cfg(["video.cam_0", "video.cam_1", "video.cam_2"], ["state.robot_state"], ["action.robot_action"], semantics="joint"),
    "umi_dexumi_hand_6d": cfg(["video.camera_left"], ["state.hand_pose"], ["action.hand_action"], semantics="dexterous_hand"),
    "umi_manipforce_native_8d": cfg(["video.handeye_cam_1", "video.handeye_cam_2"], ["state.robot_state"], ["action.robot_action"], semantics="native_manipforce"),
    "umi_hifi_native_20d": cfg(["video.head_main", "video.head_main_stereo_right", "video.left_hand_down", "video.left_hand_up", "video.right_hand_down", "video.right_hand_up"], ["state.robot_state"], ["action.robot_action"], semantics="native_hifi"),
    "umi_fastdata_native_7d": cfg(["video.camera_left"], ["state.robot_state"], ["action.robot_action"], semantics="absolute_eef"),
    "umi_tamen_joint_16d": cfg(["video.camera1", "video.camera2", "video.camera3", "video.camera4"], ["state.robot_state"], ["action.robot_action"], semantics="joint"),
    "umi_genrobot_dual_abs_16d": cfg(["video.robot_0", "video.robot_1"], ["state.robot_state"], ["action.robot_action"], semantics="dual_absolute_eef"),
    "umi_dexwild_abs_23d": cfg(["video.pinky", "video.thumb"], ["state.robot_state"], ["action.robot_action"], semantics="dexterous_hand"),
}

DATASET_NAMED_MIXTURES = {
    "umi_openeai_400": [("OpenEAI-UMI", 1.0, "umi_openeai_7d")],
    "umi_mv_400": [("MV-UMI-400", 1.0, "umi_mv_delta_7d")],
    "umi_fast_400": [("FastUMI-100K-400", 1.0, "umi_fast_dual_delta_14d")],
    "umi_vista_400": [("VISTA-task-003", 1.0, "umi_vista_dual_abs_16d"), ("VISTA-task-010", 1.0, "umi_vista_dual_abs_16d"), ("VISTA-task-001", 0.41, "umi_vista_dual_abs_16d")],
    "umi_original_258": [("Original-UMI-258", 1.0, "umi_original_dual_abs_14d")],
    "umi_maniwav_400": [("ManiWAV-400", 1.0, "umi_openeai_7d")],
    "umi_exumi_400": [("exUMI-400", 1.0, "umi_openeai_7d")],
    "umi_vitamin_400": [("ViTaMIn-400", 1.0, "umi_openeai_7d")],
    "umi_tacthru_381": [("TacThru-UMI-381", 1.0, "umi_openeai_7d")],
    "umi_scaling_400": [("Data-Scaling-Laws-400", 1.0, "umi_openeai_7d")],
    "umi_nonhuman_1": [("NONHUMAN-DexUMI-1", 1.0, "umi_nonhuman_joint_6d")],
    "umi_vitamin_b_400": [("ViTaMIn-B-400", 1.0, "umi_openeai_7d")],
    "umi_touch_wild_186": [("Touch-in-the-Wild-186", 1.0, "umi_openeai_7d")],
    "umi_manipforce_101": [("ManipForce-Gear-101", 1.0, "umi_manipforce_native_8d")],
    "umi_dexumi_400": [("DexUMI-400", 1.0, "umi_dexumi_hand_6d")],
    "umi_original_dynamic_284": [("Original-UMI-Dynamic-Tossing-284", 1.0, "umi_openeai_7d")],
    "umi_hifi_400": [("HiFi-UMI-400", 1.0, "umi_hifi_native_20d")],
    "umi_fastdata_400": [("FastUMI-Data-400", 1.0, "umi_fastdata_native_7d")],
    "umi_tamen_400": [("TAMEn-400", 1.0, "umi_tamen_joint_16d")],
    "umi_3d_400": [("UMI-3D-400", 1.0, "umi_openeai_7d")],
    "umi_genrobot_400": [("GenRobot-400", 1.0, "umi_genrobot_dual_abs_16d")],
    "umi_dexwild_400": [("DexWild-400", 1.0, "umi_dexwild_abs_23d")],
    "umi_abs7_balanced": [("OpenEAI-UMI", 1.0, "umi_openeai_7d"), ("ManiWAV-400", 1.0, "umi_openeai_7d"), ("exUMI-400", 1.0, "umi_openeai_7d")],
    "umi_abs7_native_heavy": [("OpenEAI-UMI", 3.0, "umi_openeai_7d"), ("ManiWAV-400", 1.0, "umi_openeai_7d"), ("exUMI-400", 1.0, "umi_openeai_7d")],
}
