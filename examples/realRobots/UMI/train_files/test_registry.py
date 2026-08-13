"""Fast import check for the external UMI registry."""
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP

EXPECTED_MIXTURES = 24
EXPECTED_ROBOT_TYPES = 13
umi_mixtures = {k: v for k, v in DATASET_NAMED_MIXTURES.items() if k.startswith("umi_")}
umi_robot_types = {k: v for k, v in ROBOT_TYPE_CONFIG_MAP.items() if k.startswith("umi_")}
assert len(umi_mixtures) == EXPECTED_MIXTURES, (len(umi_mixtures), sorted(umi_mixtures))
assert len(umi_robot_types) == EXPECTED_ROBOT_TYPES, (len(umi_robot_types), sorted(umi_robot_types))
print({"umi_mixtures": len(umi_mixtures), "umi_robot_types": len(umi_robot_types)})
