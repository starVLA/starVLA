#!/usr/bin/env python3
"""Install MiniCPM-RobotManip's modality mapping into four LIBERO suites."""

import argparse
import json
import shutil
from pathlib import Path

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()

    template = Path(__file__).with_name("modality.json")
    json.loads(template.read_text())

    for suite in SUITES:
        matches = sorted(path for path in args.dataset_root.glob(f"{suite}_*") if path.is_dir())
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {suite!r} directory under {args.dataset_root}, "
                f"found {[path.name for path in matches]}"
            )
        info_path = matches[0] / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        features = json.loads(info_path.read_text()).get("features", {})
        required = {
            "observation.images.image",
            "observation.images.wrist_image",
            "observation.xvla_abs_ee6d",
            "task_index",
        }
        missing = sorted(required - features.keys())
        if missing:
            raise ValueError(f"{matches[0].name} is missing required features: {missing}")

        destination = matches[0] / "meta" / "modality.json"
        shutil.copyfile(template, destination)
        print(f"installed {destination}")


if __name__ == "__main__":
    main()
