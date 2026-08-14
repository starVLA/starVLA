import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from types import SimpleNamespace

from starVLA.dataloader.umi_datasets import UMISampleAdapter, UMISamplePolicy, umi_collate_fn


class FakeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def sample(action):
    return {
        "action": action,
        "state": np.ones((1, 7), dtype=np.float32),
        "image": [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
        "lang": "  pick   object  ",
        "robot_tag": "new_embodiment",
    }


def main():
    inferred = UMISamplePolicy.from_config(
        {}, SimpleNamespace(action_horizon=8, action_dim=7, state_dim=7)
    )
    assert inferred == UMISamplePolicy(action_horizon=8, action_dim=7, state_dim=7)
    stateless = UMISamplePolicy.from_config(
        {"include_state": False}, SimpleNamespace(action_horizon=8, action_dim=7, state_dim=7)
    )
    assert stateless.state_dim is None
    try:
        UMISamplePolicy.from_config(
            {"action_dim": 6}, SimpleNamespace(action_horizon=8, action_dim=7)
        )
    except ValueError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("conflicting action dimensions were accepted")

    policy = UMISamplePolicy(action_horizon=8, action_dim=7, state_dim=7)
    adapter = UMISampleAdapter(FakeDataset([sample(np.ones((8, 7)))]), policy)
    item = adapter[0]
    assert item["action"].shape == (8, 7)
    assert item["action_mask"].all()
    assert item["state"].shape == (1, 7)
    assert item["lang"] == "pick object"
    assert len(umi_collate_fn([item])) == 1

    padded = UMISampleAdapter(
        FakeDataset([sample(np.ones((3, 5)))]),
        UMISamplePolicy(action_horizon=8, action_dim=7, state_dim=7, strict_dimensions=False),
    )[0]
    assert padded["action"].shape == (8, 7)
    assert int(padded["action_mask"].sum()) == 15

    recovered = UMISampleAdapter(
        FakeDataset([sample(np.full((8, 7), np.nan)), sample(np.ones((8, 7)))]),
        policy,
        seed=42,
    )[0]
    assert np.isfinite(recovered["action"]).all()

    try:
        UMISampleAdapter(FakeDataset([sample(np.ones((7, 7)))]), policy)[0]
    except RuntimeError as error:
        assert isinstance(error.__cause__, ValueError)
        assert "shape" in str(error.__cause__)
    else:
        raise AssertionError("strict mode accepted the wrong action horizon")
    print("UMI dataloader tests passed")


if __name__ == "__main__":
    main()
