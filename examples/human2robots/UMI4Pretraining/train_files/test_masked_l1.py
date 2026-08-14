import torch

from starVLA.model.framework.VLM4A.QwenOFT import masked_l1_loss


def main():
    prediction = torch.tensor([[[1.0, 100.0], [3.0, 100.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[True, False], [True, False]]])
    assert torch.isclose(masked_l1_loss(prediction, target, mask), torch.tensor(2.0))
    assert torch.isclose(masked_l1_loss(prediction, target), torch.tensor(51.0))
    try:
        masked_l1_loss(prediction, target, torch.zeros_like(mask))
    except ValueError as error:
        assert "no valid" in str(error)
    else:
        raise AssertionError("empty action mask was accepted")
    print("masked L1 tests passed")


if __name__ == "__main__":
    main()
