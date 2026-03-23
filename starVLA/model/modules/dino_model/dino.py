"""
DINOv2 vision backbone wrapper.

Features:
  - Loads DINOv2 variants via torch.hub (with local fallback)
  - Exposes patch token features (x_norm_patchtokens)
  - Provides preprocessing (resize + normalization) for multi-view PIL images
  - Parallel per-view preprocessing using ThreadPoolExecutor
"""

import os
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn
from torchvision import transforms


def apply_transform(view, transform):
    return transform(view)


# from llavavla.model.modules.dino_model.dino_transforms import make_classification_train_transform


class DINOv2BackBone(nn.Module):
    """
    Thin wrapper around a DINOv2 model.

    Args:
        backone_name: DINOv2 model id (e.g. dinov2_vits14, dinov2_vitb14).
        output_channels: (Unused placeholder; retained for future extension).

    Attributes:
        body: Loaded DINOv2 model.
        num_channels: Feature dimension of patch tokens.
        dino_transform: Preprocessing pipeline (resize + tensor + normalize).
    """

    def __init__(self, backone_name="dinov2_vits14", output_channels=1024) -> None:
        super().__init__()
        try:
            self.body = torch.hub.load("facebookresearch/dinov2", backone_name)
        except:
            import traceback

            traceback.print_exc()
            print("Failed to load dinov2 from torch hub, loading from local")
            TORCH_HOME = os.environ.get("TORCH_HOME", "~/.cache/torch/")
            weights_path = os.path.expanduser(f"{TORCH_HOME}/hub/checkpoints/{backone_name}_pretrain.pth")

            code_path = os.path.expanduser(f"{TORCH_HOME}/hub/facebookresearch_dinov2_main")

            self.body = torch.hub.load(code_path, backone_name, source="local", pretrained=False)

            state_dict = torch.load(weights_path)
            self.body.load_state_dict(state_dict)
        if backone_name == "dinov2_vits14":
            self.num_channels = 384
        elif backone_name == "dinov2_vitb14":
            self.num_channels = 768
        elif backone_name == "dinov2_vitl14":
            self.num_channels = 1024
        elif backone_name == "dinov2_vitg14":
            self.num_channels = 1408
        else:
            raise NotImplementedError(f"DINOv2 backbone {backone_name} not implemented")
        self.dino_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # self.dino_transform = make_classification_train_transform()

    # @torch.no_grad()
    def forward(self, tensor):
        """
        Forward pass.

        Args:
            tensor: Image batch tensor [B*views, 3, H, W].

        Returns:
            torch.Tensor: Patch token features [B*views, N_tokens, C].
        """
        xs = self.body.forward_features(tensor)["x_norm_patchtokens"]

        return xs  # B*views, token, dim

    def prepare_dino_input(self, img_list):
        """
        Preprocess a batch of multi-view PIL image lists into a tensor suitable for DINO.

        Args:
            img_list: List of samples; each sample is List[PIL.Image] (multi-view).

        Returns:
            torch.Tensor: Flattened batch of shape [B * num_view, 3, H, W] on model device.
        """
        # img_list: is a list of [PIL], each representing multi views of the same example.
        # refer to https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py

        # use thread pool to parallel process each view
        with ThreadPoolExecutor() as executor:
            image_tensors = torch.stack(
                [
                    torch.stack(list(executor.map(lambda view: apply_transform(view, self.dino_transform), views)))
                    for views in img_list
                ]
            )

        # move the tensor to the device of DINO encoder
        B, num_view, C, H, W = image_tensors.shape
        image_tensors = image_tensors.view(B * num_view, C, H, W)
        device = next(self.parameters()).device
        image_tensors = image_tensors.to(device)

        return image_tensors


def get_dino_model(backone_name="dinov2_vits14") -> DINOv2BackBone:
    """
    Factory helper returning a configured DINOv2BackBone.

    Args:
        backone_name: DINOv2 variant name.

    Returns:
        DINOv2BackBone: Initialized backbone instance.
    """
    return DINOv2BackBone(backone_name)


if __name__ == "__main__":
    dino = DINOv2BackBone()
    pass
