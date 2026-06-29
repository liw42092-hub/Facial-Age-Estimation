from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """LayerNorm that remains stable when the rest of the model uses fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(x.dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(width, width * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(width * 4, width)),
                ]
            )
        )
        self.ln_2 = LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_1(x)
        attended = self.attn(normalized, normalized, normalized, need_weights=False)[0]
        x = x + attended
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class FaRLVisualEncoder(nn.Module):
    """OpenAI CLIP ViT-B/16 visual tower used by the official FaRL weights."""

    def __init__(
        self,
        input_resolution: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        output_dim: int = 512,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width**-0.5
        grid_size = input_resolution // patch_size
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(grid_size * grid_size + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def _position_embedding(self, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        position = self.positional_embedding
        source_size = int((position.size(0) - 1) ** 0.5)
        if source_size * source_size != position.size(0) - 1:
            raise RuntimeError("FaRL positional embedding is not a square grid.")
        if (height, width) == (source_size, source_size):
            return position.to(dtype=dtype)

        class_position = position[:1]
        spatial = position[1:].reshape(1, source_size, source_size, -1).permute(0, 3, 1, 2)
        spatial = F.interpolate(spatial.float(), size=(height, width), mode="bicubic", align_corners=False)
        spatial = spatial.permute(0, 2, 3, 1).reshape(height * width, -1)
        return torch.cat((class_position, spatial), dim=0).to(dtype=dtype)

    def forward(self, images: torch.Tensor, return_tokens: bool = False):
        x = self.conv1(images)
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, channels, height * width).permute(0, 2, 1)
        class_token = self.class_embedding.to(x.dtype).view(1, 1, -1).expand(batch_size, 1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.ln_pre(x + self._position_embedding(height, width, x.dtype).unsqueeze(0))
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)

        class_feature = self.ln_post(x[:, 0]) @ self.proj
        if not return_tokens:
            return class_feature
        patch_features = self.ln_post(x[:, 1:]) @ self.proj
        return class_feature, patch_features


class AgeMLP(nn.Module):
    def __init__(self, in_features: int = 512, hidden_features: int = 512, num_ages: int = 101):
        super().__init__()
        self.layers = nn.Sequential(
            LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_features, num_ages),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class FaRLAgeEstimator(nn.Module):
    def __init__(self, num_ages: int = 101):
        super().__init__()
        self.encoder = FaRLVisualEncoder()
        self.vanilla_head = AgeMLP(num_ages=num_ages)
        self.balanced_head = AgeMLP(num_ages=num_ages)
        self.vanilla_local_head = AgeMLP(num_ages=num_ages)
        self.balanced_local_head = AgeMLP(num_ages=num_ages)
        self.num_ages = num_ages

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images).float()

    def encode_with_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        class_features, patch_features = self.encoder(images, return_tokens=True)
        return class_features.float(), patch_features.float()

    def forward(self, images: torch.Tensor, head: str = "vanilla", return_local: bool = False):
        if return_local:
            features, patch_features = self.encode_with_tokens(images)
            return self.logits_from_features(features, head), self.local_logits_from_tokens(patch_features, head)
        features = self.encode(images)
        return self.logits_from_features(features, head)

    def logits_from_features(self, features: torch.Tensor, head: str = "vanilla") -> torch.Tensor:
        if head == "vanilla":
            return self.vanilla_head(features)
        if head == "balanced":
            return self.balanced_head(features)
        raise ValueError(f"Unknown FaRL age head: {head}")

    def local_logits_from_tokens(self, patch_features: torch.Tensor, head: str = "vanilla") -> torch.Tensor:
        if head == "vanilla":
            return self.vanilla_local_head(patch_features)
        if head == "balanced":
            return self.balanced_local_head(patch_features)
        raise ValueError(f"Unknown FaRL local age head: {head}")

    def freeze_encoder(self) -> None:
        self.encoder.requires_grad_(False)

    def unfreeze_last_blocks(self, count: int) -> None:
        self.freeze_encoder()
        if count < 0 or count > len(self.encoder.transformer.resblocks):
            raise ValueError("unfreeze block count must be between 0 and 12")
        if count:
            for block in self.encoder.transformer.resblocks[-count:]:
                block.requires_grad_(True)
            self.encoder.ln_post.requires_grad_(True)
            self.encoder.proj.requires_grad_(True)

    def reset_balanced_head(self) -> None:
        self.balanced_head.load_state_dict(self.vanilla_head.state_dict())
        self.balanced_local_head.load_state_dict(self.vanilla_local_head.state_dict())


def load_farl_visual_weights(model: FaRLAgeEstimator, checkpoint_path: str | Path) -> dict[str, int]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"FaRL weights not found: {checkpoint_path}. Run download_farl_weights.ps1 first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.encoder.state_dict()
    visual_state = {}
    ignored_tensors = 0
    shape_mismatches = []
    for name, value in source_state.items():
        name = name.removeprefix("module.")
        if name.startswith("visual."):
            visual_name = name.removeprefix("visual.")
            if visual_name not in target_state:
                # FaRL also stores masked-image-modeling modules that are not
                # used by the downstream CLIP visual encoder.
                ignored_tensors += 1
                continue
            if target_state[visual_name].shape != value.shape:
                shape_mismatches.append(
                    f"{visual_name}: expected {tuple(target_state[visual_name].shape)}, "
                    f"got {tuple(value.shape)}"
                )
                continue
            visual_state[visual_name] = value

    if not visual_state:
        raise RuntimeError("The checkpoint does not contain FaRL visual.* weights.")
    if shape_mismatches:
        raise RuntimeError("FaRL visual tensor shape mismatch: " + "; ".join(shape_mismatches))
    incompatible = model.encoder.load_state_dict(visual_state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "FaRL visual checkpoint is incompatible. "
            f"Missing={incompatible.missing_keys}"
        )
    return {
        "loaded_tensors": len(visual_state),
        "ignored_pretraining_tensors": ignored_tensors,
        "source_tensors": len(source_state),
    }
