from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch.nn as nn

from .attention import CBAM, SEBlock


def _tv_resnet(weights, arch: str) -> nn.Module:
    import torchvision.models as M

    if arch == "resnet18":
        return M.resnet18(weights=weights)
    if arch == "resnet34":
        return M.resnet34(weights=weights)
    raise ValueError(f"Unknown arch: {arch}")


def _replace_fc(model: nn.Module, num_classes: int) -> None:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)


class ResNetWithStageAttention(nn.Module):
    def __init__(self, base: nn.Module, attention: Literal["se", "cbam"]) -> None:
        super().__init__()
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc

        if attention == "se":
            self.att1 = SEBlock(64)
            self.att2 = SEBlock(128)
            self.att3 = SEBlock(256)
            self.att4 = SEBlock(512)
        elif attention == "cbam":
            self.att1 = CBAM(64)
            self.att2 = CBAM(128)
            self.att3 = CBAM(256)
            self.att4 = CBAM(512)
        else:
            raise ValueError(f"Unknown attention: {attention}")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.att1(x)
        x = self.layer2(x)
        x = self.att2(x)
        x = self.layer3(x)
        x = self.att3(x)
        x = self.layer4(x)
        x = self.att4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


@dataclass(frozen=True)
class ModelConfig:
    model: Literal[
        "resnet18",
        "resnet34",
        "resnet18_se",
        "resnet34_se",
        "resnet18_cbam",
        "resnet34_cbam",
    ] = "resnet18"
    num_classes: int = 102
    pretrained: bool = True

    def build(self) -> nn.Module:
        import torchvision.models as M

        base_arch = "resnet18" if "resnet18" in self.model else "resnet34"
        if self.pretrained:
            weights = M.ResNet18_Weights.IMAGENET1K_V1 if base_arch == "resnet18" else M.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        base = _tv_resnet(weights, base_arch)
        _replace_fc(base, self.num_classes)

        if self.model.endswith("_se"):
            return ResNetWithStageAttention(base, attention="se")
        if self.model.endswith("_cbam"):
            return ResNetWithStageAttention(base, attention="cbam")
        return base

    @staticmethod
    def split_params_for_finetune(model: nn.Module):
        head = []
        backbone = []
        for name, p in model.named_parameters():
            if "fc." in name:
                head.append(p)
            else:
                backbone.append(p)
        return backbone, head

