# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Networks:
  * .features(x) accepts [batch_size, num_channels, height, width]
  * .features(x) returns [batch_size, num_features]
  * .forward(x) accepts [batch_size, num_channels, height, width]
  * .forward(x) returns [batch_size, num_classes]
"""

import torch
import torchvision
import copy


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def classify(self, f):
        return self.classifier(f).cpu()

    def featurize(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return self.featurizer(x)

    def forward(self, x):
        return self.classify(self.featurize(x))

    def flt(self, loader):
        features, logits, targets = [], [], []

        with torch.no_grad():
            for x, y in loader:
                f = self.featurize(x)
                l = self.classify(f)
                features.append(f.cpu())
                logits.append(l.cpu())
                targets.append(y)

        features = torch.cat(features)
        logits = torch.cat(logits)
        targets = torch.cat(targets)

        return features, logits, targets


class ResNet(Network):
    def __init__(self, ResNetConstructor, version=1):
        super().__init__()
        self.featurizer = ResNetConstructor(
            weights="IMAGENET1K_V1" if version == 1 else "IMAGENET1K_V2")
        self.classifier = copy.deepcopy(self.featurizer.fc)
        self.featurizer.fc = torch.nn.Identity()
        if torch.cuda.is_available():
            self.cuda()
        self.eval()


class ResNet50(ResNet):
    def __init__(self, version=1):
        super().__init__(torchvision.models.resnet50, version)


class ResNet152(ResNet):
    def __init__(self, version=1):
        super().__init__(torchvision.models.resnet152, version)


class VisionTransformer(Network):
    def __init__(self, VisionConstructor):
        super().__init__()
        self.transformer = VisionConstructor(weights="IMAGENET1K_V1")
        self.classifier = copy.deepcopy(self.transformer.heads)
        self.transformer.heads = torch.nn.Identity()
        if torch.cuda.is_available():
            self.cuda()
        self.eval()

    def featurizer(self, x):
        # From: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L289-L305
        x = self.transformer._process_input(x)
        n = x.shape[0]
        batch_class_token = self.transformer.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.transformer.encoder(x)
        return x[:, 0]


class ViTB16(VisionTransformer):
    def __init__(self):
        super().__init__(torchvision.models.vit_b_16)


class ViTB32(VisionTransformer):
    def __init__(self):
        super().__init__(torchvision.models.vit_b_32)


def get(name):
    if name == "names":
        return ["ResNet50-v1", "ResNet152-v1", "ResNet50-v2", "ResNet152-v2", "ViT-B-16", "ViT-B-32"]
    elif name == "ResNet50-v1":
        return ResNet50(version=1)
    elif name == "ResNet152-v1":
        return ResNet152(version=1)
    elif name == "ResNet50-v2":
        return ResNet50(version=2)
    elif name == "ResNet152-v2":
        return ResNet152(version=2)
    elif name == "ViT-B-16":
        return ViTB16()
    elif name == "ViT-B-32":
        return ViTB32()
    else:
        raise NotImplementedError
