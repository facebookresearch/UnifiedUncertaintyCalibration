# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Uncertainties:
  * require one network at construction
  * may need fitting on one in-domain validation loader
  * .forward(x) accepts [batch_size, num_channels, height, width]
  * .forward(x) returns [batch_size, 1], containing uncertainty estimates
"""

import torch
import faiss


class Uncertainty(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, features, logits, targets):
        pass

    def forward(self, features, logits):
        raise NotImplementedError


class MaxLogit(Uncertainty):
    def __init__(self):
        super().__init__()

    def forward(self, features, logits):
        return logits.max(1).values.view(-1, 1).mul(-1)


class Certificates(Uncertainty):
    def __init__(self, k=500):
        super().__init__()
        self.k = k

    def fit(self, features, logits, targets):
        self.mean = features.mean(0, keepdim=True)
        _, _, Vt = torch.linalg.svd(features - self.mean)
        self.certificates = Vt[:, -self.k:]

    def forward(self, features, logits):
        return ((features - self.mean) @ self.certificates).pow(2).mean(
            1, keepdim=True)


class Gaussians(Uncertainty):
    def __init__(self, covariance_type="diagonal"):
        super().__init__()
        self.covariance_type = covariance_type

    def fit(self, features, logits, targets):
        self.global_mean = features.mean(0, keepdim=True)
        center_f = features - self.global_mean
        n, dim = features.size()

        if self.covariance_type == "diagonal":
            self.W = 1. / center_f.std(0, keepdim=True)
            transf_f = center_f * self.W
        else:
            reg = torch.eye(dim) * 1e-6
            self.W = (center_f.t() @ center_f).div(n - 1).add(reg).inverse()
            self.W = torch.linalg.cholesky(self.W)
            transf_f = center_f @ self.W

        self.mu = torch.stack([
            transf_f[targets == t].mean(0) for t in targets.unique()
        ])

    def forward(self, features, logits):
        if self.covariance_type == "diagonal":
            transf_f = (features - self.global_mean) * self.W
        else:
            transf_f = (features - self.global_mean) @ self.W

        return torch.cdist(transf_f, self.mu).min(1).values.view(-1, 1)


class Mahalanobis(Uncertainty):
    # https://arxiv.org/abs/2106.09022
    def __init__(self):
        super().__init__()

    def fit(self, features, logits, targets):
        n, dim = features.size()
        reg = torch.eye(dim) * 1e-6

        # parameters for MD_0
        self.m0 = features.mean(0, keepdim=True)
        center0 = features - self.m0
        self.W0 = (center0.t() @ center0).div(n).add(reg).inverse()
        self.W0 = torch.linalg.cholesky(self.W0)
        self.m0 @= self.W0

        # parameters for MD_k
        self.mk = torch.stack([
            features[targets == t].mean(0) for t in targets.unique()
        ])
        centerk = features - self.mk[targets]
        self.Wk = (centerk.t() @ centerk).div(n).add(reg).inverse()
        self.Wk = torch.linalg.cholesky(self.Wk)
        self.mk @= self.Wk

    def forward(self, features, logits):
        md0 = (features @ self.W0 - self.m0).norm(2, 1, keepdim=True)
        mdk = torch.cdist(features @ self.Wk, self.mk)
        return (mdk - md0).min(1).values.view(-1, 1)


class KNN(Uncertainty):
    # https://arxiv.org/abs/2204.06507
    def __init__(self, k=10):
        super().__init__()
        self.k = k

    def normalize(self, features):
        return features.div(features.norm(2, 1, keepdim=True))

    def fit(self, features, logits, targets):
        self.index = faiss.IndexFlatL2(features.size(1))
        self.index.add(self.normalize(features))

    def forward(self, features, logits):
        dist, _ = self.index.search(self.normalize(features), self.k)
        return torch.Tensor(dist[:, -1]).view(-1, 1)


class ASH(Uncertainty):
    # https://arxiv.org/abs/2209.09858
    # Adapted from: https://github.com/Jingkang50/OpenOOD/blob/main/openood/networks/ash_net.py
    def __init__(self, fc, p=90):
        super().__init__()
        self.fc = fc
        self.p = p

    def ash_b(self, x):
        x = x.view(len(x), -1, 1, 1)
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])

        n = x.shape[1:].numel()
        k = n - int(round(n * self.p / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        fill = s1 / k
        fill = fill.unsqueeze(dim=1).expand(v.shape)
        t.zero_().scatter_(dim=1, index=i, src=fill)
        return x.view(len(x), -1)

    def forward(self, features, logits):
        clipped_features = self.ash_b(features)
        clipped_logits = self.fc(features).detach()
        return -torch.logsumexp(clipped_logits, dim=1, keepdim=True)


def get(name, fc=None):
    if name == "names":
        return ["MaxLogit", "ASH", "Mahalanobis", "KNN"]
    elif name == "MaxLogit":
        return MaxLogit()
    elif name == "Certificates":
        return Certificates()
    elif name == "Gaussian-Diagonal":
        return Gaussians(covariance_type="diagonal")
    elif name == "Gaussian-Full":
        return Gaussians(covariance_type="full")
    elif name == "Mahalanobis":
        return Mahalanobis()
    elif name == "KNN":
        return KNN()
    elif name == "ASH":
        return ASH(fc)
    else:
        raise NotImplementedError
