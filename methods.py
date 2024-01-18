# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Methods:
  * combine one Network and one Uncertainty
  * receive an additional float parameter alpha
  * need training on a in-domain validation loader
  * .forward(x) accepts [batch_size, num_channels, height, width]
  * .forward(x) returns softmax [batch_size, num_classes + 1]
"""

import torch
import utils


class Method(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def fit(self, logits, targets, uncertainties):
        raise NotImplementedError

    def forward(self, logits, uncertainties):
        raise NotImplementedError


class Filter(Method):
    def __init__(self, alpha=0.05):
        super().__init__(alpha)

    def fit(self, logits, targets, uncertainties, num_iterations=None):
        self.thres_unc = uncertainties.quantile(1 - self.alpha).item()
        self.cal_net = torch.nn.Linear(1, 1, bias=False)
        self.cal_net.weight.data.fill_(1)
        self.ood_acc = 1

    def forward(self, logits, uncertainties):
        preds = logits.softmax(1)
        preds = torch.cat((preds, torch.zeros(len(preds), 1)), -1)
        p_ood = torch.zeros_like(preds)
        p_ood[:, -1] = 1
        is_id = uncertainties.lt(self.thres_unc)
        return preds * is_id + p_ood * (~is_id)


class Dual(Method):
    def __init__(self, alpha=0.05, variant="mlp"):
        super().__init__(alpha)

        self.variant = variant
        if variant == "linear":
            self.cal_net = torch.nn.Linear(1, 1)
        elif variant == "mlp":
            self.cal_net = torch.nn.Sequential(
                torch.nn.Linear(1, 64),
                torch.nn.ReLU(),
                # torch.nn.Linear(64, 64),
                # torch.nn.ReLU(),
                torch.nn.Linear(64, 1))

        if variant == "linear" or variant == "mlp":
            self.cal_opt = torch.optim.Adam(
                self.cal_net.parameters(), lr=1e-3, weight_decay=1e-4)
            self.cal_obj = torch.nn.CrossEntropyLoss()

    def fit(self, logits, targets, uncertainties, num_iterations=1000):
        if self.variant == "vim":
            norm = (logits.max(1).values.sum() / uncertainties.sum()).item()
            self.cal_net = lambda u: u * norm
            return

        self.thres_unc = uncertainties.quantile(1 - self.alpha).item()
        idx_out = uncertainties.view(-1) > self.thres_unc

        extended_targets = targets.clone()
        extended_targets[idx_out] = max(targets) + 1

        # inputs = torch.cat((logits.max(1, keepdim=True).values, uncertainties), -1)
        inputs = uncertainties

        for iteration in range(num_iterations):
            preds = torch.cat((logits, self.cal_net(inputs)), -1)
            loss_value = self.cal_obj(preds, extended_targets)
            self.cal_opt.zero_grad()
            loss_value.backward()
            self.cal_opt.step()

        self.ood_acc = preds.argmax(1).eq(max(targets) + 1).eq(
            extended_targets.eq(max(targets) + 1)).float().mean().item()

    def forward(self, logits, uncertainties):
        # inputs = torch.cat((logits.max(1, keepdim=True).values, uncertainties), -1)
        inputs = uncertainties
        return torch.cat((
            logits, self.cal_net(inputs)), -1).softmax(dim=1)


def get(name):
    if name == "names":
        return ["Filter", "Dual", "Dual-Linear"] # , "Dual-ViM"]
    elif name == "Filter":
        return Filter()
    elif name == "Dual":
        return Dual(variant="mlp")
    elif name == "Dual-Linear":
        return Dual(variant="linear")
    elif name == "Dual-ViM":
        return Dual(variant="vim")
    else:
        raise NotImplementedError
