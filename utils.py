# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.functional.classification import multiclass_calibration_error
import torch


def metrics(preds, targets, logits=True):
    err = 1 - multiclass_accuracy(
        preds, targets, preds.size(1), average="micro").item()
    ece = multiclass_calibration_error(preds, targets, preds.size(1)).item()
    if logits:
        nll = torch.nn.functional.cross_entropy(preds, targets).item()
    else:
        nll = torch.nn.functional.nll_loss(preds.add(1e-6).log(), targets).item()
    return err, ece, nll


def calibrate(logits, targets, num_iterations=1000):
    tau = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([tau])
    loss = torch.nn.CrossEntropyLoss()
    for _ in range(num_iterations):
        optimizer.zero_grad()
        loss(logits / tau, targets).backward()
        optimizer.step()
    return tau.item()


class Tee:
    def __init__(self, fname, stream, mode="w"):
        self.stream = stream
        self.file = open(fname, mode)

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

def metrics_ooo(unc, is_ood, tpr_th=0.95):
    from sklearn import metrics
    import numpy as np

    # following convention in ML we treat OOD as positive (is_ood = 1)
    id_ood = is_ood.view(-1).numpy()

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    conf = -unc.view(-1).numpy()

    fpr_list, tpr_list, thresholds = metrics.roc_curve(is_ood, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(is_ood, -conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(1 - is_ood, conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, fpr
