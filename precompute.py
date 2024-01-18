# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import datasets, networks
from main import run_jobs


def precompute_flt(args):
    loader = datasets.get(args["dataset_name"])
    network = networks.get(args["network_name"])
    f, l, t = network.flt(loader)
    torch.save(
        (f, l, t),
        "{}/precomputed_{}_{}.pt".format(
            datasets.ROOT_, args["dataset_name"], args["network_name"]))


if __name__ == "__main__":
    commands = []
    for dataset_name in datasets.get("names"):
        for network_name in networks.get("names"):
            commands.append({
                "dataset_name": dataset_name, "network_name": network_name
            })

    for command in commands:
        print(command)

    run_jobs(precompute_flt, commands)
