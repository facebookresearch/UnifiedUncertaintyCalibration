# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import submitit
import argparse
import warnings
import getpass
import random
import json
import sys
import os

import networks, datasets, uncertainties, methods, utils


def run_experiment(args):
    warnings.filterwarnings(
        "ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    os.makedirs(args["results_dir"], exist_ok=True)
    outfile = os.path.join(
        args["results_dir"],
        "_".join(["{}={}".format(k, args[k]) for k in args.keys() if
                  k != "results_dir"]))

    sys.stdout = utils.Tee(outfile + ".out", sys.stdout)
    sys.stderr = utils.Tee(outfile + ".err", sys.stderr)

    if args["precomputed"] == "yes":
        torch.cuda.is_available = lambda : False

    network = networks.get(args["network"])

    # f: features, l: logits, t: targets
    if args["precomputed"] == "yes":
        root = "/checkpoint/dlp/datasets_ood/precomputed_"
        f_va, l_va, t_va = torch.load(
            root + "ImageNet-va_{}.pt".format(args["network"]))
        f_te, l_te, t_te = torch.load(
            root + "ImageNet-te_{}.pt".format(args["network"]))
        f_oo, l_oo, t_oo = torch.load(
            root + "{}_{}.pt".format(args["dataset"], args["network"]))
    else:
        loader_va = datasets.get("ImageNet-va")
        loader_te = datasets.get("ImageNet-te")
        loader_oo = datasets.get(args["dataset"])

        f_va, l_va, t_va = network.flt(loader_va)
        f_te, l_te, t_te = network.flt(loader_te)
        f_oo, l_oo, t_oo = network.flt(loader_oo)

    tau_in = utils.calibrate(
        l_va, t_va, num_iterations=args["num_iterations_1"])
    l_va /= tau_in
    l_te /= tau_in
    l_oo /= tau_in

    uncertainty = uncertainties.get(
        args["uncertainty"], network.classifier.cpu())
    uncertainty.fit(f_va, l_va, t_va)
    u_va = uncertainty(f_va, l_va)
    u_te = uncertainty(f_te, l_te)
    u_oo = uncertainty(f_oo, l_oo)

    # OpenOOD metrics
    metrics_ooo = utils.metrics_ooo(
        torch.cat((u_te, u_oo)).view(-1),
        torch.cat((torch.zeros(len(u_te)), torch.ones(len(u_oo)))))

    method = methods.get(args["method"])
    method.fit(
        l_va, t_va, u_va, num_iterations=args["num_iterations_2"])
    s_va = method(l_va, u_va)
    s_te = method(l_te, u_te)
    s_oo = method(l_oo, u_oo)

    err, ece, nll = utils.metrics(s_oo, t_oo, logits=False)

    results = {
        "auc": metrics_ooo[0],
        "fpr": metrics_ooo[1],
        "err": err,
        "ece": ece,
        "nll": nll,
        "ood_acc": method.ood_acc,
    }

    if args["save_stats"]:
        torch.save({
            "args": args,
            #
            "logits_va": l_va.detach(),
            "logits_te": l_te.detach(),
            "logits_oo": l_oo.detach(),
            #
            "uncert_va": u_va.detach(),
            "uncert_te": u_te.detach(),
            "uncert_oo": u_oo.detach(),
            #
            "method_va": s_va.detach(),
            "method_te": s_te.detach(),
            "method_oo": s_oo.detach(),
            #
            "unc_thr": float(method.thres_unc),
            "unc_net": method.cal_net.state_dict(),
            #
            "results": results,
        }, outfile + ".pt")

    results["args"] = args
    print(json.dumps(results, indent=2))


def run_jobs(function, commands):
    executor = submitit.SlurmExecutor(
        folder="/checkpoint/" + getpass.getuser() + "/submitit/")
    executor.update_parameters(
        time=3 * 24 * 60,
        gpus_per_node=1,
        array_parallelism=512,
        cpus_per_task=16,
        mem="512GB",
        partition="learnlab")
    executor.map_array(function, commands)


def run_sweep(args):
    commands = []
    for network in networks.get("names"):
        for dataset in datasets.get("names"):
            for uncertainty in uncertainties.get("names"):
                for method in methods.get("names"):
                    args["network"] = network
                    args["dataset"] = dataset
                    args["uncertainty"] = uncertainty
                    args["method"] = method
                    commands.append(dict(args))

    for command in commands:
        print(command)

    run_jobs(run_experiment, commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNet Uncertainties")
    parser.add_argument("--results_dir", type=str, default="esults/")
    parser.add_argument("--network", type=str, default="ResNet50-v1")
    parser.add_argument("--dataset", type=str, default="iNaturalist")
    parser.add_argument("--uncertainty", type=str, default="MaxLogit")
    parser.add_argument("--method", type=str, default="Filter")
    parser.add_argument("--precomputed", type=str, default="yes")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num_iterations_1", type=int, default=10000)
    parser.add_argument("--num_iterations_2", type=int, default=10000)
    parser.add_argument("--save_stats", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = vars(parser.parse_args())

    if args["sweep"]:
        run_sweep(args)
    else:
        run_experiment(args)
