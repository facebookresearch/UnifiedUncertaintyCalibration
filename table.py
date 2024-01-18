# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import glob
import json
import argparse
import pandas as pd

import networks, datasets, uncertainties, methods


class Records:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]

    def unique(self, key):
        tmp = []
        for record in self:
            tmp.append(record[key])
        return list(set(tmp))

    def where(self, key, value):
        tmp = []
        for record in self:
            if record[key] == value:
                tmp.append(record)
        return Records(tmp)


def cell_string(old, new):
    old *= 100
    new *= 100
    if abs(old - new) < 0.1:
        return "${:<5.1f} \\,\\,{{\\color{{gray}}        (- {:<5.1f})}}$".format(old, 0)
    elif old > new:
        return "${:<5.1f} \\,\\,{{\\color{{ForestGreen}} (- {:<5.1f})}}$".format(old, old - new)
    else:
        return "${:<5.1f} \\,\\,{{\\color{{BrickRed}}    (+ {:<5.1f})}}$".format(old, new - old)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNet Uncertainties")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--network", type=str, default="ResNet50-v1")
    parser.add_argument("--old_method", type=str, default="Filter")
    parser.add_argument("--new_method", type=str, default="Dual")
    args = vars(parser.parse_args())

    record_list = []
    for fname in glob.glob(args["results_dir"] + "/*.out"):
        with open(fname, "r") as f:
            if os.stat(fname).st_size == 0:
                continue
            record = json.load(f)
            parsed_record = {
                "dataset": record["args"]["dataset"],
                "network": record["args"]["network"],
                "uncertainty": record["args"]["uncertainty"],
                "method": record["args"]["method"],
                "err": record["err"],
                "ece": record["ece"],
                "nll": record["nll"],
            }
            record_list.append(parsed_record)

    records = Records(record_list)
    records_2 = records.where("network", args["network"])

    UNCERTAINTIES = uncertainties.get("names")
    DATASETS = datasets.get("names")
    STATS = ["err", "ece"]

    # print("\\documentclass{article}")
    # print("\\usepackage{booktabs}")
    # print("\\usepackage{fullpage}")
    # print("\\usepackage{multirow}")
    # print("\\usepackage[dvipsnames]{xcolor}")
    # print("\\usepackage{nicematrix}")
    # print("\\begin{document}")
    print("\\begin{NiceTabular}{ll" + "r" * len(UNCERTAINTIES) + "}")
    print("\\CodeBefore")
    print("\\rectanglecolor{white}{2-1}{5-6}")
    print("\\rectanglecolor{gray!10}{6-1}{11-6}")
    print("\\rectanglecolor{yellow!10}{12-1}{15-6}")
    print("\\rectanglecolor{orange!10}{16-1}{24-6}")
    print("\\Body")
    print("\\toprule")
    print("& & " + " & ".join(["\\textbf{{{}}}".format(u) for u in UNCERTAINTIES]) + "\\\\")
    print("\\midrule")

    for dataset in DATASETS:
        records_3 = records_2.where("dataset", dataset)
        for stat in STATS:
            if stat == STATS[0]:
                print("{:<30} & {} & ".format("\multirow{{2}}{{*}}{{{}}}".format(dataset), stat), end="")
            else:
                print("{:<30} & {} & ".format("", stat), end="")
            cells = []
            for uncertainty in UNCERTAINTIES:
                records_4 = records_3.where("uncertainty", uncertainty)
                old_res = records_4.where("method", args["old_method"])[0]
                new_res = records_4.where("method", args["new_method"])[0]
                cells.append(cell_string(old_res[stat], new_res[stat]))
            print(" & ".join(cells) + "\\\\")
        if dataset != DATASETS[-1]:
            print("\\midrule")
    print("\\bottomrule")
    print("\\end{NiceTabular}")
    # print("\\end{document}")
