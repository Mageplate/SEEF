import os
import sys
import shutil
import pandas as pd
import numpy as np
from cluster import mk_cluster, ak_cluster, rk_cluster
from sklearn.decomposition import NMF
from method_evaluation import MethodEvaluator

sys.path.append(os.path.dirname(__file__) + "/../AE")
from ptAE import mseAE, klAE

sys.path.append(os.path.dirname(__file__) + "/../tools")
from tools import plot_signatures, plot_weights

current_file_path = os.path.dirname(__file__)
evaluator = MethodEvaluator()


def find_datasets(path: str = os.path.dirname(__file__) + "/datasets"):
    datasets = []
    # for all folders in path, check if they contain a dataset
    for folder in os.listdir(path):
        if folder[0] == ".":
            continue

        if os.path.isdir(path + "/" + folder):
            # if folder contains one file else if contains 3 files (my contain one folder but shuld be ignored)
            folder_list = [
                x
                for x in os.listdir(path + "/" + folder)
                if not os.path.isdir(path + "/" + folder + "/" + x)
            ]
            if len(folder_list) == 1:
                # find dataset
                datasets.append(
                    {
                        "type": 0,
                        "folder": path + "/" + folder,
                        "dataset": folder_list[0],
                    }
                )

            elif len(folder_list) == 3:
                # find dataset, signature and weights
                dataset = None
                signature = None
                weights = None
                for file in folder_list:
                    if "sig" in file:
                        signature = file
                    elif "wei" in file:
                        weights = file
                    else:
                        dataset = file

                datasets.append(
                    {
                        "type": 1,
                        "folder": path + "/" + folder,
                        "dataset": dataset,
                        "signature": signature,
                        "weights": weights,
                    }
                )
    return datasets


def nmf(df, components):
    model = NMF(n_components=components, init="random", random_state=0)
    W = model.fit_transform(df)
    H = model.components_
    return W, H, model.reconstruction_err_


def klnmf(df, components):
    model = NMF(
        n_components=components,
        init="random",
        random_state=0,
        beta_loss="kullback-leibler",
        solver="mu",
    )
    W = model.fit_transform(df)
    H = model.components_
    return W, H, model.reconstruction_err_


def evaluate(dataset: dict):
    if dataset["type"] == 0:
        return evaluator.COSMICevaluate(
            current_file_path + "/nmf_output/signatures.tsv",
            "GRCh37",
        )
    else:
        return evaluator.evaluate(
            current_file_path + "/nmf_output/signatures.tsv",
            current_file_path
            + "/nmf_output/output_weights/Assignment_Solution/Activities/Assignment_Solution_Activities.txt",
            dataset["folder"] + "/" + dataset["signature"],
            dataset["folder"] + "/" + dataset["weights"],
        )


def save_results(results, methods, dataset):
    if dataset["type"] == 0:
        columns = ["found", ">0.8", ">0.95", "best>0.95", "best>0.99", "match"]
    elif dataset["type"] == 1:
        columns = [
            "found",
            ">0.8",
            ">0.95",
            "best>0.95",
            "best>0.99",
            "match",
            "mse",
            "mae",
            "rmse",
        ]
    else:
        columns = None
    return pd.DataFrame(
        results,
        columns=columns,
        index=methods,
    )


def save_output(method, dataset):
    # make results method folder if not exist
    if not os.path.exists(dataset["folder"] + f"/results/{method}"):
        os.makedirs(dataset["folder"] + f"/results/{method}")

    shutil.copy(
        current_file_path + "/nmf_output/aux_loss.png",
        dataset["folder"] + f"/results/{method}/aux_loss.png",
    )
    shutil.copy(
        current_file_path + "/nmf_output/signatures.tsv",
        dataset["folder"] + f"/results/{method}/signatures.tsv",
    )
    shutil.copy(
        current_file_path
        + "/nmf_output/output_weights/Assignment_Solution/Activities/Assignment_Solution_Activities.txt",
        dataset["folder"] + f"/results/{method}/weights.txt",
    )


def plot_output(method, dataset):
    plot_signatures(
        dataset["folder"] + f"/results/{method}/signatures.tsv",
        method,
        dataset["folder"] + f"/results/{method}",
    )

    plot_weights(
        dataset["folder"] + f"/results/{method}/weights.txt",
        dataset["folder"] + f"/results/{method}",
    )


def test_dataset(dataset, methods):
    # make results folder if not exist
    if not os.path.exists(dataset["folder"] + "/results"):
        os.makedirs(dataset["folder"] + "/results")

    results = []

    for method in methods:
        print(f"Testing {dataset['folder'].split('/')[-1]} with {method}")
        if method == "nmf_mk_mse":
            mk_cluster(dataset["folder"] + "/" + dataset["dataset"], nmf).run()
        elif method == "nmf_mk_kl":
            mk_cluster(dataset["folder"] + "/" + dataset["dataset"], klnmf).run()
        elif method == "AE_ak_mse":
            ak_cluster(
                dataset["folder"] + "/" + dataset["dataset"], mseAE, latents=200
            ).run()
        elif method == "AE_ak_kl":
            ak_cluster(
                dataset["folder"] + "/" + dataset["dataset"], klAE, latents=200
            ).run()

        save_output(method, dataset)

        results.append(evaluate(dataset))

    # save results
    save_results(results, methods, dataset).to_csv(
        dataset["folder"] + "/results/results.csv"
    )


if __name__ == "__main__":
    datasets = find_datasets(current_file_path + "/datasets")
    # methods = ["nmf_mk_mse", "nmf_mk_kl"]
    # methods = ["AE_ak_mse", "AE_ak_kl"]
    methods = ["nmf_mk_mse", "nmf_mk_kl", "AE_ak_mse", "AE_ak_kl"]
    for dataset in datasets:
        test_dataset(dataset, methods)

    for dataset in datasets:
        for method in methods:
            plot_output(method, dataset)
