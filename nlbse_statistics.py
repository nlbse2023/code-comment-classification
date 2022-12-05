import os
import sys

import pandas as pd
from typing import Tuple


def get_precision_recall_f1(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall / (precision + recall))
    return precision, recall, f1


def get_csv_path(input_path: str, category: str, language: str, type: str) -> str:
    csv_path_train = os.path.join(input_path, category)
    return os.path.join(csv_path_train, "{}_{}_{}.csv".format(language, category, type))


def get_statistics(dataset_path: str, stats_filename: str, remove_weighted: bool = False) -> None:
    languages = ["java", "pharo", "python"]

    categories = [["deprecation", "pointer", "summary", "expand", "ownership", "rational", "usage"],
                  ["classreferences", "example", "keyimplementation", "collaborators", "intent", "keymessages", "responsibilities"],
                  ["developmentnotes", "parameters", "summary", "expand", "usage"]]

    stats = {"category": [], "precision": [], "recall": [], "f1": [],
             "weighted_precision": [], "weighted_recall": [], "weighted_f1": []}

    for idx, language in enumerate(languages):
        result_path = os.path.join(os.path.join(dataset_path, language), "results")
        for category in categories[idx]:
            new_res = {"true_positive": [], "false_positive": [], "true_negative": [], "false_negative": [], "precision": [], "recall": [], "f1": []}
            # Get results from CSV
            csv_path_result = os.path.join(result_path, category)
            csv_weighted_filename = os.path.join(csv_path_result, "0-0-{}-tfidf-heuristic-randomforest-outputs.csv".format(category))
            df_result = pd.read_csv(csv_weighted_filename, index_col="type")

            # Get only test row from CSV
            test_metrics = df_result.loc["test"]

            # Calculate Precision, Recall, anf F1 as in Colab
            precision, recall, f1 = get_precision_recall_f1(test_metrics["tp"], test_metrics["fp"], test_metrics["fn"])

            # Append statistics of our classifiers
            stats["category"].append("{}_{}".format(language, category))
            stats["precision"].append(precision)
            stats["recall"].append(recall)
            stats["f1"].append(f1)
            # Append also paper's statistics
            stats["weighted_precision"].append(test_metrics["w_pr"])
            stats["weighted_recall"].append(test_metrics["w_re"])
            stats["weighted_f1"].append(test_metrics["w_f_measure"])

            # Save non-weighted metrics
            new_res["true_positive"].append(test_metrics["tp"])
            new_res["false_positive"].append(test_metrics["fp"])
            new_res["true_negative"].append(test_metrics["tn"])
            new_res["false_negative"].append(test_metrics["fn"])
            new_res["precision"].append(precision)
            new_res["recall"].append(recall)
            new_res["f1"].append(f1)
            csv_non_weighted_filename = os.path.join(csv_path_result, "0-0-{}-tfidf-heuristic-randomforest-outputs-non-weighted.csv".format(category))
            pd.DataFrame(new_res).to_csv(csv_non_weighted_filename, index=False)

            # Remove weighted metrics CSV
            if remove_weighted:
                os.remove(csv_weighted_filename)
                os.rename(csv_non_weighted_filename, csv_weighted_filename)

    d = pd.DataFrame(stats, index=stats["category"])
    d = d.loc[:, d.columns != "category"]
    d.to_csv(stats_filename)


def generate_tables(stat_filename: str) -> None:
    data = pd.read_csv(stat_filename)
    print("| Language | Category           | Precision $P_c$ | Recall $R_c$ | $F_ {1,c} $ |")
    print("|----------|--------------------|----------------:|-------------:|------------:|")
    for row in data.iterrows():
        row = row[1]
        language, category = row["Unnamed: 0"].split('_')
        print("| {: <8} | {:18} | {: >14}% | {: >11}% | {: >10}% |".
              format(language.capitalize(), category.capitalize(), round(row["precision"] * 100, 1), round(row["recall"] * 100, 1), round(row["f1"] * 100, 1)))
    print("|----------|--------------------|-----------------|--------------|-------------|")
    print("| Overall  |                    | {: >14}% | {: >11}% | {: >10}% |".
          format(round(data["precision"].mean() * 100, 1), round(data["recall"].mean() * 100, 1), round(data["f1"].mean() * 100, 1)))
    print()


def get_new_stats(dataset_path: str) -> None:
    languages = ["java", "pharo", "python"]

    print("| Language | Category           | Training | Training | Testing  | Testing  | Total  |")
    print("|----------|--------------------|---------:|---------:|---------:|---------:|-------:|")
    for language in languages:
        filename = os.path.join(dataset_path, language)
        filename = os.path.join(filename, "input")
        filename = os.path.join(filename, "{}.csv".format(language))
        df = pd.read_csv(filename)

        cats = set()
        for row in df.iterrows():
            cats.add(row[1]['category'])

        print("|          |                    | Positive | Negative | Positive | Negative |        |")
        for category in cats:
            train_pos = len(df.loc[(df['partition'] == 0) & (df['instance_type'] == 1) & (df['category'] == category)])
            test_pos = len(df.loc[(df['partition'] == 0) & (df['instance_type'] == 0) & (df['category'] == category)])
            train_neg = len(df.loc[(df['partition'] == 1) & (df['instance_type'] == 1) & (df['category'] == category)])
            test_neg = len(df.loc[(df['partition'] == 1) & (df['instance_type'] == 0) & (df['category'] == category)])
            cat_sum = train_pos + test_pos + train_neg + test_neg

            print("| {: <8} | {:18} | {: >8} | {: >8} | {: >8} | {: >8} | {: >6} |"
                  .format(language.capitalize(), category.capitalize(), train_pos, test_pos, train_neg, test_neg, cat_sum))

    print("|----------|--------------------|----------|----------|----------|----------|--------|")
    # print()


if __name__ == '__main__':
    # Pass as the fist argument the path of the dataset. E.g., "/Users/luca/Downloads/code-comment-classification"
    path = sys.argv[1]

    # get_new_stats(path)

    stat_file = "statistics.csv"
    get_statistics(path, stat_file)
    generate_tables(stat_file)
