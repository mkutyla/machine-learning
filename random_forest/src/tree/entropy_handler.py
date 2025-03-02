import math
import random

import numpy as np
import pandas as pd


def best_split_mean(data: pd.DataFrame, cls: str, attribute: str) -> list:
    split = data[attribute].mean()
    entropy = calc_conditional_entropy(data, cls, attribute, split)
    return [entropy, [split]]


def best_split_median(data: pd.DataFrame, cls: str, attribute: str) -> list:
    split = data[attribute].median()
    entropy = calc_conditional_entropy(data, cls, attribute, split)
    return [entropy, [split]]


def best_split_buckets(data: pd.DataFrame, cls: str, attribute: str) -> list:
    number_of_bins = math.floor(math.sqrt(len(data)))
    _, values = pd.qcut(data[attribute],
                        q=number_of_bins,
                        retbins=True, duplicates='drop')
    best_split = 0
    best_entropy = float("inf")

    for split in values:
        entropy = calc_conditional_entropy(data, cls, attribute, split)
        if entropy < best_entropy:
            best_entropy = entropy
            best_split = split

    if best_entropy == float("inf"):
        best_entropy = calc_entropy(data, cls)

    return [best_entropy, [best_split]]


def best_split_search(data: pd.DataFrame, cls: str, attribute: str) -> list:
    best_split = 0
    best_entropy = float("inf")

    values = list(set(data[attribute].to_list()))  # eliminating duplicates
    values.sort()

    for first, second in zip(values[:-1], values[1:]):
        split = (first + second) / 2
        entropy = calc_conditional_entropy(data, cls, attribute, split)
        if entropy < best_entropy:
            best_entropy = entropy
            best_split = split
    if best_entropy == float("inf"):
        best_entropy = calc_entropy(data, cls)

    return [best_entropy, [best_split]]


def calc_conditional_entropy(data: pd.DataFrame, cls: str, attribute: str, split: float):
    df = data[data[attribute].values <= split]
    df2 = data[data[attribute] > split]
    p1 = calc_entropy(df, cls)
    p2 = calc_entropy(df2, cls)

    entropy = len(df) / len(data) * p1 + len(df2) / len(data) * p2
    return entropy


def calc_entropy(df: pd.DataFrame, column_name: str) -> float:
    probabilities = df[column_name].value_counts(normalize=True, sort=False).values
    return -np.sum(probabilities * np.log2(probabilities), where=(probabilities > 0))


def get_random_element(elements: list, probabilities: list):
    denominator = sum(probabilities)
    probabilities = [x / denominator for x in probabilities]
    selected = int(np.random.choice(np.arange(0, len(elements)), p=probabilities))
    choice = elements[selected]

    return choice


def get_most_frequent_val(data: pd.DataFrame, column_name: str):
    classes_counters = data[column_name].value_counts(sort=False).to_dict()
    max_counter = max(classes_counters.values())
    max_classes = [k for k, v in classes_counters.items() if v == max_counter]

    # return best class or select one of the best at random
    return max_classes[random.randint(0, len(max_classes) - 1)]
