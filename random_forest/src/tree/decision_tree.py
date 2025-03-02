import numpy as np
import pandas as pd

import src.tree.node as nd
import src.utilities.dataframes_handler as dfh
import src.tree.entropy_handler as eh

pd.set_option('chained_assignment', None)  # default='warn'


class DecisionTree:

    def __init__(
            self, class_attribute: str, continuous_attr: list[str]
    ) -> None:
        self.cls = class_attribute
        self.continuous_attr = continuous_attr

        self.split_strategies = ('mean', 'median', 'buckets', 'search')
        self.threshold_strategies = ('none', 'best')
        self.split_strategy_msg = f'[!] Permitted split strategies are [{", ".join(self.split_strategies)}]'
        self.threshold_strategy_msg = f'[!] Permitted threshold strategies are [{", ".join(self.threshold_strategies)}]'

        self.predicted = None
        self.root = None
        self.training_set = None
        self.split_strategy = None
        self.threshold_strategy = None

    def threshold(self, data: pd.DataFrame, attributes: list[str]) -> dict:
        information_gain = dict()
        if len(attributes) == 0:
            return information_gain

        src_entropy = eh.calc_entropy(data, self.cls)
        for A in attributes:
            entropy, split = self.calculate_best_split(data, A)
            info_gain = src_entropy - entropy
            if info_gain <= 0:
                # information_gain = 0 <= it's not worth to split;
                # if <0 then it's caused by binary representation on most likely ~0, noticed: -1.11e-16
                continue
            information_gain[A] = [info_gain, split]

        if self.threshold_strategy == 'best' and len(information_gain) != 0:
            max_key = max(information_gain, key=information_gain.get)
            information_gain = {max_key: information_gain[max_key]}

        return information_gain

    def calculate_best_split(self, data: pd.DataFrame, attribute: str) -> list[float, float]:
        """
        Calculates the best possible split for passed data when splitting by passed attribute.

        Args:
            data:       a data set
            attribute:  to split the data set by

        Returns: a two element list
            first element   : float                 - information gain
            second element  : float or list[float]  - best split(s)
        """
        if attribute in self.continuous_attr:
            match self.split_strategy:
                case 'mean':
                    return eh.best_split_mean(data, self.cls, attribute)
                case 'median':
                    return eh.best_split_median(data, self.cls, attribute)
                case 'buckets':
                    return eh.best_split_buckets(data, self.cls, attribute)
                case 'search':
                    return eh.best_split_search(data, self.cls, attribute)
                case _:
                    raise Exception(f'{self.split_strategy_msg}, got {self.split_strategy} instead')

        else:
            entropy = 0
            domain = dfh.get_attribute_domain(data, attribute)
            for x in domain:
                df = data[data[attribute].values == x]
                s = len(df) / len(data) * eh.calc_entropy(df, self.cls)
                entropy += s
            return [entropy, domain]

    def add_branch(self, other_node: nd.Node, split_by) -> None:
        self.root.add_child_branch(other_node, split_by)

    def get_next_node(
            self, data: pd.DataFrame,
            split_data: pd.DataFrame, attributes: list[str],
            split_attribute: str
    ) -> nd.Node:
        next_node = nd.Node()
        if len(split_data) == 0:
            next_node.cls = eh.get_most_frequent_val(data, self.cls)
        else:
            sub_tree = DecisionTree(self.cls, self.continuous_attr)
            sub_tree.split_strategy = self.split_strategy
            sub_tree.threshold_strategy = self.threshold_strategy

            next_node = sub_tree.build_tree(split_data, [x for x in attributes if x != split_attribute])

        return next_node

    def fit(self, data: pd.DataFrame,
            attributes: list[str],
            split_strategy: str = 'mean',
            threshold_strategy: str = 'none') -> None:

        if split_strategy in self.split_strategies:
            self.split_strategy = split_strategy
        else:
            raise Exception(self.split_strategy_msg)

        if threshold_strategy in self.threshold_strategies:
            self.threshold_strategy = threshold_strategy
        else:
            raise Exception(self.threshold_strategy_msg)

        self.training_set = data
        self.build_tree(self.training_set, attributes)

    def build_tree(self, data: pd.DataFrame, attributes: list[str]) -> nd.Node:
        self.root = nd.Node()
        if len(c := dfh.get_attribute_domain(data, self.cls)) == 1:
            self.root.cls = c[0]
            return self.root

        attribute_ig = self.threshold(data, attributes)
        if len(attribute_ig) == 0:
            self.root.cls = eh.get_most_frequent_val(data, self.cls)
            return self.root

        attribute = eh.get_random_element(list(attribute_ig.keys()), [x[0] for x in attribute_ig.values()])
        self.root.label = attribute
        split_values = attribute_ig[attribute][1]  #

        if len(split_values) == 1:  # continuous attribute: split in the form of a binary inequality
            split_value = split_values[0]

            next_node1 = self.get_next_node(data, data[data[attribute] <= split_value], attributes, attribute)
            next_node2 = self.get_next_node(data, data[data[attribute] > split_value], attributes, attribute)

            if next_node1.is_leaf() and next_node2.is_leaf():
                if next_node1.cls == next_node2.cls:
                    # collapse, change to leaf
                    self.root.transform_to_leaf(next_node1.cls)
                    return self.root

            self.add_branch(next_node1, f"<={split_value}")
            self.add_branch(next_node2, f">{split_value}")

        else:
            for split_value in split_values:
                split_data = data[data[attribute] == split_value]

                next_node = self.get_next_node(data, split_data, attributes, attribute)

                self.add_branch(next_node, f"={split_value}")

        return self.root  # recursive

    def predict(self, df: pd.DataFrame, prediction_label: str = None) -> np.array:
        if self.root is None:
            Exception("The tree was not trained yet!")

        if prediction_label is None:
            prediction_label = self.cls + " prediction"

        df[prediction_label] = None

        df.loc[:, prediction_label] = self.__classify(self.root, df, prediction_label)
        prediction = df[prediction_label].copy()
        df.drop(columns=prediction_label, inplace=True)

        # random selection if couldn't classify based on training set
        classes = dfh.get_attribute_domain(self.training_set, self.cls)
        prediction.apply(lambda x: x if x is not None else np.random.choice(classes))
        self.predicted = np.array(prediction)

        return self.predicted

    def __classify(self, node: nd.Node, df: pd.DataFrame, label: str):
        if node.is_leaf():
            df[label].replace(np.nan, node.cls, inplace=True)
            return df[label]

        attribute = node.label
        for branch in node.children_branches:
            condition = str(branch.label)
            if "<=" in condition:
                value = float(condition[2:])  # cut off "<="
                condition = df[attribute] <= value

            elif ">" in condition:
                value = float(condition[1:])  # cut off ">"
                condition = df[attribute] > value

            else:  # discrete attribute

                value = condition[1:]  # cut off "=", value is either str or numeric
                try:
                    value = float(value)
                except ValueError:
                    ...
                condition = df[attribute] == value

            next_node = branch.child
            df.loc[condition, label] = self.__classify(next_node, df[condition], label).values

        return df[label]
