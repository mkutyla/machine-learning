import pandas as pd
import numpy as np
import random
import asyncio

from src.tree.decision_tree import DecisionTree
import src.utilities.dataframes_handler as dfh

from sklearn import tree


class RandomForest:
    def __init__(self, df: pd.DataFrame, continuous_attributes: list[str] = None) -> None:
        # data frame handling
        self.df = df
        self.continuous_attributes = continuous_attributes
        self.preprocess_set()
        self.training_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        self.predicted = pd.DataFrame()
        self.expected = pd.DataFrame()

        # default values
        self.model = None
        self.cls = None
        self.split_strategy = None
        self.threshold_strategy = None
        self.trees = list()
        self.attributes = list()
        self.forest_size = 0
        self.attributes_per_tree = 0
        self.cls_type = None

        if self.continuous_attributes is not None:
            self.preprocess_set()

        # constants
        self.models = ('default', 'id3')
        self.split_strategies = ('mean', 'median', 'buckets', 'search')
        self.threshold_strategies = ('none', 'best')
        self.modes_msg = f'[!] Permitted modes are {", ".join(self.models)}'
        self.split_strategy_msg = f'[!] Permitted split strategies are [{", ".join(self.split_strategies)}]'
        self.threshold_strategy_msg = f'[!] Permitted threshold strategies are [{", ".join(self.threshold_strategies)}]'

    def preprocess_set(self):
        """
        Handles NaN values
        """
        for a in self.continuous_attributes:
            self.df[a] = self.df[a].replace(np.NaN, self.df[a].mean())

    def fit(self,
            class_attribute: str,
            forest_size: int,
            attributes_per_tree: int = None,
            model: str = 'default',
            split_strategy: str = 'buckets',
            threshold_strategy: str = 'none') -> None:

        if model in self.models:
            self.model = model
        else:
            raise Exception(self.modes_msg)

        if split_strategy in self.split_strategies:
            self.split_strategy = split_strategy
        else:
            raise Exception(self.split_strategy_msg)

        if threshold_strategy in self.threshold_strategies:
            self.threshold_strategy = threshold_strategy
        else:
            raise Exception(self.threshold_strategy_msg)

        self.attributes = self.df.columns.tolist()
        self.attributes.remove(class_attribute)

        if attributes_per_tree is None:
            self.attributes_per_tree = int(np.sqrt(len(self.attributes)))
        elif attributes_per_tree > 0:
            self.attributes_per_tree = attributes_per_tree
        else:
            raise Exception(f'[!] attributes_per_tree must be greater than 0')

        self.cls = class_attribute
        self.forest_size = forest_size
        self.split_strategy = split_strategy
        self.threshold_strategy = threshold_strategy

        self.training_set, self.test_set = dfh.split_set(self.df, self.cls)
        self.expected = self.test_set[self.cls]

        self.cls_type = self.df[self.cls].dtype

        self.trees = list()
        self.predicted = pd.DataFrame()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.async_fit())

    async def async_fit(self):
        tasks = []
        if len(self.trees) != 0:  # already fitted
            return

        match self.model:
            case 'default':
                func = self.async_fit_tree
            case 'id3':
                func = self.async_fit_tree_id3
            case _:
                raise Exception(self.modes_msg)

        for _ in range(self.forest_size):
            task = asyncio.ensure_future(func())
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def async_fit_tree(self):
        tree_set = self.training_set.sample(replace=True, frac=1)
        chosen_attributes = random.sample(self.attributes, self.attributes_per_tree)

        clf = DecisionTree(self.cls, self.continuous_attributes)
        clf.fit(tree_set,
                chosen_attributes,
                split_strategy=self.split_strategy,
                threshold_strategy=self.threshold_strategy)
        self.trees.append(clf)

    async def async_fit_tree_id3(self):
        tree_set = self.training_set.sample(replace=True, frac=1)
        chosen_attributes = random.sample(self.attributes, self.attributes_per_tree)

        target = tree_set.loc[:, self.cls]
        chosen_columns = tree_set.loc[:, chosen_attributes]

        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=len(chosen_attributes))
        clf = clf.fit(chosen_columns, target)
        pr = np.array(clf.predict(self.test_set.loc[:, chosen_attributes]))
        pr = pd.DataFrame(pr)

        self.predicted = pd.concat(objs=[self.predicted, pr], axis=1)

    def predict(self):
        match self.model:
            case 'id3':
                return self.get_most_frequent()
            case 'default':
                if len(self.predicted) != 0:
                    return self.get_most_frequent()
            case _:
                raise Exception(self.modes_msg)
        # runs only if model = 'default' and len(self.predicted)==0
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(self.async_predict())
        loop.run_until_complete(future)

        return self.get_most_frequent()

    async def async_predict(self):
        tasks = []
        for clf in self.trees:
            task = asyncio.ensure_future(self.async_predict_tree(clf))
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def async_predict_tree(self, clf: DecisionTree):
        pr = pd.DataFrame(clf.predict(self.test_set))
        self.predicted = pd.concat(objs=[self.predicted, pr], axis=1)

    def get_most_frequent(self) -> list:
        most_frequent = self.predicted.mode(axis=1, dropna=True)
        most_frequent = most_frequent.stack().groupby(level=0).sample(n=1)

        if self.cls_type == 'int64':
            most_frequent = most_frequent.astype(int)
        return most_frequent.tolist()
