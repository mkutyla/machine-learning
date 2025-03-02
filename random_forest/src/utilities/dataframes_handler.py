import numpy as np
import pandas as pd

params = {
    "wine": {
        "filename": "data/wine.csv",
        "target": "quality",
        "continuous": ["fixed acidity", "volatile acidity", "citric acid",
                       "residual sugar", "chlorides", "free sulfur dioxide",
                       "total sulfur dioxide", "density", "pH",
                       "sulphates", "alcohol", "quality"]
    },
    "mushrooms": {
        "filename": "data/mushrooms.csv",
        "target": "class",
        "continuous": []
    },
    "heart": {
        "filename": "data/heart.csv",
        "target": "output",
        "continuous": []
    },
    "diabetes": {
        "filename": "data/diabetes.csv",
        "target": "diabetes",
        "continuous": ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    }
}


def read_file(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df


def get_attribute_domain(data: pd.DataFrame, cls: str):
    return data[cls].unique().tolist()


def split_set(df: pd.DataFrame, cls: str,
              training_size: int = 5,
              test_size: int = 5):
    df_classes = [df[df[cls] == c] for c in get_attribute_domain(df, cls)]

    training_set = None
    test_set = None
    for df_class in df_classes:
        splits = np.array_split(df_class, training_size + test_size)
        for split in splits[:test_size]:
            if test_set is None:
                test_set = split
            else:
                test_set = pd.concat([test_set, split], ignore_index=True)
        for split in splits[test_size:]:
            if training_set is None:
                training_set = split
            else:
                training_set = pd.concat([training_set, split], ignore_index=True)

    if type(df[cls]) == 'int64':
        training_set[cls] = training_set[cls].astype('int')
        test_set[cls] = test_set[cls].astype('int')

    return [training_set, test_set]
