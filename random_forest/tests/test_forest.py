from src.forest.random_forest import RandomForest
import pandas as pd
import src.utilities.quality as qt
import src.utilities.dataframes_handler as fh


# def compare_modes():
#     for param_set in params:
#         df = read_file(param_set["filename"])
#         target = param_set["target"]
#         attributes = df.columns.tolist()
#         attributes.remove(target)
#
#         dt = DecisionTree(target, param_set["continuous"])
#         x = get_random_attributes(attributes)
#         for mode in dt.modes:
#             for _ in range(10):
#                 print(f'Fitting with mode = {mode}')
#                 dt.fit(df, x, mode=mode)
#                 predicted = dt.predict(df)
#                 expected = df[target]
#                 qt.check_correctness(predicted, expected)
#                 print()
#
#     # classes = df[target].unique().tolist()
#     # classes.sort()
#     # cm.print_conf_matrix(classes, predicted, expected)
#     # cm.draw_conf_matrix(classes, predicted, expected)
#
def get_classes(data: pd.DataFrame, cls: str):
    return data[cls].unique().tolist()


if __name__ == "__main__":
    param = fh.params["wine"]
    df = fh.read_file(param["filename"])
    target = param["target"]

    attributes = df.columns.tolist()
    attributes.remove(target)
    quality = qt.Quality(get_classes(df, target))

    rf = RandomForest(df, continuous_attributes=param["continuous"])
    for i in range(1):
        rf.fit(target, 500,
               model='id3',
               split_strategy='mean',
               threshold_strategy='best',
               attributes_per_tree=3)
        predicted = rf.predict()
        expected = rf.expected

        quality.set_predicted(predicted)
        quality.set_expected(expected)

        quality.get_confusion_matrix()
        quality.print_confusion_matrix()
        measures = quality.get_measures()
        print(measures)
