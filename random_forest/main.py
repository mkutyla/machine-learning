from src.forest.random_forest import RandomForest
import pandas as pd
import src.utilities.quality as qt
import src.utilities.dataframes_handler as fh


def get_classes(data: pd.DataFrame, cls: str):
    return data[cls].unique().tolist()


param = fh.params["wine"]
df = fh.read_file(param["filename"])
target = param["target"]

attributes = df.columns.tolist()
attributes.remove(target)
quality = qt.Quality(get_classes(df, target))

rf = RandomForest(df, continuous_attributes=param["continuous"])
for i in range(1):
    rf.fit(target, 100, split_strategy='buckets', threshold_strategy='none')
    predicted = rf.predict()
    expected = rf.expected

    quality.set_predicted(predicted)
    quality.set_expected(expected)

    quality.get_confusion_matrix()
    quality.print_confusion_matrix()
    measures = quality.get_measures()
    print(measures)
    print(f'[n={10 * i}] ACC={measures["ACC"]}')
    quality.draw_confusion_matrix()
