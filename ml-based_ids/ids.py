__author__ = "Miłosz Kutyła, Jakub Ossowski, Jan Walczak, and Patryk Jankowicz"
__credits__ = ["Miłosz Kutyła", "Jakub Ossowski", "Jan Walczak", "Patryk Jankowicz"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Miłosz Kutyła, Jakub Ossowski"
__email__ = "milosz.kutyla.stud@pw.edu.pl, jakub.ossowski.stud@pw.edu.pl"
__status__ = "Production"

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ais import ArtificialImmuneSystem
from som_w_lvq import SelfOrganizingMap

def get_data(
        file_names = 'dataset/nll-kdd/KDD.names',
        file_data = 'dataset/nll-kdd/KDDTest-21.txt',
        to_drop = ['difficulty'],
    ):
    pandas_names = []
    features = {}

    with open(file_names, 'r') as file:
        for line in file.readlines():
            feature, feature_type = line.strip().split(':')
            feature_type = feature_type.strip().replace('.','')
            features[feature] = feature_type
            pandas_names.append(feature)

    data = pd.read_csv(
        file_data,   
        names = pandas_names
        )
    
    symbolic_feature_mapping = {}
    for feature in features.keys():
        if features[feature] == 'symbolic':
            values = data[feature].unique().tolist()
            symbolic_feature_mapping[feature] = {}
            for i, value in enumerate(values):
                symbolic_feature_mapping[feature][value] = i

    if len(to_drop) != 0:
        data = data.drop(to_drop, axis=1)

    if len(symbolic_feature_mapping.keys()) != 0:
        data = data.replace(symbolic_feature_mapping)
        data = data.replace(symbolic_feature_mapping)

    return data

def main():
    ais_file = 'pickles/ais_500_4.1.pickle'
    som_file = 'pickles/som_150x150_lvq.pickle'

    # with open(ais_file, 'rb') as pickle_file:
    #     ais = pickle.load(pickle_file)
    ais = ArtificialImmuneSystem(1, 1)
    ais : ArtificialImmuneSystem = ais.load_pickle(ais_file)
    som = SelfOrganizingMap(1, [1,1])
    som : SelfOrganizingMap = som.load_pickle(som_file)
    
    actual_class_col = 'attack_type'
    AIS_prediction_col = 'AIS'
    SOM_prediction_col = 'SOM'

    print('Getting data')
    data = get_data(    
        # file_names = 'dataset/kddcup.names',
        # file_data = 'dataset/kddcup.data_10_percent/kddcup.data_10_percent',
        # to_drop = []
    )

    print('Predicting with AIS')
    ais_data = data.drop([actual_class_col], axis=1)

    sc = ais.normalizer
    ais_data = sc.fit_transform(ais_data)

    ais_prediction = ais.predict(
        ais_data
    )
   
    data[AIS_prediction_col] = ais_prediction

    print('Predicting with SOM')
    som_data = data.drop([actual_class_col, AIS_prediction_col], axis=1)

    sc = som.normalizer
    som_data = sc.fit_transform(som_data)
    som_prediction = som.predict(
        som_data
    )
    data[SOM_prediction_col] = som_prediction

    # tp wtedy gdy ruch jest 'normal' i:
    # AIS = 'normal' 
    # lub
    # AIS = 'malicious' oraz SOM = 'normal'
    tp_condition = ((data[actual_class_col] == 'normal') & (data[AIS_prediction_col] == 'normal')) | ((data[actual_class_col] == 'normal') & (data[SOM_prediction_col] == 'normal'))
    
    # fn wtedy gdy ruch jest 'normal' i:
    # AIS = 'malicious' oraz SOM = 'malicious'
    fn_condition = ((data[actual_class_col] == 'normal') & (data[AIS_prediction_col] != 'normal') & (data[SOM_prediction_col] != 'normal'))

    # fp wtedy gdy ruch jest 'malicious' i:
    # AIS = 'normal'
    # lub
    # AIS = 'malicious' oraz SOM = 'normal'
    fp_condition = ((data[actual_class_col] != 'normal') & (data[AIS_prediction_col] == 'normal')) | ((data[actual_class_col] != 'normal') & (data[SOM_prediction_col] == 'normal'))
    # fp_condition = ((data[actual_class_col] != 'normal') & (data[AIS_prediction_col] != 'normal') & (data[SOM_prediction_col] == 'normal'))

    # tn wtedy gdy ruch jest 'malicious' i:
    # AIS = 'malicious' oraz SOM = 'malicious'
    tn_condition = ((data[actual_class_col] != 'normal') & (data[AIS_prediction_col] != 'normal') & (data[SOM_prediction_col] != 'normal'))

    tp = len(data.loc[tp_condition])
    fn = len(data.loc[fn_condition])
    fp = len(data.loc[fp_condition])
    tn = len(data.loc[tn_condition])

    print('IDS EVALUATION: ')
    print(f'TPR = {tp / (tp + fn)}')
    print(f'FPR = {fp / (fp + tn)}')
    print(f'ACC = {(tp + tn) / (tp + fp + tn + fn)}')

    #    1   0
    # 1  TP  FP
    # 0  FN  TN
    ais_tp_condition = ((data[actual_class_col] == 'normal') & (data[AIS_prediction_col] == 'normal'))
    ais_fn_condition = ((data[actual_class_col] == 'normal') & (data[AIS_prediction_col] != 'normal'))
    ais_fp_condition = ((data[actual_class_col] != 'normal') & (data[AIS_prediction_col] == 'normal'))
    ais_tn_condition = ((data[actual_class_col] != 'normal') & (data[AIS_prediction_col] != 'normal'))

    tp = len(data.loc[ais_tp_condition])
    fn = len(data.loc[ais_fn_condition])
    fp = len(data.loc[ais_fp_condition])
    tn = len(data.loc[ais_tn_condition])

    print('AIS EVALUATION: ')
    print(f'TPR = {tp / (tp + fn)}')
    print(f'FPR = {fp / (fp + tn)}')
    print(f'ACC = {(tp + tn) / (tp + fp + tn + fn)}')

    #    1   0
    # 1  TP  FP
    # 0  FN  TN
    som_tp_condition = ((data[actual_class_col] == 'normal') & (data[SOM_prediction_col] == 'normal'))
    som_fn_condition = ((data[actual_class_col] == 'normal') & (data[SOM_prediction_col] != 'normal'))
    som_fp_condition = ((data[actual_class_col] != 'normal') & (data[SOM_prediction_col] == 'normal'))
    som_tn_condition = ((data[actual_class_col] != 'normal') & (data[SOM_prediction_col] != 'normal'))

    tp = len(data.loc[som_tp_condition])
    fn = len(data.loc[som_fn_condition])
    fp = len(data.loc[som_fp_condition])
    tn = len(data.loc[som_tn_condition])

    print('SOM EVALUATION: ')
    print(f'TPR = {tp / (tp + fn)}')
    print(f'FPR = {fp / (fp + tn)}')
    print(f'ACC = {(tp + tn) / (tp + fp + tn + fn)}')

    

    

if __name__ == '__main__':
    main()