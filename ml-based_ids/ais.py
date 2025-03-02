__author__ = "Miłosz Kutyła, Jakub Ossowski, Jan Walczak, and Patryk Jankowicz"
__credits__ = ["Miłosz Kutyła", "Jakub Ossowski", "Jan Walczak", "Patryk Jankowicz"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Miłosz Kutyła, Jakub Ossowski"
__email__ = "milosz.kutyla.stud@pw.edu.pl, jakub.ossowski.stud@pw.edu.pl"
__status__ = "Production"

import pickle

import tqdm
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import utilities

pd.set_option('future.no_silent_downcasting', True)

from numba import jit

@jit(target_backend='cuda')
def is_malicious(x_vector, detectors, threshold):
    for detector in detectors:
        if calc_euclidean_dist(x_vector, detector) < threshold: 
                return True
    return False

@jit(target_backend="cuda")
def generate_detector(length) -> np.array:
    return np.random.rand(length)

@jit(target_backend="cuda")
def calc_euclidean_dist(vector1: np.array, vector2: np.array):
    return np.linalg.norm(vector1 - vector2)

@jit(target_backend="cuda")
def negative_selection(dataset: np.array, no_detectors, n_features, threshold) -> np.array:
    detectors = []
    changed = True

    counter = 0
    while len(detectors) != no_detectors:
        if changed and len(detectors)%10 == 0:
            print(f"Generated {len(detectors)} / {no_detectors} in {counter} iterations")
            counter = 0
            changed = False
        detector = generate_detector(n_features) 
        counter += 1
        
        # for i, _ in enumerate(tqdm.trange(len(dataset), desc='{0: <16}'.format('Training'))):
        #     if calc_euclidean_dist(dataset[i], detector) < threshold:    
        #         matched = True
        #         break
        # if not matched:
        #     detectors.append(detector)
        
        matched = False
        for item in dataset:
            if calc_euclidean_dist(item, detector) < threshold:    
                matched = True
                break

        # if not matched:
        #     detectors.append(detector)
        #     changed = True    
        
        if not matched:
            close = False
            for item in detectors:
                if calc_euclidean_dist(item, detector) < threshold / 2:  
                    close = True
                    break
            if not close:
                detectors.append(detector)
                changed = True

    return detectors

    
class ArtificialImmuneSystem():

    def __init__(self, no_detectors: int, threshold: int):
        if no_detectors < 1:
            raise ValueError("Number of detectors must be a positive integer")

        if threshold < 0:
            raise ValueError("Threshold must be be a positive integer")
        
        # if threshold > 1000:
        #     raise ValueError("Threshold must be between [0, 1]")
        
        self.no_detectors   = no_detectors
        self.threshold      = threshold


    def fit(self,
            X_train: pd.DataFrame
            ):
        

        self.detectors = negative_selection(dataset=X_train, no_detectors=self.no_detectors,  
                                           n_features=len(X_train[0]), threshold=self.threshold)
           

    def check_accuracy(self, 
                       X_test: pd.DataFrame, 
                       Y_test: pd.DataFrame,
                       ) -> float:
        hit = 0
        attack_not_detected = 0
        false_positive = 0
        attacks = 0
        normal_traffic = 0
        Y_test = np.array(Y_test)
        
        for i, _ in enumerate(tqdm.trange(len(X_test), desc='{0: <16}'.format('Testing'))):
            x_vector = X_test[i]
            
            matched = False
            for detector in self.detectors:
                if calc_euclidean_dist(x_vector, detector) < self.threshold: 
                     assigned_label = 'malicious'
                     matched = True
                     break

            if not matched:
                assigned_label = 'normal'  

            actual_label = Y_test[i]

            if actual_label == 'normal':
                normal_traffic += 1
            else:
                attacks += 1

            if actual_label == 'normal' and assigned_label == 'normal':
                hit += 1
            elif actual_label != 'normal' and assigned_label == 'malicious':
                hit += 1
            else:
                if actual_label != 'normal' and assigned_label == 'normal':
                    attack_not_detected +=1
                elif actual_label == 'normal':
                     false_positive += 1

        print(f'Overall accuracy:       {hit/len(X_test)}')
        print(f'Attacks not detected:   {attack_not_detected/attacks}')
        print(f'False positives:        {false_positive/normal_traffic}')
        return hit/len(X_test)


    def predict(self, X_test: pd.DataFrame) -> np.array:
        Y_predicted = []
        # X_test = self.normalizer.transform(X_test)
        detectors = np.array(self.detectors)
        for i, _ in enumerate(tqdm.trange(len(X_test), desc='{0: <16}'.format('Testing'))):
            x_vector = X_test[i]
            
            if is_malicious(x_vector, detectors, self.threshold):
                assigned_label = 'malicious'
            else:
                assigned_label = 'normal'

            Y_predicted.append(assigned_label)
        
        return np.array(Y_predicted)
     
    def load_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            ais = pickle.load(f)
        return ais
    
    def set_normalizer(self, sc: MinMaxScaler):
        self.normalizer = sc
        

def main():

    # file_names = 'dataset/iris.data/iris.names'
    # file_train_data = 'dataset/iris.data/iris.data'
    # file_test_data = 'dataset/iris.data/iris.data'
    # label_name = 'class'
    # to_drop = []

    # file_names = 'dataset/kddcup.names'
    # file_data = 'dataset/kddcup.data_10_percent/kddcup.data_10_percent'
    # label_name = 'attack_type'
    # to_drop = []

    file_names = 'dataset/nll-kdd/KDD.names'
    file_data = 'dataset/nll-kdd/KDDTrain+.txt'
    label_name = 'attack_type'
    to_drop = ['difficulty']

    pandas_names = []
    features = {}

    print(f'[{utilities.get_timestamp()}] Reading .names file')
    with open(file_names, 'r') as file:
        for line in file.readlines():
            feature, feature_type = line.strip().split(':')
            feature_type = feature_type.strip().replace('.','')
            features[feature] = feature_type
            pandas_names.append(feature)

    print(f'[{utilities.get_timestamp()}] Reading data')
    data = pd.read_csv(
        file_data,   
        names = pandas_names
        )


    print(f'[{utilities.get_timestamp()}] Preprocessing the data')
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

    normal_traffic = data.loc[data[label_name] == 'normal'] 
    sus_traffic = data.loc[data[label_name] != 'normal'] 
    
    normal_training, normal_testing = train_test_split(normal_traffic, test_size=0.2, random_state=42)

    testing = pd.concat([sus_traffic, normal_testing ], ignore_index=True)

    X_train, y_train = normal_training.drop([label_name], axis=1), normal_training[label_name]
    X_test, y_test = testing.drop([label_name], axis=1), testing[label_name]

    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)


    ais = ArtificialImmuneSystem(no_detectors=500, threshold=4.1)

    ais.fit(X_train)
    ais.set_normalizer(sc)

    print(f'[{utilities.get_timestamp()}] Starting testing')

    ais.check_accuracy(X_test, y_test)

    with open('aisPickle', 'wb') as pickle_file:
        pickle.dump(ais, pickle_file)

if __name__ == "__main__":
    main()