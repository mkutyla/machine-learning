__author__ = "Miłosz Kutyła, Jakub Ossowski, Jan Walczak, and Patryk Jankowicz"
__credits__ = ["Miłosz Kutyła", "Jakub Ossowski", "Jan Walczak", "Patryk Jankowicz"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Miłosz Kutyła, Jakub Ossowski"
__email__ = "milosz.kutyla.stud@pw.edu.pl, jakub.ossowski.stud@pw.edu.pl"
__status__ = "Production"

import math
import random
import os
import pprint

import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from numba import jit
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import utilities

matplotlib.use('Agg')
pd.set_option('future.no_silent_downcasting', True)

import pickle

@jit(target_backend="cuda")
def som_get_winning_neuron(
                    input_vector: np.array, 
                       x_size, 
                       y_size, 
                       hidden_layer
                       ) -> tuple[int]:
    smallest_dist = calc_euclidean_dist(input_vector, hidden_layer[0, 0])
    winning_neuron = (0, 0)

    for x in range(x_size):
        for y in range(y_size):
            dist = calc_euclidean_dist(input_vector, hidden_layer[x, y])
            if dist < smallest_dist:
                smallest_dist = dist
                winning_neuron = (x, y)

    return winning_neuron

@jit(target_backend='cuda')
def lvq_get_winning_neurons(
                    input_vector: np.array, 
                    input_label,
                    x_size, 
                    y_size, 
                    neuron_map_weights,
                    neuron_flat_labels
                       ):
    
    local_min_dist = -1
    global_min_dist = calc_euclidean_dist(input_vector, neuron_map_weights[0, 0])

    global_x = 0
    global_y = 0
    local_x = -1
    local_y = -1

    for x in range(x_size):
        for y in range(y_size):
            dist = calc_euclidean_dist(input_vector, neuron_map_weights[x, y])
            if dist < global_min_dist:
                global_min_dist = dist
                global_x = x
                global_y = y
            if dist < local_min_dist or local_min_dist < 0:
                if neuron_flat_labels[x * y_size + y] == input_label:
                    local_min_dist = dist
                    local_x = x
                    local_y = y

    return global_x, global_y, local_x, local_y

@jit(target_backend="cuda")
def calc_euclidean_dist(vector1: np.array, vector2: np.array):
    return np.linalg.norm(vector1 - vector2)

@jit(target_backend="cuda")
def som_update_neurons( 
                   vector, 
                   winning_neuron_x: int, 
                   winning_neuron_y: int,
                   adj_radius,
                   learning_rate,
                   x_size,
                   y_size,
                   hidden_layer
                   ):

        for x in range(x_size):
            for y in range(y_size):
                delta = learning_rate * adj((winning_neuron_x, winning_neuron_y), (x, y), adj_radius) * (vector - hidden_layer[x,y])
                hidden_layer[x,y] += delta

        return hidden_layer

@jit(target_backend="cuda")
def lvq_update_neurons(
        global_x: int,
        global_y: int,
        local_x: int,
        local_y: int,
        input_vector,
        neuron_map_weights,
        learning_rate: float,
        ):
    
    if global_x < 0 or global_y < 0:
        return
    
    if global_x == local_x and global_y == local_y:
        delta = learning_rate * (input_vector - neuron_map_weights[global_x, global_y])
        neuron_map_weights[global_x, global_y] += delta

    elif local_x >= 0 and local_y >=0:
        global_delta = learning_rate * (input_vector - neuron_map_weights[global_x, global_y])
        neuron_map_weights[global_x, global_y] -= global_delta
        
        local_delta = learning_rate * (input_vector - neuron_map_weights[local_x, local_y])
        neuron_map_weights[local_x, local_y] += local_delta

@jit(target_backend="cuda")
def adj(
        winning_neuron: tuple[int], 
        other_neuron: tuple[int], 
        adj_radius
        ):
    p = other_neuron[0] - winning_neuron[0]
    q = other_neuron[1] - winning_neuron[1]
    if adj_radius != 0:
        return math.exp(- max(p,q)**2 / adj_radius)

    return 0

@jit(target_backend="cuda")
def get_learning_rate(init_learning_rate, epoch: int, epochs: int):
    return init_learning_rate * (1 - epoch / epochs)

@jit(target_backend="cuda")
def get_adj_radius(init_radius, epoch: int, epochs: int):
    return init_radius * (1 - epoch / epochs)

class SelfOrganizingMap():

    def __init__(self, input_layer_size: int, hidden_layer_dim: list[int]):
        if input_layer_size < 1:
            raise ValueError("Input layer must be a positive integer")

        if len(hidden_layer_dim) != 2:
            raise ValueError("Hidden layer of size x by y has to be declared as a list of the form [x,y]")
        
        if hidden_layer_dim[0] < 1:
            raise ValueError("Hidden layer dimension must be a positive integer")
        
        if hidden_layer_dim[1] < 1:
            raise ValueError("Hidden layer dimension must be a positive integer")
        
        self.input_layer_size   = input_layer_size
        self.hidden_layer_x     = hidden_layer_dim[0]
        self.hidden_layer_y     = hidden_layer_dim[1]

    def init_hidden_layer(self):
        self.hidden_layer = np.random.rand(self.hidden_layer_x, self.hidden_layer_y, self.input_layer_size)
        
    def init_labels(self):
        labels_dict = {label: 0 for label in self.labels}

        self.hidden_layer_labels = [
                                [labels_dict.copy() for _ in range(self.hidden_layer_y)]
                                for _ in range(self.hidden_layer_x)
                            ]
        
    def select_best_labels(self):
        for x in range(self.hidden_layer_x):
            for y in range(self.hidden_layer_y):
                labels = self.hidden_layer_labels[x][y]
                biggest_counter = labels[max(labels, key=labels.get)]
                best_labels = []
                for key in labels.keys():
                    if labels[key] == biggest_counter:
                        best_labels.append(key)

                self.hidden_layer_labels[x][y] = random.choice(best_labels)

    def __fit_silent_mode(
            self,
            X_train: pd.DataFrame, 
            Y_train: pd.DataFrame, 
            epochs: int,
            init_adj_radius: int,
            init_learning_rate: float
            ):
        self.epochs = epochs
        self.labels = Y_train.unique()
        self.init_hidden_layer()
        self.init_labels()

        for e in range(epochs):
            X_train, Y_train = shuffle(X_train, Y_train)
            Y_train = np.array(Y_train)

            self.learning_rate = get_learning_rate(init_learning_rate, e, epochs)
            self.adj_radius = get_adj_radius(init_adj_radius, e, epochs)

            for i in range(len(X_train)):
                x_vector = X_train[i]  
                winning_neuron_x, winning_neuron_y = som_get_winning_neuron(
                                        input_vector = x_vector,
                                        x_size = self.hidden_layer_x,
                                        y_size = self.hidden_layer_y,
                                        hidden_layer = self.hidden_layer
                                        )

                self.hidden_layer = som_update_neurons(
                    vector = x_vector, 
                    winning_neuron_x = winning_neuron_x, 
                    winning_neuron_y = winning_neuron_y,
                    x_size = self.hidden_layer_x,
                    y_size = self.hidden_layer_y,
                    hidden_layer = self.hidden_layer,
                    adj_radius = self.adj_radius,
                    learning_rate = self.learning_rate
                    )

        for i in range(len(X_train)):
            winning_neuron_x, winning_neuron_y = som_get_winning_neuron(
                                    input_vector = X_train[i],
                                    x_size = self.hidden_layer_x,
                                    y_size = self.hidden_layer_y,
                                    hidden_layer = self.hidden_layer
                                    )
            actual_class = Y_train[i]
            self.hidden_layer_labels[winning_neuron_x][winning_neuron_y][actual_class] += 1


        self.select_best_labels()

    def __fit_debug_mode(
            self,
            X_train: pd.DataFrame, 
            Y_train: pd.DataFrame, 
            epochs: int,
            init_adj_radius: int,
            init_learning_rate: float
            ):
        
        self.epochs = epochs
        self.labels = Y_train.unique()
        self.init_hidden_layer()
        self.init_labels()

        for e in range(epochs):
            X_train, Y_train = shuffle(X_train, Y_train)
            Y_train = np.array(Y_train)

            self.learning_rate = get_learning_rate(init_learning_rate, e, epochs)
            self.adj_radius = get_adj_radius(init_adj_radius, e, epochs)

            for i, _ in enumerate(tqdm.trange(len(X_train), desc='{0: <16}'.format(f"Epoch #{e+1}"))):
                x_vector = X_train[i]  
                winning_neuron_x, winning_neuron_y = som_get_winning_neuron(
                                        input_vector = x_vector,
                                        x_size = self.hidden_layer_x,
                                        y_size = self.hidden_layer_y,
                                        hidden_layer = self.hidden_layer
                                        )

                self.hidden_layer = som_update_neurons(
                    vector = x_vector, 
                    winning_neuron_x = winning_neuron_x, 
                    winning_neuron_y = winning_neuron_y,
                    x_size = self.hidden_layer_x,
                    y_size = self.hidden_layer_y,
                    hidden_layer = self.hidden_layer,
                    adj_radius = self.adj_radius,
                    learning_rate = self.learning_rate
                    )

        for i, _ in enumerate(tqdm.trange(len(X_train), desc='{0: <16}'.format('Labeling clusters'))):
            winning_neuron_x, winning_neuron_y = som_get_winning_neuron(
                                    input_vector = X_train[i],
                                    x_size = self.hidden_layer_x,
                                    y_size = self.hidden_layer_y,
                                    hidden_layer = self.hidden_layer
                                    )
            actual_class = Y_train[i]
            self.hidden_layer_labels[winning_neuron_x][winning_neuron_y][actual_class] += 1


        self.select_best_labels()

    def fit(self,
            X_train: pd.DataFrame, 
            Y_train: pd.DataFrame, 
            epochs: int,
            init_adj_radius: int,
            init_learning_rate: float,
            debug: bool = False
            ):
        
        if debug:
            self.__fit_debug_mode(
                X_train = X_train, 
                Y_train = Y_train, 
                epochs = epochs,
                init_adj_radius = init_adj_radius,
                init_learning_rate = init_learning_rate
            )
        else:
            self.__fit_silent_mode(
                X_train = X_train, 
                Y_train = Y_train, 
                epochs = epochs,
                init_adj_radius = init_adj_radius,
                init_learning_rate = init_learning_rate
            )

    def __evaluate_silent_mode(self, 
                       X_test: pd.DataFrame, 
                       Y_test: pd.DataFrame,
                       ids_class: str = 'normal',
                       ) -> dict:
        
        total_positives = 0
        total_negatives = 0

        ids_TP = 0
        ids_TN = 0
        ids_FP = 0
        ids_FN = 0

        Y_test = np.array(Y_test)
        
        for i in range(len(X_test)):
            x_vector = X_test[i]
            winning_neuron_x, winning_neuron_y = som_get_winning_neuron(
                                        input_vector = x_vector,
                                        x_size = self.hidden_layer_x,
                                        y_size = self.hidden_layer_y,
                                        hidden_layer = self.hidden_layer
                                        )
            
            assigned_label = self.hidden_layer_labels[winning_neuron_x][winning_neuron_y]
            actual_label = Y_test[i]

            if assigned_label == actual_label:
                total_positives += 1
            else:
                total_negatives += 1
            
            if actual_label == ids_class:  
            # actual class = positive
                if assigned_label == ids_class:
                    ids_TP += 1
                else:
                    ids_FN += 1
            else: 
            # actual class = negative
                if assigned_label == ids_class:
                    ids_FP += 1
                else:
                    ids_TN += 1

        if total_positives + total_negatives == 0:
            total_acc = -1
        else:
            total_acc = total_positives / (total_positives + total_negatives)
        
        if (ids_TP + ids_FN) == 0:
            ids_tpr = -1
        else:
            ids_tpr = ids_TP / (ids_TP + ids_FN)

        if (ids_FP + ids_TN) == 0:
            ids_fpr = -1
        else:
            ids_fpr = ids_FP / (ids_FP + ids_TN)

        if (ids_TP + ids_FP) == 0:
            ids_ppv = -1
        else:
            ids_ppv = ids_TP / (ids_TP + ids_FP)

        if (ids_TP + ids_FP + ids_TN + ids_FN) == 0:
            ids_acc = -1
        else:
            ids_acc = (ids_TP + ids_TN) / (ids_TP + ids_FP + ids_TN + ids_FN)

        if (2*ids_TP + ids_FP + ids_FN) == 0:
            ids_f1 = -1
        else:
            ids_f1 = 2*ids_TP / (2*ids_TP + ids_FP + ids_FN)

        return {
            'total_acc' : total_acc,
            'ids_tpr'   : ids_tpr,
            'ids_fpr'   : ids_fpr,
            'ids_ppv'   : ids_ppv,
            'ids_acc'   : ids_acc,
            'ids_f1'    : ids_f1,
        }

    def __evaluate_debug_mode(self, 
                       X_test: pd.DataFrame, 
                       Y_test: pd.DataFrame,
                       ids_class: str = 'normal',
                       ) -> dict:
        
        total_positives = 0
        total_negatives = 0

        ids_TP = 0
        ids_TN = 0
        ids_FP = 0
        ids_FN = 0

        Y_test = np.array(Y_test)
        
        for i, _ in enumerate(tqdm.trange(len(X_test), desc='{0: <16}'.format("Testing"))):
            x_vector = X_test[i]
            winning_neuron_x, winning_neuron_y = som_get_winning_neuron(
                                        input_vector = x_vector,
                                        x_size = self.hidden_layer_x,
                                        y_size = self.hidden_layer_y,
                                        hidden_layer = self.hidden_layer
                                        )
            
            assigned_label = self.hidden_layer_labels[winning_neuron_x][winning_neuron_y]
            actual_label = Y_test[i]

            if assigned_label == actual_label:
                total_positives += 1
            else:
                total_negatives += 1
            
            if actual_label == ids_class:  
            # actual class = positive
                if assigned_label == ids_class:
                    ids_TP += 1
                else:
                    ids_FN += 1
            else: 
            # actual class = negative
                if assigned_label == ids_class:
                    ids_FP += 1
                else:
                    ids_TN += 1

        if total_positives + total_negatives == 0:
            total_acc = -1
        else:
            total_acc = total_positives / (total_positives + total_negatives)
        
        if (ids_TP + ids_FN) == 0:
            ids_tpr = -1
        else:
            ids_tpr = ids_TP / (ids_TP + ids_FN)

        if (ids_FP + ids_TN) == 0:
            ids_fpr = -1
        else:
            ids_fpr = ids_FP / (ids_FP + ids_TN)

        if (ids_TP + ids_FP) == 0:
            ids_ppv = -1
        else:
            ids_ppv = ids_TP / (ids_TP + ids_FP)

        if (ids_TP + ids_FP + ids_TN + ids_FN) == 0:
            ids_acc = -1
        else:
            ids_acc = (ids_TP + ids_TN) / (ids_TP + ids_FP + ids_TN + ids_FN)

        if (2*ids_TP + ids_FP + ids_FN) == 0:
            ids_f1 = -1
        else:
            ids_f1 = 2*ids_TP / (2*ids_TP + ids_FP + ids_FN)

        return {
            'total_acc' : total_acc,
            'ids_tpr'   : ids_tpr,
            'ids_fpr'   : ids_fpr,
            'ids_ppv'   : ids_ppv,
            'ids_acc'   : ids_acc,
            'ids_f1'    : ids_f1,
        }

    def evaluate(self, 
                       X_test: pd.DataFrame, 
                       Y_test: pd.DataFrame,
                       ids_class: str = 'normal',
                       debug: bool = False,
                       ) -> dict:
        if debug:
            return self.__evaluate_debug_mode(
                X_test = X_test,
                Y_test = Y_test,
                ids_class = ids_class
            )
        else:
            return self.__evaluate_silent_mode(
                X_test = X_test,
                Y_test = Y_test,
                ids_class = ids_class
            )   

    def __lvq_enforcement_silent_mode(self,
        epochs: int,
        init_learning_rate: float,
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame
        ):
        
        label_dict = dict(zip(self.labels, range(len(self.labels))))
        neuron_flat_labels = np.array([label_dict[item] for sublist in self.hidden_layer_labels for item in sublist])
        Y_train = Y_train.replace(label_dict)

        for e in range(epochs):
            X_train, Y_train = shuffle(X_train, Y_train)
            Y_train = np.array(Y_train)

            learning_rate = get_learning_rate(init_learning_rate, e, epochs)

            for i in range(len(X_train)):
                input_vector = X_train[i]  
                input_label = Y_train[i]
                global_x, global_y, local_x, local_y = lvq_get_winning_neurons(
                        input_vector = input_vector,
                        input_label = input_label,
                        x_size = self.hidden_layer_x,
                        y_size = self.hidden_layer_y,
                        neuron_map_weights = self.hidden_layer,
                        neuron_flat_labels = neuron_flat_labels
                    )
                
                lvq_update_neurons(
                    global_x = global_x,
                    global_y = global_y,
                    local_x = local_x,
                    local_y = local_y,
                    input_vector = input_vector,
                    learning_rate = learning_rate,
                    neuron_map_weights = self.hidden_layer
                )
    
    def __lvq_enforcement_debug_mode(self,
        epochs: int,
        init_learning_rate: float,
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame
        ):

        label_dict = dict(zip(self.labels, range(len(self.labels))))
        neuron_flat_labels = np.array([label_dict[item] for sublist in self.hidden_layer_labels for item in sublist])
        Y_train = Y_train.replace(label_dict)

        for e in range(epochs):
            X_train, Y_train = shuffle(X_train, Y_train)
            Y_train = np.array(Y_train)

            learning_rate = get_learning_rate(init_learning_rate, e, epochs)

            for i, _ in enumerate(tqdm.trange(len(X_train), desc='{0: <16}'.format(f"Epoch #{e+1}"))):
                input_vector = X_train[i]  
                input_label = Y_train[i]
                global_x, global_y, local_x, local_y = lvq_get_winning_neurons(
                        input_vector = input_vector,
                        input_label = input_label,
                        x_size = self.hidden_layer_x,
                        y_size = self.hidden_layer_y,
                        neuron_map_weights = self.hidden_layer,
                        neuron_flat_labels = neuron_flat_labels
                    )
                
                lvq_update_neurons(
                    global_x = global_x,
                    global_y = global_y,
                    local_x = local_x,
                    local_y = local_y,
                    input_vector = input_vector,
                    learning_rate = learning_rate,
                    neuron_map_weights = self.hidden_layer
                )

    def lvq_enforcement(
        self,
        epochs: int,
        init_learning_rate: float,
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame,
        debug: bool = False
        ):
        if debug:
            self.__lvq_enforcement_debug_mode(
                epochs = epochs,
                init_learning_rate = init_learning_rate,
                X_train = X_train,
                Y_train = Y_train,
            )
        else:
            self.__lvq_enforcement_silent_mode(
                epochs = epochs,
                init_learning_rate = init_learning_rate,
                X_train = X_train,
                Y_train = Y_train,
            )
        
    def get_kohonen_map_image(self,
                              name_prefix = '',
                              binary_label_name = 'normal',
                              layers: bool = False):

        output_dir = 'output_images'
        os.makedirs(output_dir, exist_ok=True)

        colors = utilities.COLOR_MAP31
        custom_cmap = mcolors.ListedColormap(colors)
        unique_labels = ['neptune', 'warezclient', 'ipsweep', 'portsweep', 'teardrop', 'nmap', 'satan',
                        'smurf', 'pod', 'back', 'guess_passwd', 'ftp_write', 'multihop', 'rootkit',
                        'buffer_overflow', 'imap', 'warezmaster', 'phf', 'land', 'loadmodule', 'spy',
                        'perl', 'normal']


        label_colors = dict(zip(unique_labels, custom_cmap(np.linspace(0, 1, len(unique_labels)))))

        # for label, color in label_colors.items():
        #     r, g, b = [int(value * 255) for value in color[:3]]
        #     print(f"\033[38;2;{r};{g};{b}m{label}\033[0m")


        kohonen_map_labels_image = np.zeros((self.hidden_layer_x, self.hidden_layer_y, 3), dtype=np.float32)

        for x in range(self.hidden_layer_x):
            for y in range(self.hidden_layer_y):
                label = self.hidden_layer_labels[x][y]
                color = label_colors[label][:3]
                kohonen_map_labels_image[x, y, :] = color

        plt.imshow(kohonen_map_labels_image, vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{name_prefix}_kohonen_map_labels.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        kohonen_map_binary_image = np.zeros((self.hidden_layer_x, self.hidden_layer_y, 3), dtype=np.float32)
        for x in range(self.hidden_layer_x):
            for y in range(self.hidden_layer_y):
                label = self.hidden_layer_labels[x][y]
                if label == binary_label_name:
                    color = np.array([0, 1, 0])
                else:
                    color = np.array([1, 0, 0])
                kohonen_map_binary_image[x, y, :] = color

        plt.imshow(kohonen_map_binary_image, vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{name_prefix}_kohonen_map_attack.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

        if layers:
            kohonen_layer_images = {}
            for label in self.labels:
                kohonen_layer_image = np.zeros((self.hidden_layer_x, self.hidden_layer_y, 3), dtype=np.float32)
                kohonen_layer_images[label] = kohonen_layer_image.copy()
            for x in range(self.hidden_layer_x):
                for y in range(self.hidden_layer_y):
                    neuron_label = self.hidden_layer_labels[x][y]
                    for label in self.labels:
                        if neuron_label == label:
                            color = np.array([0, 1, 0])
                            kohonen_layer_images[label][x, y, :] = color
                        else:
                            color = np.array([0, 0, 0])
                            for deep_label in self.labels:
                                if deep_label != neuron_label:
                                    kohonen_layer_images[deep_label][x, y, :] = color
            for label in self.labels:
                plt.imshow(kohonen_layer_images[label], vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f'{name_prefix}_{label}_kohonen_layer.png'), bbox_inches='tight', pad_inches=0)
                plt.close()

    def predict(self, 
                X_test: pd.DataFrame
                ) -> np.array:
        Y_predicted = []
        # X_test = self.normalizer.transform(X_test)

        for i, _ in enumerate(tqdm.trange(len(X_test), desc='{0: <16}'.format('Predicting'))):
            x_vector = X_test[i]
            winning_neuron_x, winning_neuron_y = som_get_winning_neuron(
                                        input_vector = x_vector,
                                        x_size = self.hidden_layer_x,
                                        y_size = self.hidden_layer_y,
                                        hidden_layer = self.hidden_layer
                                        )
            
            assigned_label = self.hidden_layer_labels[winning_neuron_x][winning_neuron_y]

            Y_predicted.append(assigned_label)
        
        return np.array(Y_predicted)
    
    def load_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            som = pickle.load(f)
        return som
    
    def set_normalizer(self, sc: MinMaxScaler):
        self.normalizer = sc

def split_data(
            data: pd.DataFrame, 
            label_name = 'attack_type',
            sc : MinMaxScaler = None
            ):
    X, y = data.drop([label_name], axis=1), data[label_name]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

    if sc == None:
        sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)

    return X_train, X_test, y_train, y_test, sc

def simulate_ais(
        data: pd.DataFrame,
        AIS_accuracy: float,
        label_name = 'attack_type',
        AIS_label = 'normal'
    ):
    

    print(f'[{utilities.get_timestamp()}] Simulating AIS')
    normal_traffic = data.loc[data[label_name] == AIS_label]
    data = data[data[label_name] != AIS_label]
    normal_traffic = shuffle(normal_traffic)
    normal_traffic, _ = train_test_split(
                                normal_traffic, 
                                train_size=(1-AIS_accuracy), 
                                random_state=42
                                )
    data = pd.concat([data, normal_traffic], ignore_index=True)

    return data

def get_data(
        file_names = 'dataset/nll-kdd/KDD.names',
        file_data = 'dataset/nll-kdd/KDDTrain+.txt',
        to_drop = ['difficulty']
):
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

    return data
    
def debug():
    data = get_data()
    data = simulate_ais(data, AIS_accuracy=0.8)
    X_train, X_test, y_train, y_test, _ = split_data(data)

    print(f'[{utilities.get_timestamp()}] Starting training')
    x_size = 15
    y_size = 15
    som = SelfOrganizingMap(X_train.shape[1], [x_size, y_size])
    som.fit(
            X_train = X_train,
            Y_train = y_train,
            epochs = 10,
            init_adj_radius = max(x_size, y_size) - 1,
            init_learning_rate=1,
            debug = True
            )
    

    print(f'[{utilities.get_timestamp()}] Pure SOM evaluation')
    evaluation = som.evaluate(
                X_test, 
                y_test,
                debug = True
                )
    pprint.pp(evaluation)

    print(f'[{utilities.get_timestamp()}] Starting LVQ-enforcing')
    som.lvq_enforcement(
        epochs = 10,
        init_learning_rate=1,
        X_train = X_train,
        Y_train = y_train,
        debug = True
    )

    print(f'[{utilities.get_timestamp()}] LVQ-enforced SOM evaluation')
    evaluation = som.evaluate(X_test, y_test,
            debug = True)
    pprint.pp(evaluation)

    with open('som.pickle', 'wb') as f:
        pickle.dump(som, f)

    with open('som.pickle', 'rb') as f:
        som2 : SelfOrganizingMap = pickle.load(f)

    evaluation = som2.evaluate(X_test, y_test,
            debug = True)
    pprint.pp(evaluation)
    
    print(f'[{utilities.get_timestamp()}] Kohonen map image creation')
    som.get_kohonen_map_image()

def test_ais_acc():
    start = 0.0
    end = 1.0
    step = 0.05
    x_size = 15
    y_size = 15
    lr = 1
    epochs = 10
    init_adj_radius = max(x_size, y_size) - 1
    ais_tests_dir = 'test/ais_acc'

    total_acc = open(f'{ais_tests_dir}/total_acc.txt', 'w')
    tpr_file = open(f'{ais_tests_dir}/tpr.txt', 'w')
    fpr_file = open(f'{ais_tests_dir}/fpr.txt', 'w')
    ppv_file = open(f'{ais_tests_dir}/ppv.txt', 'w')
    acc_file = open(f'{ais_tests_dir}/acc.txt', 'w')
    f1_file = open(f'{ais_tests_dir}/f1.txt', 'w')

    total_acc.write(f'AIS_ACC SOM LVQ\n')
    tpr_file. write(f'AIS_ACC SOM LVQ\n')
    fpr_file. write(f'AIS_ACC SOM LVQ\n')
    ppv_file. write(f'AIS_ACC SOM LVQ\n')
    acc_file. write(f'AIS_ACC SOM LVQ\n')
    f1_file.  write(f'AIS_ACC SOM LVQ\n')


    for i in range(int((end - start + step) / step)):
        AIS_acc = start + step*i
        if AIS_acc <= 0:
            AIS_acc = 0.01
        if AIS_acc >= 1:
            AIS_acc = 0.99
        AIS_acc = round(AIS_acc, 2)
        print(f'AIS_acc = {AIS_acc}')

        data = get_data()
        try:
            data = simulate_ais(data, AIS_accuracy=AIS_acc)
            X_train, X_test, y_train, y_test, _ = split_data(data)
        except Exception as e:
            print(f'An exception occured {e}')
            break

        print(f'[{utilities.get_timestamp()}] Starting training')
        som = SelfOrganizingMap(X_train.shape[1], [x_size, y_size])
        som.fit(
                X_train = X_train,
                Y_train = y_train,
                epochs = epochs,
                init_adj_radius = init_adj_radius,
                init_learning_rate = lr
            )
        

        print(f'[{utilities.get_timestamp()}] SOM evaluation')
        som_evaluation = som.evaluate(X_test, y_test)

        print(f'[{utilities.get_timestamp()}] LVQ-enforcing')
        som.lvq_enforcement(
            epochs = epochs,
            init_learning_rate = lr,
            X_train = X_train,
            Y_train = y_train
        )

        print(f'[{utilities.get_timestamp()}] LVQ evaluation')
        lvq_evaluation = som.evaluate(X_test, y_test)
        total_acc.write(f'{AIS_acc} {som_evaluation['total_acc']} {lvq_evaluation['total_acc']}\n')
        tpr_file. write(f'{AIS_acc} {som_evaluation['ids_tpr']  } {lvq_evaluation['ids_tpr']  }\n')
        fpr_file. write(f'{AIS_acc} {som_evaluation['ids_fpr']  } {lvq_evaluation['ids_fpr']  }\n')
        ppv_file. write(f'{AIS_acc} {som_evaluation['ids_ppv']  } {lvq_evaluation['ids_ppv']  }\n')
        acc_file. write(f'{AIS_acc} {som_evaluation['ids_acc']  } {lvq_evaluation['ids_acc']  }\n')
        f1_file.  write(f'{AIS_acc} {som_evaluation['ids_f1']   } {lvq_evaluation['ids_f1']   }\n')

        print(f'[{utilities.get_timestamp()}] Kohonen map image creation')
        som.get_kohonen_map_image(name_prefix=f'{AIS_acc}')


    total_acc.close()
    tpr_file.close()
    fpr_file.close()
    ppv_file.close()
    acc_file.close()
    f1_file.close()

def test_som_size():
    x_size = 15
    y_size = 15

    lr = 1
    AIS_acc = 0.8
    epochs = 10
    init_adj_radius = max(x_size, y_size) - 1
    ais_tests_dir = 'test/size'

    total_acc = open(f'{ais_tests_dir}/total_acc.txt', 'w')
    tpr_file = open(f'{ais_tests_dir}/tpr.txt', 'w')
    fpr_file = open(f'{ais_tests_dir}/fpr.txt', 'w')
    ppv_file = open(f'{ais_tests_dir}/ppv.txt', 'w')
    acc_file = open(f'{ais_tests_dir}/acc.txt', 'w')
    f1_file = open(f'{ais_tests_dir}/f1.txt', 'w')

    total_acc.write(f'x y SOM LVQ\n')
    tpr_file. write(f'x y SOM LVQ\n')
    fpr_file. write(f'x y SOM LVQ\n')
    ppv_file. write(f'x y SOM LVQ\n')
    acc_file. write(f'x y SOM LVQ\n')
    f1_file.  write(f'x y SOM LVQ\n')
    X_Y_pairs = [
        (1,1), (2,2), (3,3),
        (4,4), (5,5), (6,6),
        (7,7), (8,8), (9,9),
        (10,10), (11,11), (12,12),
        (13,13), (14,14), (15,15),
        (50, 50), (100, 100)
    ]

    for i in range(len(X_Y_pairs)):
        data = get_data()
        data = simulate_ais(data, AIS_accuracy=AIS_acc)
        X_train, X_test, y_train, y_test, _ = split_data(data)

        x_size, y_size = X_Y_pairs[i]
        print(f'(x,y) = ({x_size},{y_size})')
        print(f'[{utilities.get_timestamp()}] Starting training')

        som = SelfOrganizingMap(X_train.shape[1], [x_size, y_size])
        som.fit(
                X_train = X_train,
                Y_train = y_train,
                epochs = epochs,
                init_adj_radius = init_adj_radius,
                init_learning_rate = lr,
                debug = True
            )
        

        print(f'[{utilities.get_timestamp()}] SOM evaluation')
        som_evaluation = som.evaluate(X_test, y_test)

        print(f'[{utilities.get_timestamp()}] LVQ-enforcing')
        som.lvq_enforcement(
            epochs = epochs,
            init_learning_rate = lr,
            X_train = X_train,
            Y_train = y_train
        )

        print(f'[{utilities.get_timestamp()}] LVQ evaluation')
        lvq_evaluation = som.evaluate(X_test, y_test)

        area = x_size*y_size
        total_acc.write(f'{area} {som_evaluation['total_acc']} {lvq_evaluation['total_acc']}\n')
        tpr_file. write(f'{area} {som_evaluation['ids_tpr']  } {lvq_evaluation['ids_tpr']  }\n')
        fpr_file. write(f'{area} {som_evaluation['ids_fpr']  } {lvq_evaluation['ids_fpr']  }\n')
        ppv_file. write(f'{area} {som_evaluation['ids_ppv']  } {lvq_evaluation['ids_ppv']  }\n')
        acc_file. write(f'{area} {som_evaluation['ids_acc']  } {lvq_evaluation['ids_acc']  }\n')
        f1_file.  write(f'{area} {som_evaluation['ids_f1']   } {lvq_evaluation['ids_f1']   }\n')

        print(f'[{utilities.get_timestamp()}] Kohonen map image creation')
        som.get_kohonen_map_image(name_prefix=f'size/{x_size}x{y_size}')


    total_acc.close()
    tpr_file.close()
    fpr_file.close()
    ppv_file.close()
    acc_file.close()
    f1_file.close()

def generate_mega_som():
    data = get_data()
    data = simulate_ais(data, AIS_accuracy=0.7)
    X_train, X_test, y_train, y_test, sc = split_data(data)

    print(f'[{utilities.get_timestamp()}] Starting training')
    x_size = 150
    y_size = 150
    som = SelfOrganizingMap(X_train.shape[1], [x_size, y_size])
    som.set_normalizer(sc)
    som.fit(
            X_train = X_train,
            Y_train = y_train,
            epochs = 10,
            init_adj_radius = max(x_size, y_size) - 1,
            init_learning_rate = 1,
            debug = True
            )
    

    print(f'[{utilities.get_timestamp()}] Pure SOM evaluation')
    som_evaluation = som.evaluate(
                X_test, 
                y_test,
                debug = True
                )
    pprint.pp(som_evaluation)
    with open('som_giga_without_lvq.pickle', 'wb') as f:
        pickle.dump(som, f)

    print(f'[{utilities.get_timestamp()}] Starting LVQ-enforcing')
    som.lvq_enforcement(
        epochs = 10,
        init_learning_rate=1,
        X_train = X_train,
        Y_train = y_train,
        debug = True
    )

    print(f'[{utilities.get_timestamp()}] LVQ-enforced SOM evaluation')
    lvq_evaluation = som.evaluate(X_test, y_test)
    pprint.pp(lvq_evaluation)

    with open('som_giga_with_lvq.pickle', 'wb') as f:
        pickle.dump(som, f)

    with open('giga_som_evaluation.txt', 'w') as file:
        file.write(f'{som_evaluation['total_acc']} {lvq_evaluation['total_acc']}\n')
        file. write(f'{som_evaluation['ids_tpr']  } {lvq_evaluation['ids_tpr']  }\n')
        file. write(f'{som_evaluation['ids_fpr']  } {lvq_evaluation['ids_fpr']  }\n')
        file. write(f'{som_evaluation['ids_ppv']  } {lvq_evaluation['ids_ppv']  }\n')
        file. write(f'{som_evaluation['ids_acc']  } {lvq_evaluation['ids_acc']  }\n')
        file.  write(f'{som_evaluation['ids_f1']   } {lvq_evaluation['ids_f1']   }\n')


    with open('som_giga_with_lvq.pickle', 'rb') as f:
        som2 : SelfOrganizingMap = pickle.load(f)

    evaluation = som2.evaluate(X_test, y_test)
    pprint.pp(evaluation)

    
    print(f'[{utilities.get_timestamp()}] Kohonen map image creation')
    som.get_kohonen_map_image(name_prefix=f'{x_size}x{y_size}')

def test_pickle():
    with open('pickles/som_150x150_lvq.pickle', 'rb') as f:
        som : SelfOrganizingMap = pickle.load(f)
    
    som.get_kohonen_map_image(
        layers = True,
        name_prefix='150x150'
    )

def lvq_giga_som():
    data = get_data()
    data = simulate_ais(data, AIS_accuracy=0.7)
    x_size = 150
    y_size = 150

    with open('som_giga_without_lvq.pickle', 'rb') as f:
        som : SelfOrganizingMap = pickle.load(f)

    print(f'[{utilities.get_timestamp()}] Kohonen map image creation')
    som.get_kohonen_map_image(name_prefix=f'{x_size}x{y_size}')
    
    X_train, X_test, y_train, y_test, _ = split_data(data, sc = som.normalizer)
    som.lvq_enforcement(
        epochs = 10,
        init_learning_rate=1,
        X_train = X_train,
        Y_train = y_train,
        debug = True
    )
    with open('som_giga_with_lvq.pickle', 'wb') as f:
        pickle.dump(som, f)

    print(f'[{utilities.get_timestamp()}] LVQ-enforced SOM evaluation')
    lvq_evaluation = som.evaluate(X_test, y_test)
    pprint.pp(lvq_evaluation)

if __name__ == '__main__':
    # test_ais_acc()
    # test_som_size()
    # generate_mega_som()
    # test_pickle()
    # lvq_giga_som()
    debug()
