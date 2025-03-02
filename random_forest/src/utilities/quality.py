from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Quality:
    def __init__(self, classes: list, expected: list = None, predicted: list = None) -> None:
        self.classes = map(str, classes)
        self.expected = [str(x) for x in expected] if expected is not None else expected
        self.predicted = [str(x) for x in predicted] if predicted is not None else predicted

        self.confusion_matrix = None

    def set_expected(self, expected):
        self.expected = [str(x) for x in expected]

    def set_predicted(self, predicted):
        self.predicted = [str(x) for x in predicted]

    def check_init(self):
        if self.classes is None:
            raise Exception('[!] this.classes is None')
        if self.expected is None:
            raise Exception('[!] this.expected is None')
        if self.predicted is None:
            raise Exception('[!] this.predicted is None')

    def check_conf_matrix_init(self):
        if self.confusion_matrix is None:
            raise Exception('[!] this.confusion_matrix is None')

    def check_full_init(self):
        self.check_init()
        self.check_conf_matrix_init()

    def get_tp(self, cls) -> int:
        self.check_full_init()
        cls = str(cls)

        return self.confusion_matrix[cls][cls]

    def get_fp(self, cls) -> int:
        self.check_full_init()
        cls = str(cls)
        fp = 0

        for c in self.classes:
            if c == cls:
                continue
            fp += self.confusion_matrix[c][cls]
        return fp

    def get_fn(self, cls) -> int:
        self.check_full_init()
        cls = str(cls)

        fn = sum(self.confusion_matrix[cls].values()) - self.get_tp(cls)
        return fn

    def get_tn(self, cls: str) -> int:
        self.check_full_init()

        result = self.sum_conf_matrix()
        result -= self.get_fp(cls)
        result -= self.get_fn(cls)
        result -= self.get_tp(cls)

        return result

    def sum_conf_matrix(self):
        self.check_full_init()

        result = 0
        for c in self.classes:
            result += sum(self.confusion_matrix[c].values())
        return result

    def get_tpr(self, cls: str, tp: int = -1, fn: int = -1):
        if tp < 0:
            tp = self.get_tp(cls)
        if fn < 0:
            fn = self.get_fn(cls)
        return tp / (tp + fn)

    def get_tnr(self, cls: str, tn: int = -1, fp: int = -1):
        if tn < 0:
            tn = self.get_tp(cls)
        if fp < 0:
            fp = self.get_fp(cls)
        return tn / (fp + tn)

    def get_ppv(self, cls: str, tp: int = -1, fp: int = -1):
        if tp < 0:
            tp = self.get_tp(cls)
        if fp < 0:
            fp = self.get_fp(cls)
        return tp / (tp + fp)

    def get_acc(self, cls: str, tp: int = -1, tn: int = -1, fp: int = -1, fn: int = -1):
        if tp < 0:
            tp = self.get_tp(cls)
        if tn < 0:
            tn = self.get_tn(cls)
        if fp < 0:
            fp = self.get_fp(cls)
        if fn < 0:
            fn = self.get_fn(cls)
        return (tp + tn) / (tp + fp + tn + fn)

    def get_f1(self, cls: str, tp: int = -1, fp: int = -1, fn: int = -1):
        if tp < 0:
            tp = self.get_tp(cls)
        if fp < 0:
            fp = self.get_fp(cls)
        if fn < 0:
            fn = self.get_fn(cls)
        return 2 * tp / (2 * tp + fp + fn)

    def get_class_measures(self, cls: str):
        tp = self.get_tp(cls)
        tn = self.get_tn(cls)
        fp = self.get_fp(cls)
        fn = self.get_fn(cls)

        return {
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "TPR": self.get_tpr(cls, tp=tp, fn=fn),
            "TNR": self.get_tnr(cls, tn=tn, fp=fp),
            "PPV": self.get_ppv(cls, tp=tp, fp=fp),
            "ACC": self.get_acc(cls, tp=tp, tn=tn, fp=fp, fn=fn),
            "F1": self.get_f1(cls, tp=tp, fp=fp, fn=fn)
        }

    def get_measures(self):
        self.check_init()
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        macro_f1 = 0
        weighted_f1 = 0

        for cls in self.classes:
            _tp = self.get_tp(cls)
            _tn = self.get_tn(cls)
            _fp = self.get_fn(cls)
            _fn = self.get_fp(cls)

            tp += _tp
            tn += _tn
            fp += _fp
            fn += _fn

            _f1 = self.get_f1(cls, tp=_tp, fp=_fp, fn=_fn)
            macro_f1 += _f1
            weighted_f1 += self.expected.count(cls) * _f1

        cls = ""
        micro_f1 = self.get_f1(cls, tp=tp, fp=fp, fn=fn)
        macro_f1 = macro_f1 / len(self.expected)
        weighted_f1 = weighted_f1 / len(self.expected)

        return {
            "TPR": self.get_tpr(cls, tp=tp, fn=fn),
            "TNR": self.get_tnr(cls, tn=tn, fp=fp),
            "PPV": self.get_ppv(cls, tp=tp, fp=fp),
            "ACC": self.get_acc(cls, tp=tp, tn=tn, fp=fp, fn=fn),
            "Micro F1": micro_f1,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1
        }

    #
    # Confusion matrix handlers
    #

    def get_confusion_matrix(self):
        self.check_init()

        self.classes = sorted(self.classes)

        class_dict = {}
        for c in self.classes:
            class_dict[c] = 0

        matrix = {}
        for c in self.classes:
            matrix[c] = class_dict.copy()

        for co, pr in zip(self.expected, self.predicted):
            if pr == 'None':
                continue
            matrix[co][pr] += 1

        self.confusion_matrix = matrix
        return matrix

    def get_printable_conf_matrix(self) -> list:
        self.check_conf_matrix_init()

        first_row = [" "]
        for c in self.classes:
            first_row.append(c)

        class_dicts = self.confusion_matrix
        class_dicts = OrderedDict(sorted(class_dicts.items()))

        matrix = [first_row]
        for c in self.classes:
            row = [c]
            class_dicts[c] = OrderedDict(sorted(class_dicts[c].items()))
            data = list(class_dicts[c].values())
            row.extend(np.array(data))
            matrix.append(row)

        return matrix

    def get_drawable_conf_matrix(self) -> list:
        self.check_conf_matrix_init()
        class_dicts = self.confusion_matrix
        class_dicts = OrderedDict(sorted(class_dicts.items()))

        matrix = []
        for c in self.classes:
            row = []
            class_dicts[c] = OrderedDict(sorted(class_dicts[c].items()))
            data = list(class_dicts[c].values())
            row.extend(np.array(data))
            matrix.append(row)

        return matrix

    def print_confusion_matrix(self):
        matrix = self.get_printable_conf_matrix()
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = "\t".join("{{:{}}}".format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print("\n".join(table))

    def draw_confusion_matrix(self,
                              title: str = "Confusion matrix",
                              filename: str = "output/confusion_matrix.png"):

        cm = confusion_matrix(self.expected, self.predicted)
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=self.classes, yticklabels=self.classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(title)
        plt.savefig(filename)

    @staticmethod
    def plot_roc(x: list, y: list, labels: list = list()):

        plt.scatter(x[:-1], y[:-1])
        plt.plot(x[-1], y[-1], 'ro')

        if len(labels) != 0:
            for i, txt in enumerate(labels):
                plt.annotate(txt, (x[i], y[i]))

        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        plt.show()
