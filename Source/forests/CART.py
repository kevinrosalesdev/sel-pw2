import pandas as pd
import numpy as np
import itertools


class CART:

    def __init__(self):
        self.nodes = {0: self.Node()}
        pass

    def fit(self, training_set: pd.DataFrame):
        self.ts = training_set
        self._calculate_gini_root()
        self._take_best_split()

    def predict(self, test_set: pd.DataFrame):
        pass

    def _calculate_gini_root(self):
        class_count = list(self.ts.iloc[:, -1:].value_counts())
        self.nodes[0].gini = 1 - (np.sum([np.power((i/self.ts.shape[0]), 2) for i in class_count]))

    def _calculate_gini(self, node):
        node.gini = 0

    def _take_best_split(self):
        class_column = self.ts.columns[-1]
        possible_labels = np.unique(self.ts.iloc[:, -1])

        for column in self.ts.columns[:-1]:
            values = self.ts[column].value_counts()
            frequencies = np.array(values)
            full_set_names = list(values.index)
            full_set = list(range(len(values)))

            first_subset = list()
            for n_comb in range(1, int(len(values)/2) + 1):
                first_subset.extend([list(subset) for subset in itertools.combinations(full_set, n_comb)])

            second_subset = [list(set(full_set) - set(subset)) for subset in first_subset]

            gini = list()
            for split in zip(first_subset, second_subset):
                count = [sum([count for count in frequencies[split[0]]]),
                         sum([count for count in frequencies[split[1]]])]

                frequency = [count[0] / self.ts.shape[0], count[1] / self.ts.shape[0]]

                filters_left = self.ts[column] == full_set_names[split[0][0]]
                filters_right = self.ts[column] == full_set_names[split[1][0]]

                for idx in range(1, len(split[0])):
                    filters_left |= self.ts[column] == full_set_names[split[0][idx]]
                for idx in range(1, len(split[1])):
                    filters_right |= self.ts[column] == full_set_names[split[1][idx]]

                cf_left = [self.ts[filters_left & (self.ts[class_column] == pl)].shape[0]
                           for pl in possible_labels]

                cf_right = [self.ts[filters_right & (self.ts[class_column] == pl)].shape[0]
                            for pl in possible_labels]

                gini.append(frequency[0] * (1 - (np.sum([np.power((i/count[0]), 2) for i in cf_left]))) +
                            frequency[1] * (1 - (np.sum([np.power((i/count[1]), 2) for i in cf_right]))))

            print(gini)

    class Node:

        def __init__(self):
            self.left = None
            self.right = None
            self.gini = None
