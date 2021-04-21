import pandas as pd
import numpy as np
import itertools


class CART:

    def __init__(self):
        self.nodes = {}
        self.ts = None
        self.gini_root = None

    def fit(self, training_set: pd.DataFrame):
        self.ts = training_set
        self.gini_root = self._calculate_gini_root()
        self.nodes[0] = self.Node(samples=self.ts, left=1, right=2)
        left_set, right_set = self.nodes[0].take_best_split()
        self._build_tree(left_set, right_set)

    def _build_tree(self, split_left, split_right):
        classes = pd.unique(split_left.iloc[:, -1])
        if len(classes) > 1:
            self.nodes[len(self.nodes)] = self.Node(samples=split_left, left=len(self.nodes)+1, right=len(self.nodes)+2)
            left_set, right_set = self.nodes[len(self.nodes)-1].take_best_split()
            self._build_tree(left_set, right_set)
        else:
            self.nodes[len(self.nodes)] = self.Node(samples=split_left, result=classes[0])

        classes = pd.unique(split_right.iloc[:, -1])
        if len(classes) > 1:
            self.nodes[len(self.nodes)] = self.Node(samples=split_right, left=len(self.nodes)+1, right=len(self.nodes)+2)
            left_set, right_set = self.nodes[len(self.nodes)-1].take_best_split()
            self._build_tree(left_set, right_set)
        else:
            self.nodes[len(self.nodes)] = self.Node(samples=split_right, result=classes[0])

    def predict(self, test_set: pd.DataFrame):
        result = []
        for idx in range(test_set.shape[0]):
            current_node = self.nodes[0]
            while current_node.result is None:
                if test_set.dtypes[current_node.attribute] == 'object':
                    if test_set.iloc[idx][current_node.attribute] in current_node.split[0]:
                        current_node = self.nodes[current_node.left]
                    elif test_set.iloc[idx][current_node.attribute] in current_node.split[1]:
                        current_node = self.nodes[current_node.right]
                else:
                    if test_set.iloc[idx][current_node.attribute] <= current_node.split:
                        current_node = self.nodes[current_node.left]
                    elif test_set.iloc[idx][current_node.attribute] > current_node.split:
                        current_node = self.nodes[current_node.right]
            result.append(current_node.result)
        return result

    def _calculate_gini_root(self):
        class_count = list(self.ts.iloc[:, -1:].value_counts())
        return 1 - (np.sum([np.power((i/self.ts.shape[0]), 2) for i in class_count]))

    class Node:

        def __init__(self, samples, left=None, right=None, result=None):
            self.left = left
            self.right = right
            self.samples = samples
            self.result = result
            self.gini = None
            self.attribute = None
            self.split = None

        def __repr__(self):
            return str({'left': self.left,
                        'right': self.right,
                        'gini': self.gini,
                        'attribute': self.attribute,
                        'split': self.split,
                        'class': self.result})

        def take_best_split(self):
            class_column = self.samples.columns[-1]
            possible_labels = np.unique(self.samples.iloc[:, -1])
            best_gini = 1
            best_split = [best_gini, '', (), [], []]

            for column in self.samples.columns[:-1]:
                if self.samples.dtypes[column] == 'object':
                    values = self.samples[column].value_counts()

                    if len(values) == 1:
                        continue

                    frequencies = np.array(values)
                    full_set_names = np.array(values.index)
                    full_set = list(range(len(values)))

                    first_subset = list()
                    for n_comb in range(1, int(len(values)/2) + 1):
                        first_subset.extend([list(subset) for subset in itertools.combinations(full_set, n_comb)])

                    second_subset = [list(set(full_set) - set(subset)) for subset in first_subset]

                    for split in zip(first_subset, second_subset):
                        count = [sum([count for count in frequencies[split[0]]]),
                                 sum([count for count in frequencies[split[1]]])]

                        frequency = [count[0] / self.samples.shape[0], count[1] / self.samples.shape[0]]

                        filters_left = self.samples[column] == full_set_names[split[0][0]]
                        filters_right = self.samples[column] == full_set_names[split[1][0]]

                        for idx in range(1, len(split[0])):
                            filters_left |= self.samples[column] == full_set_names[split[0][idx]]
                        for idx in range(1, len(split[1])):
                            filters_right |= self.samples[column] == full_set_names[split[1][idx]]

                        cf_left = [self.samples[filters_left & (self.samples[class_column] == pl)]
                                   for pl in possible_labels]

                        cf_right = [self.samples[filters_right & (self.samples[class_column] == pl)]
                                    for pl in possible_labels]

                        new_gini = frequency[0] * (1 - (np.sum([np.power((i.shape[0]/count[0]), 2) for i in cf_left]))) + \
                                   frequency[1] * (1 - (np.sum([np.power((i.shape[0]/count[1]), 2) for i in cf_right])))

                        if new_gini < best_split[0]:
                            best_split = [new_gini, column, (list(full_set_names[split[0]]),
                                                             list(full_set_names[split[1]])),
                                          pd.concat(cf_left), pd.concat(cf_right)]

                else:
                    values = sorted(set(self.samples[column]))
                    split_points = [(values[i] + values[i+1])/2 for i in range(len(values) - 1)]

                    for split in split_points:
                        left_samples = self.samples[self.samples[column] <= split]
                        right_samples = self.samples[self.samples[column] > split]
                        count = [left_samples.shape[0], right_samples.shape[0]]

                        frequency = [count[0] / self.samples.shape[0], count[1] / self.samples.shape[0]]

                        cf_left = [left_samples[left_samples[class_column] == pl] for pl in possible_labels]
                        cf_right = [right_samples[right_samples[class_column] == pl] for pl in possible_labels]

                        new_gini = frequency[0] * (1 - (np.sum([np.power((i.shape[0]/count[0]), 2) for i in cf_left]))) + \
                                   frequency[1] * (1 - (np.sum([np.power((i.shape[0]/count[1]), 2) for i in cf_right])))

                        if new_gini < best_split[0]:
                            best_split = [new_gini, column, split, pd.concat(cf_left), pd.concat(cf_right)]

            self.gini = best_split[0]
            self.attribute = best_split[1]
            self.split = best_split[2]
            return best_split[3], best_split[4]

