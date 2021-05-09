import pandas as pd
import numpy as np
import itertools


class CART:

    def __init__(self, threshold_ratio=0.05):
        self.nodes = {}
        self.ts = None
        self.feat_freq = {}
        self.gini_root = None
        self.f = None
        self.threshold_ratio = threshold_ratio

    def fit(self, training_set: pd.DataFrame, f=None):
        self.ts = training_set
        for column in self.ts.columns[:-1]:
            self.feat_freq[column] = 0

        if f is None:
            self.f = training_set.shape[1] - 1
        else:
            self.f = f

        self.gini_root = self._calculate_gini_root()
        classes = pd.unique(self.ts.iloc[:, -1])
        if len(classes) == 1:
            self.nodes['r'] = self.Node(samples=self.ts, f=self.f, result=classes[0])
        else:
            self.nodes['r'] = self.Node(samples=self.ts, f=self.f, left='rl', right='rr')
            left_set = None
            while left_set is None:
                left_set, right_set = self.nodes['r'].take_best_split()

            self.feat_freq[self.nodes['r'].attribute] += 1
            self._build_tree(left_set, right_set, 'r', 'rl', 'rr')

    def _build_tree(self, split_left, split_right, parent_idx, left_idx, right_idx):
        classes = pd.unique(split_left.iloc[:, -1])
        if len(classes) > 1:
            new_left_idx = f"{left_idx}l"
            new_right_idx = f"{left_idx}r"
            self.nodes[left_idx] = self.Node(samples=split_left, f=self.f,
                                             left=new_left_idx, right=new_right_idx)
            left_set, right_set = self.nodes[left_idx].take_best_split()
            if left_set is not None \
                    and self.nodes[left_idx].gini < self.nodes[parent_idx].gini * (1 - self.threshold_ratio):
                self.feat_freq[self.nodes[left_idx].attribute] += 1
                self._build_tree(left_set, right_set, left_idx, new_left_idx, new_right_idx)
            else:
                self.nodes[left_idx].left = None
                self.nodes[left_idx].right = None
                self.nodes[left_idx].attribute = None
                self.nodes[left_idx].split = None
                self.nodes[left_idx].result = self.nodes[left_idx].samples.iloc[:, -1].value_counts().index[0]

            # else:
            #     self.nodes[left_idx].gini = self.nodes[parent_idx].gini
            #     self.nodes[left_idx].attribute = self.nodes[parent_idx].attribute
            #     self.nodes[left_idx].split = self.nodes[parent_idx].split
            #     self._build_tree(split_left, split_right, left_idx, new_left_idx, new_right_idx)
        else:
            self.nodes[left_idx] = self.Node(samples=split_left, f=self.f, result=classes[0])

        classes = pd.unique(split_right.iloc[:, -1])
        if len(classes) > 1:
            new_left_idx = f"{right_idx}l"
            new_right_idx = f"{right_idx}r"
            self.nodes[right_idx] = self.Node(samples=split_right, f=self.f,
                                              left=new_left_idx, right=new_right_idx)
            left_set, right_set = self.nodes[right_idx].take_best_split()
            if left_set is not None \
                    and self.nodes[right_idx].gini < self.nodes[parent_idx].gini * (1 - self.threshold_ratio):
                self.feat_freq[self.nodes[right_idx].attribute] += 1
                self._build_tree(left_set, right_set, right_idx, new_left_idx, new_right_idx)
            else:
                self.nodes[right_idx].left = None
                self.nodes[right_idx].right = None
                self.nodes[right_idx].attribute = None
                self.nodes[right_idx].split = None
                self.nodes[right_idx].result = self.nodes[right_idx].samples.iloc[:, -1].value_counts().index[0]

            # else:
            #     self.nodes[right_idx].gini = self.nodes[parent_idx].gini
            #     self.nodes[right_idx].attribute = self.nodes[parent_idx].attribute
            #     self.nodes[right_idx].split = self.nodes[parent_idx].split
            #     self._build_tree(split_left, split_right, right_idx, new_left_idx, new_right_idx)
        else:
            self.nodes[right_idx] = self.Node(samples=split_right, f=self.f, result=classes[0])

    def predict(self, test_set: pd.DataFrame):
        result = []
        for idx in range(test_set.shape[0]):
            current_node = self.nodes['r']
            while current_node.result is None:
                if test_set.dtypes[current_node.attribute] == 'object':
                    if test_set.iloc[idx][current_node.attribute] in current_node.split[0]:
                        current_node = self.nodes[current_node.left]
                    elif test_set.iloc[idx][current_node.attribute] in current_node.split[1]:
                        current_node = self.nodes[current_node.right]
                    else:
                        break
                else:
                    if test_set.iloc[idx][current_node.attribute] <= current_node.split:
                        current_node = self.nodes[current_node.left]
                    elif test_set.iloc[idx][current_node.attribute] > current_node.split:
                        current_node = self.nodes[current_node.right]
            result.append(current_node.result)
        return result

    def _calculate_gini_root(self):
        class_count = list(self.ts.iloc[:, -1:].value_counts())
        return 1 - (np.sum([np.power((i / self.ts.shape[0]), 2) for i in class_count]))

    class Node:

        def __init__(self, samples, f, left=None, right=None, result=None):
            self.left = left
            self.right = right
            self.samples = samples
            self.result = result
            self.gini = None
            self.attribute = None
            self.split = None
            self.f = f

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
            best_split = [1, '', (), [], []]
            if isinstance(self.f, list):
                random_features = self.f
            else:
                random_features = []
                while len(random_features) < self.f:
                    random_feature = np.random.random_integers(low=0, high=self.samples.shape[1] - 2)
                    if random_feature not in random_features:
                        random_features.append(random_feature)

            for column in self.samples.columns[random_features]:
                if self.samples.dtypes[column] == 'object':
                    values = self.samples[column].value_counts()

                    if len(values) == 1:
                        continue

                    frequencies = np.array(values)
                    full_set_names = np.array(values.index)
                    full_set = list(range(len(values)))

                    first_subset = list()
                    for n_comb in range(1, int(len(values) / 2) + 1):
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

                        new_gini = frequency[0] * (
                                1 - (np.sum([np.power((i.shape[0] / count[0]), 2) for i in cf_left]))) + \
                                   frequency[1] * (
                                           1 - (np.sum([np.power((i.shape[0] / count[1]), 2) for i in cf_right])))

                        if new_gini < best_split[0]:
                            best_split = [new_gini, column, (list(full_set_names[split[0]]),
                                                             list(full_set_names[split[1]])),
                                          pd.concat(cf_left), pd.concat(cf_right)]

                else:
                    values = sorted(set(self.samples[column]))
                    split_points = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

                    for split in split_points:
                        left_samples = self.samples[self.samples[column] <= split]
                        right_samples = self.samples[self.samples[column] > split]
                        count = [left_samples.shape[0], right_samples.shape[0]]

                        frequency = [count[0] / self.samples.shape[0], count[1] / self.samples.shape[0]]

                        cf_left = [left_samples[left_samples[class_column] == pl] for pl in possible_labels]
                        cf_right = [right_samples[right_samples[class_column] == pl] for pl in possible_labels]

                        new_gini = frequency[0] * (
                                1 - (np.sum([np.power((i.shape[0] / count[0]), 2) for i in cf_left]))) + \
                                   frequency[1] * (
                                           1 - (np.sum([np.power((i.shape[0] / count[1]), 2) for i in cf_right])))

                        if new_gini < best_split[0]:
                            best_split = [new_gini, column, split, pd.concat(cf_left), pd.concat(cf_right)]

            if best_split[0] == 1:
                return None, None
            else:
                self.gini = best_split[0]
                self.attribute = best_split[1]
                self.split = best_split[2]
                return best_split[3], best_split[4]
