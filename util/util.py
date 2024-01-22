import numpy as np
import torch
from config import data_dirichlet_alpha, participation_dirichlet_alpha, participation_prob_mean, participation_prob_min
from torch.utils.data import Dataset


class WorkerSampler:
    def __init__(self, method, participation_prob):
        self.participation_prob = participation_prob

        s = method.split('-')
        self.method = s[0]

        if self.method == 'cyclic':
            self.cycle = int(s[1])
            self.active_rounds = max(1, int(np.round(self.cycle * self.participation_prob)))
            self.inactive_rounds = max(1, self.cycle - self.active_rounds)
            self.currently_active = False
            self.rounds_to_switch = np.random.randint(self.inactive_rounds)
        elif self.method == 'markov':
            self.transition_to_active_prob = float(s[1])
            self.transition_to_inactive_prob = self.transition_to_active_prob * (1 / self.participation_prob - 1)
            if self.transition_to_inactive_prob > 1:
                self.transition_to_active_prob /= self.transition_to_inactive_prob
                self.transition_to_inactive_prob = 1
            self.currently_active = (np.random.binomial(1, self.participation_prob) == 1)

    def cyclic_update(self):
        if self.rounds_to_switch == 0:
            self.currently_active = not self.currently_active
            if self.currently_active:
                self.rounds_to_switch = self.active_rounds
            else:
                self.rounds_to_switch = self.inactive_rounds

        self.rounds_to_switch -= 1

    def markov_update(self):
        if self.currently_active:
            if np.random.binomial(1, self.transition_to_inactive_prob) == 1:
                self.currently_active = False
        else:
            if np.random.binomial(1, self.transition_to_active_prob) == 1:
                self.currently_active = True

    def sample(self):
        if self.method == 'bernoulli':
            return np.random.binomial(1, self.participation_prob) == 1
        elif self.method == 'cyclic':
            self.cyclic_update()
            return self.currently_active
        elif self.method == 'markov':
            self.markov_update()
            return self.currently_active


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def partition(dataset, n_nodes):
    dict_users = {i: np.array([], dtype='int64') for i in range(n_nodes)}

    if isinstance(dataset.targets, torch.Tensor):
        labels = dataset.targets.numpy()
    else:
        labels = np.array(dataset.targets)

    min_label = min(labels)
    max_label = max(labels)
    num_labels = max_label - min_label + 1

    label_distributions_each_node = np.random.dirichlet(data_dirichlet_alpha * np.ones(num_labels), n_nodes)
    sum_prob_per_label = np.sum(label_distributions_each_node, axis=0)

    indices_per_label = []
    for i in range(min_label, max_label + 1):
        indices_per_label.append([j for j in range(len(labels)) if labels[j] == i])

    start_index_per_label = np.zeros(num_labels, dtype='int64')
    for n in range(n_nodes):
        for i in range(num_labels):
            end_index = int(np.round(len(indices_per_label[i]) * np.sum(label_distributions_each_node[:n+1, i]) / sum_prob_per_label[i]))
            dict_users[n] = np.concatenate((dict_users[n], np.array(indices_per_label[i][start_index_per_label[i] : end_index], dtype='int64')), axis=0)
            start_index_per_label[i] = end_index

    actual_label_distributions_each_node = [np.array([len([j for j in labels[dict_users[n]] if j == i]) for i in range(min_label, max_label + 1)], dtype='int64')
                                            / len(dict_users[n]) for n in range(n_nodes)]

    return dict_users, actual_label_distributions_each_node, num_labels


def data_participation_each_node(data_train, n_nodes):
    dict_users, actual_label_distributions_each_node, num_labels = partition(data_train, n_nodes)
    participation_prob_multiplier = np.random.dirichlet(participation_dirichlet_alpha * np.ones(num_labels), 1)
    participation_scaling = participation_prob_mean * num_labels
    participation_prob_each_node = [np.sum(np.multiply(actual_label_distributions_each_node[n], participation_prob_multiplier)) * participation_scaling for n in range(n_nodes)]
    participation_prob_each_node = np.maximum(participation_prob_min, participation_prob_each_node)
    print('participation_prob_each_node:', participation_prob_each_node, flush=True)
    return dict_users, participation_prob_each_node


