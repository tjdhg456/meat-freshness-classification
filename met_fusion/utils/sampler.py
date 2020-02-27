import torch
import torch.utils.data
import torchvision
import numpy as np


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dat, aux, label = dataset.__getitem__(idx)
        return label.item()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class Sampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, type='under'):
        self.indices = range(len(dataset))

        # distribution of classes in the dataset
        self.label_to_idx = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in self.label_to_idx:
                self.label_to_idx[label].append(idx)
            else:
                self.label_to_idx[label] = [idx]

        length = [len(lab) for lab in self.label_to_idx.values()]
        if type == 'under':
            min_ix, = np.where(length == np.min(length))
            minimum = np.min(length)
            for ix, (key, label_data) in enumerate(self.label_to_idx.items()):
                if ix == min_ix:
                    continue
                else:
                    self.label_to_idx[key] = np.random.choice(label_data, minimum, replace=False)

        elif type == 'over':
            max_ix, = np.where(length == np.max(length))
            maximum = np.max(length)
            for ix, (key, label_data) in enumerate(self.label_to_idx.items()):
                if ix == max_ix:
                    continue
                else:
                    self.label_to_idx[key] = np.random.choice(label_data, maximum, replace=True)

        self.label_to_idx = list(self.label_to_idx.values())
        self.label_to_idx = np.concatenate(self.label_to_idx, axis=0)

    def _get_label(self, dataset, idx):
        dat, aux, label = dataset.__getitem__(idx)
        return label.item()

    def __iter__(self):
        return (ix for ix in self.label_to_idx)

    def __len__(self):
        return len(self.label_to_idx)