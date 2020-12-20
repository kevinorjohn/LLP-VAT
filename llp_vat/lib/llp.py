import os
import pathlib
import time
from itertools import groupby

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from torch.utils.data import Sampler, BatchSampler, RandomSampler

from llp_vat.lib.datasets import load_dataset


class Iteration:
    def __init__(self, start=0):
        self.value = start

    def step(self, step=1):
        self.value += step


class BagMiniBatch:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.reset()

    def reset(self):
        self.bags = []
        self.bag_sizes = []
        self.targets = []  # store proportion labels

    def append(self, x, y):
        assert x.size(0) == y.size(0)
        self.targets.append(torch.mean(y, dim=0))
        if self.n_samples > 0:
            index = torch.randperm(x.size(0))[:self.n_samples]
            x = x[index]
            y = y[index]
        self.bags.append((x, y))
        self.bag_sizes.append(y.size(0))

    def __iter__(self):
        for item in zip(self.bag_sizes, self.targets):
            yield item

    @property
    def total_size(self):
        return sum(self.bag_sizes)

    @property
    def max_bag_size(self):
        return max(self.bag_sizes)

    @property
    def num_bags(self):
        return len(self.bag_sizes)


class BagSampler(Sampler):
    def __init__(self, bags, num_bags=-1):
        """
        params:
            bags: shape (num_bags, num_instances), the element of a bag
                  is the instance index of the dataset
            num_bags: int, -1 stands for using all bags
        """
        self.bags = bags
        if num_bags == -1:
            self.num_bags = len(bags)
        else:
            self.num_bags = num_bags
        assert 0 < self.num_bags <= len(bags)

    def __iter__(self):
        indices = torch.randperm(self.num_bags)
        for index in indices:
            yield self.bags[index]

    def __len__(self):
        return len(self.bags)


def uniform_creation(dataset, bag_size, replacement, seed, drop_last=True):
    """
    return:
        bags: a nested list containing instance indices, shape (n_bags, *)
    """
    torch.manual_seed(seed)

    start = time.time()
    indices = RandomSampler(range(len(dataset)), replacement=replacement)
    bags = list(BatchSampler(indices, batch_size=bag_size,
                             drop_last=drop_last))
    print("Create uniform bags in {:.2f} seconds".format(time.time() - start))
    return bags


def kmeans_creation(dataset, n_clusters, reduction, seed):
    random_state = np.random.RandomState(seed)
    data = [(x, y) for (x, y) in dataset]
    X, y = map(torch.stack, zip(*data))
    X = X.view(X.size(0), -1)

    # PCA reduction
    start = time.time()
    pca = PCA(n_components=reduction)
    X_new = pca.fit_transform(X)
    print("PCA-{} in {:.2f} seconds".format(reduction, time.time() - start))

    # assign bag label by k-means clustering
    start = time.time()
    init_size = max(3 * n_clusters, 300)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             random_state=random_state,
                             init_size=init_size)
    kmeans.fit(X_new)
    bag_labels = kmeans.predict(X_new)
    print("K-means {} in {:.2f} seconds".format(n_clusters,
                                                time.time() - start))

    # create bags
    start = time.time()
    bags = sorted(zip(bag_labels, range(len(bag_labels))), key=lambda x: x[0])
    bags = [[idx for _, idx in data]
            for _, data in groupby(bags, key=lambda x: x[0])]
    print("Create kmeans bags in {:.2f} seconds".format(time.time() - start))
    return bags


def load_llp_dataset(dataset_dir, obj_dir, dataset_name, alg, **kwargs):
    dataset = load_dataset(dataset_dir, dataset_name)

    if alg == "uniform":
        sampling = "SWR" if kwargs["replacement"] else "SWOR"
        filename = "uniform-{}-{}.npy".format(sampling, kwargs["bag_size"])
    elif alg == "kmeans":
        filename = "kmeans-{}-{}.npy".format(kwargs["n_clusters"],
                                             kwargs["reduction"])
    elif alg == "overlap":
        filename = "overlap-{}-{}.npy".format(kwargs["num_overlaps"],
                                              kwargs["bag_size"])
    else:
        raise NameError("algorithm {} is not supported".format(alg))
    path = os.path.join(obj_dir, dataset_name, filename)

    bags = np.load(path, allow_pickle=True)
    print("Load bags from {}".format(path))
    return dataset, bags


def create_llp_dataset(dataset_dir, obj_dir, dataset_name, alg, **kwargs):
    dataset = load_dataset(dataset_dir, dataset_name)
    if alg == "uniform":
        sampling = "SWR" if kwargs["replacement"] else "SWOR"
        filename = "uniform-{}-{}.npy".format(sampling, kwargs["bag_size"])
        bags = uniform_creation(dataset["train"], **kwargs)
    elif alg == "kmeans":
        filename = "kmeans-{}-{}.npy".format(kwargs["n_clusters"],
                                             kwargs["reduction"])
        bags = kmeans_creation(dataset["train"], **kwargs)
    else:
        raise NameError("algorithm {} is not supported".format(alg))
    path = os.path.join(obj_dir, dataset_name, filename)
    # dump bags
    dirname = os.path.dirname(os.path.abspath(path))
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    bags = np.array(bags)
    np.save(path, bags)
