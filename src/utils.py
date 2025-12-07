# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kishore V

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_har_dataset(data_dir="./dataset/UCI_HAR_Dataset/UCI_HAR_Dataset/"):
    # Load train/test data from UCI HAR
    X_train = np.loadtxt(data_dir + "train/X_train.txt")
    y_train = np.loadtxt(data_dir + "train/y_train.txt") - 1
    X_test = np.loadtxt(data_dir + "test/X_test.txt")
    y_test = np.loadtxt(data_dir + "test/y_test.txt") - 1

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return (X_train, y_train), (X_test, y_test)

def partition_dataset(X, y, num_clients=5, non_iid=False):
    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    shard_size = total_size // num_clients
    partitions = []

    if non_iid:
        # Sort by label to simulate skewed distribution
        indices = np.argsort(y.numpy())
        X, y = X[indices], y[indices]

    for i in range(num_clients):
        start = i * shard_size
        end = (i + 1) * shard_size
        part = TensorDataset(X[start:end], y[start:end])
        partitions.append(part)

    return partitions

    return partitions
