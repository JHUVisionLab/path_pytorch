import numpy as np
import pdb

# Developer: Alejandro Debus
# Email: aledebus@gmail.com

def partitions(number, k):
    '''
    Distribution of the folds
    Args:
        number: number of samples
        k: folds number
    '''
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions

def get_indices(n_splits = 10, samples = 400):
    '''
    Indices of the set test
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    l = partitions(samples, n_splits)
    fold_sizes = l 
    indices = np.arange(samples).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])

def k_folds(n_splits = 10, samples = 400):
    '''
    Generates folds for cross validation
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    indices = np.arange(samples).astype(int)
    for test_idx in get_indices(n_splits, samples):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx

def test_kfold(k = 10, samples = 400):
    for train_idx, test_idx in k_folds(n_splits = 10, samples = samples):
        assert np.unique(train_idx).size == samples/k*(k-1)
        assert np.unique(test_idx).size == samples/k
        train_idx = set(train_idx)
        test_idx = set(test_idx)
        s = list(train_idx.intersection(test_idx))
        assert s == []

test_kfold()

