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
    '''
    indices = np.arange(samples).astype(int)
    for test_idx in get_indices(n_splits, samples):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx

def k_folds_2(n_splits = 10, samples = 400, num_classes = 4, monte_carlo = False):
    indices = np.arange(samples).astype(int)
    for test_idx in get_indices_2(n_splits, samples, num_classes, monte_carlo = monte_carlo):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx

def get_indices_2(n_splits = 10, samples = 400, num_classes = 4, monte_carlo = False):
    # using n_splits = 10 as example
    if monte_carlo:
        subfold_sizes = int(samples*0.1/num_classes) #400/10/4 = 10 
    else:
        subfold_sizes = int(samples/n)
    
    samples_per_class = int(samples/n_splits/num_classes) #400/4 = 100
    
    indices = np.arange(samples).astype(int)
    current = 0
    
    for i in range(n_splits):
        idx = np.empty((0,)).astype(int)

        if monte_carlo:
            for c in range(num_classes):
                idx = np.append(idx, np.random.choice(indices[int(c*samples_per_class):int((c+1)*samples_per_class)], subfold_sizes, replace = False))

            yield idx
        
        else: 
            start = current #0, 10, 20, 30, 40, 50, 60, 70, 80, 90
            stop =  current + subfold_sizes #10, 20, 30, 40, 50, 60, 70, 80, 90, 100
            current = stop 
            for c in range(num_classes): #0, 1, 2, 3
                idx = np.append(idx, indices[int(start+c*samples_per_class):int(stop+c*samples_per_class)])
                #0:10, 100:110, 200:210, 300:310
                #10:20, 110:120, 210:220, 310:320
                #...
                #90:100, 190:200, 290:300, 390:400
            yield idx


def test_kfold(k = 10, samples = 400):
    for train_idx, test_idx in k_folds(n_splits = 10, samples = samples):
        print(test_idx)
        assert np.unique(train_idx).size == samples/k*(k-1)
        assert np.unique(test_idx).size == samples/k
        train_idx = set(train_idx)
        test_idx = set(test_idx)
        s = list(train_idx.intersection(test_idx))
        assert s == []

def test_kfold_2(k = 15, samples = 400, num_classes = 4, monte_carlo=True):
    for train_idx, test_idx in k_folds_2(n_splits = 10, samples = samples, num_classes=num_classes, monte_carlo=monte_carlo):
        print(test_idx)
        if not monte_carlo:
            assert np.unique(train_idx).size == samples/k*(k-1)
            assert np.unique(test_idx).size == samples/k
        train_idx = set(train_idx)
        test_idx = set(test_idx)
        s = list(train_idx.intersection(test_idx))
        assert s == []

test_kfold_2()

