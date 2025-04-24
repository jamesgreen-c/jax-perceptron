import jax.numpy as jnp
import os
import numpy as np


def polynomial_kernel(a, b, d):
    return jnp.dot(a, b.T) ** d


def gaussian_kernel(a, b, c):
    """ \\| a - b \\|^2 = <a-b, a-b> = \\sum_i (a_i - b_i)^2 """
    return jnp.exp(-c * jnp.sum((a - b) ** 2))


def find_file_directory(file_name):
    start = os.getcwd().split("algorithmic-trading")[0] + "algorithmic-trading"
    for root, dirs, files in os.walk(start):
        if file_name in files:
            return os.path.join(root, file_name)
    return None


def get_data(epochs: int = 1) -> np.ndarray:
    with open('zipcombo.txt', 'r') as f:
        # read the lines from the file and convert each line to a list of numbers
        X = np.array([list(map(float, line.split())) for line in f] * epochs)
    return X


def split(data: np.ndarray, frac: float):
    """ 
    Split data into train and test sets based on frac, where frac is the fraction of training data.
    Make sure the data still contains the labels column.
    Returns (train, test) 
    """
    
    assert 0 < frac < 1
    
    N = data.shape[0]
    permutation = np.random.permutation(N)
    
    pivot = int(N * frac)
    train_idx, test_idx = permutation[:pivot], permutation[pivot:]

    # split labels and data here so I don't have to loop over it again
    train, test = data[train_idx], data[test_idx]
    train_X, train_Y = train[:, 1:], train[:, 0].astype(int)
    test_X, test_Y = test[:, 1:], test[:, 0].astype(int)

    return train_X, train_Y, test_X, test_Y, train_idx, test_idx


def make_jax_splits(data, frac: float, n_runs: int, return_index: bool = False):
    """ Used to make matrices for vmapped function """
    
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # stores for index arrs
    train_indexes = []
    test_indexes = []
    
    # make splits
    for _ in range(n_runs):
        
        train_X, train_Y, test_X, test_Y, train_idx, test_idx = split(data, frac=frac)
        
        train_data.append(train_X) 
        train_labels.append(train_Y)
        test_data.append(test_X)
        test_labels.append(test_Y)

        train_indexes.append(train_idx)
        test_indexes.append(test_idx)

    # make training jax arrays
    train_X = jnp.array(train_data)
    train_Y = jnp.array(train_labels)
    train_N = train_Y.shape[1]

    # make test jax arrays 
    test_X = jnp.array(test_data)
    test_Y = jnp.array(test_labels)
    test_N = test_Y.shape[1]

    if return_index:

        train_index = jnp.array(train_indexes)
        test_index = jnp.array(test_indexes)
        return train_X, train_Y, train_N, test_X, test_Y, test_N, train_index, test_index
    
    return train_X, train_Y, train_N, test_X, test_Y, test_N


def batch(train_X, train_Y, n_runs, n_folds):
    """ Create batches for vmap """
    

    all_runs_test_X_fold = []
    all_runs_test_Y_fold = []
    all_runs_train_X = []
    all_runs_train_Y = []
    
    for i in range(n_runs):
        
        # create folds
        X_folds = np.array_split(train_X[i], n_folds, axis=0)
        Y_folds = np.array_split(train_Y[i], n_folds)

        max_shape = 0
        for k in range(n_folds):
            
            _x_fold = X_folds[k]
            _y_fold = Y_folds[k]
            
            max_shape = max(_x_fold.shape[0], max_shape)
            
            if _x_fold.shape[0] < max_shape:
                idx = np.random.randint(_x_fold.shape[0])
                X_folds[k] = np.concatenate([_x_fold, _x_fold[idx].reshape(1, 256)]) 
                Y_folds[k] = np.concatenate([_y_fold, _y_fold[idx].reshape(1)]) 
        
        run_test_X_fold = []
        run_test_Y_fold = []
        run_train_X = []
        run_train_Y = []

        for j in range(n_folds):

            # get training index
            train_X_folds = np.concatenate(X_folds[:j] + X_folds[j+1:])
            train_Y_folds = np.concatenate(Y_folds[:j] + Y_folds[j+1:])

            # get test fold info
            test_X_fold = X_folds[j]
            test_Y_fold = Y_folds[j]
            
            run_test_X_fold.append(test_X_fold)
            run_test_Y_fold.append(test_Y_fold)
            run_train_X.append(train_X_folds)
            run_train_Y.append(train_Y_folds)
            
        all_runs_test_X_fold.append(run_test_X_fold)
        all_runs_test_Y_fold.append(run_test_Y_fold)
        all_runs_train_X.append(run_train_X)
        all_runs_train_Y.append(run_train_Y)

    all_runs_test_X_fold = np.array(all_runs_test_X_fold)
    all_runs_test_Y_fold = np.array(all_runs_test_Y_fold)
    all_runs_train_X = np.array(all_runs_train_X)
    all_runs_train_Y = np.array(all_runs_train_Y)

    return all_runs_test_X_fold, all_runs_test_Y_fold, all_runs_train_X, all_runs_train_Y




