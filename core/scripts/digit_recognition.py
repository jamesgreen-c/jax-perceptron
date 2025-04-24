from collections import Counter
from core.models.one_versus_rest import ovr_perceptron, ovr_evaluation
import numpy as np
from core.models.utils import make_jax_splits
import jax.numpy as jnp
import matplotlib.pyplot as plt
from core.models.utils import get_data


def extract_misclassified_test_indices(test_index, mistakes):
    """
    Return the index in the original X data for the mistakes of test 
    Mistakes should be boolean such that the following is possible
    Cant vmap as dimension of return value cannot be guaranteed.
    """
    _test_index = np.array(test_index)
    _mistakes = np.array(mistakes)
    
    return _test_index[_mistakes].tolist()


def digit_recognition(data, frac: float, n_runs: int, n_epochs:int = 8, d: int = 5):
    """
    For n runs
        1. Split into test-train
        2. Train perceptron
        3. Test perceptron (return mistakes)
    Sum over mistakes to get data entries that make most mistakes

    Mistakes: from ovr_evaluation, is (20, N) array with boolean vals at where the mistakes occurred.
    """

    train_X, train_Y, train_N, test_X, test_Y, test_N, train_idx, test_idx = make_jax_splits(data, frac, n_runs, return_index=True)

    d_param_set = jnp.array([jnp.repeat(d, train_N) for _ in range(n_runs)])
        
    # get perceptron weights for each run (matrix of flatten weight arrays)
    weights, _ = ovr_perceptron(train_X, train_Y, d_param_set, n_epochs)

    # evaluate train and test error
    test_error, mistakes, C = ovr_evaluation(weights, train_X, test_X, test_Y, d_param_set)

    misclassified_indices = []
    for i in range(n_runs):
        test_index = test_idx[i]
        mistake_arr = mistakes[i]
        misclassified_indices += extract_misclassified_test_indices(test_index, mistake_arr)

    frequencies = Counter(misclassified_indices)
    return frequencies.most_common()
    

if __name__ == "__main__":

    X = get_data()
    most_error_prone: list[tuple] = digit_recognition(X, frac=0.8, n_runs=200)

        # extract the 5 hardest to predict samples
    bad_samples_idx = [tup[0] for tup in most_error_prone[:5]]

    bad_sample_labels = X[bad_samples_idx, 0]
    bad_samples = X[bad_samples_idx, 1:]
    print(bad_sample_labels)

    # plot the 5 hardest to predict samples 
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(14, 15))
    for i, b_sample in enumerate(bad_samples):
        ax[i].imshow(b_sample.reshape(16, 16), interpolation=None, cmap='gray')
        ax[i].set_title(bad_sample_labels[i].astype(int))
        ax[i].axis('off')
    plt.show()
