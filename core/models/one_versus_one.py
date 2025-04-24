from itertools import combinations
import jax.numpy as jnp
from jax import vmap, jit
import jax
from core.models.utils import polynomial_kernel, gaussian_kernel   


perceptron_vmap_kernel = vmap(gaussian_kernel, in_axes=(None, 0, 0))


def restructure_classifiers(pair, label):
    """ 
    Vmapped function to determine both whether the classifier is relevant and if so which label is the positive or negative label
    pair: arr([label 1, label 2])
    label: digit (int)

    Returns: bool, bool. First bool is whether the label is relevant to this classifier, 
                            second bool is whether the label is the positive class or not.
    """

    bools = pair == label

    # get binary label
    binary_label = (bools.astype(int)[0] * 2) - 1  # False * 2 - 1 = -1, True * 2 - 1 = 1

    return jnp.logical_or(bools[0], bools[1]), binary_label 

restructure_classifiers = vmap(restructure_classifiers, in_axes=(0, None))


def update_weights(w, boolean, learning_rate, t):
    """ Vmap for weight updates """

    rate = learning_rate * boolean.astype(int)
    w = w.at[t].add(rate)
    
    return w

update_weights = vmap(update_weights, in_axes=(0, 0, None, None))


def update_mistakes(m, boolean, t):
    """ Vmap for weight updates """
    
    mistake = boolean.astype(int) * 1  # 1 if mistake, 0 if no mistake
    m = m.at[t].set(mistake)

    return m

update_mistakes = vmap(update_mistakes, in_axes=(0, 0, None))


def ovo_perceptron_step(carry, t):
    """Step of the OvO Perceptron algorithm."""
    
    W, X, Y, M, d, pairs = carry
    x_t = X[t]
    label = Y[t]

    # filter the weight matrix to only required classifiers
    filter_arr, binary_labels = restructure_classifiers(pairs, label)

    # predict using all classifiers
    g = perceptron_vmap_kernel(x_t, X, d)
    predictions = jnp.dot(W, g)
    binary_predictions = jnp.where(predictions > 0, 1, -1)
    # binary_predictions = (predictions / abs(predictions)).astype(int)  # make predictions binary

    # find false positives and false negatives, to update weight matrix
    false_positives = (binary_predictions != binary_labels) & filter_arr & (binary_labels == -1)
    false_negatives = (binary_predictions != binary_labels) & filter_arr & (binary_labels == 1)

    # update weights
    W = update_weights(W, false_positives, -1, t)
    W = update_weights(W, false_negatives, 1, t)
    
    # # track mistakes
    M = update_mistakes(M, false_positives, t)
    M = update_mistakes(M, false_negatives, t)

    return (W, X, Y, M, d, pairs), None


def ovo_perceptron(X: jnp.ndarray, Y: jnp.ndarray, d: jnp.ndarray, epochs: int):
    """ 
    One vs One Perceptron Algorithm 
    Trains classifiers for every pair of classes.

    X: training data
    Y: labels for training data
    d: set of params for the kernel function
    """
    
    # make class pairs
    classes = jnp.arange(0, 10)
    all_pairs = jnp.array(list(combinations(classes, 2)))
    K = all_pairs.shape[0]

    # set up
    N = Y.shape[0]
    X = X.reshape(N, 256)
    W = jnp.zeros((K, N))
    M = jnp.zeros((K, N))

    # run epoch as an iteration over time steps
    def run_epoch(carry, _):
        carry, _ = jax.lax.scan(ovo_perceptron_step, carry, jnp.arange(N))
        return carry, None

    carry = (W, X, Y, M, d, all_pairs)
    carry, _ = jax.lax.scan(run_epoch, carry, None, length=epochs)

    W, _, _, M, _, _ = carry
    return W, M

# compile with vmap and jit
ovo_perceptron = jit(vmap(ovo_perceptron, in_axes=(0, 0, 0, None)), static_argnames=["epochs"])


def class_predictions(pair, score):
    """
    Vmapped function to calculate the binary prediction label.
    returns prediction label
    """

    index = (score < 0).astype(int)
    return pair[index]

class_predictions = vmap(class_predictions)


def ovo_evaluation_step(carry, t):
    """ make predictions for time t """
    
    W, train_X, test_X, P, d, pairs = carry

    # get kernel array
    test_x_t = test_X[t]
    g = perceptron_vmap_kernel(test_x_t , train_X, d)

    # predict
    scores = jnp.dot(W, g)
    predictions = class_predictions(pairs, scores)
    prediction_label = jnp.bincount(predictions, length=10).argsort(descending=True)[0]
    
    P = P.at[t].set(prediction_label)
    
    return (W, train_X, test_X, P, d, pairs), None


def ovo_evaluation(
    W: jnp.ndarray, 
    train_X: jnp.ndarray,
    test_X: jnp.ndarray,
    Y: jnp.ndarray,
    d: jnp.ndarray,
):
    """ 
    One Vs One Evaluation
        Ran as a VMAP to batch all 20 runs.

    W: learned weights from ovr_perceptron
    X: training data
    Y: labels for training data
    d: set of params for the kernel function
    """

    # make class pairs
    classes = jnp.arange(0, 10)
    all_pairs = jnp.array(list(combinations(classes, 2)))
    # K = all_pairs.shape[0]

    # shapes
    N = Y.shape[0]
    train_N = train_X.shape[0]
    
    # init prediciton array
    P = jnp.zeros(N)
    
    # run epoch as an iteration over time steps
    def run_epoch(carry, _):
        carry, _ = jax.lax.scan(ovo_evaluation_step, carry, jnp.arange(N))
        return carry, None

    # run epochs with jax
    carry = (W, train_X, test_X, P, d, all_pairs)
    carry, _ = jax.lax.scan(run_epoch, carry, None, length=1)

    # extract predictions and calculate percentage error
    _, _, _, P, _, _ = carry
    mistakes = jnp.not_equal(P, Y)
    percentage_error = (mistakes.sum() / N) * 100
        
    return percentage_error, mistakes

# compile with vmap and jit
ovo_evaluation = jit(vmap(ovo_evaluation))

