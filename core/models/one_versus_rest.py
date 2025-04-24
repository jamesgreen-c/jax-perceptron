import jax.numpy as jnp
from jax import vmap, jit
import jax
from core.models.utils import polynomial_kernel, gaussian_kernel   


perceptron_vmap_kernel = vmap(gaussian_kernel, in_axes=(None, 0, 0))


def ovr_perceptron_step(carry, t):
    """ step of the perceptron algorithm """
    
    W, X, Y, M, d = carry
    label = Y[t]

    # get kernel array
    x_t = X[t]
    g = perceptron_vmap_kernel(x_t , X, d)

    # predict
    predictions = jnp.dot(W, g)
    prediction_label = jnp.argmax(predictions)

    # Check for mistake
    mistake = prediction_label != label

    # update weights if theres a mistake
    W = jax.lax.cond(
        mistake,
        lambda w: w.at[label, t].add(1).at[prediction_label, t].add(-1),
        lambda w: w,
        W
    )

    M = M.at[t].set(mistake)

    return (W, X, Y, M, d), None

def ovr_perceptron(X: jnp.ndarray, Y: jnp.ndarray, d: jnp.ndarray, epochs: int):
    """ 
    One Vs Rest Perceptron Algorithm 
        Ran as a VMAP to batch all 20 runs.

    X: training data
    Y: labels for training data
    d: set of params for the kernel function
    """
    
    # concrete
    K = 10

    # set up
    N = Y.shape[0]
    X = X.reshape(N, 256)
    W = jnp.zeros((K, N))
    M = jnp.zeros(N)

    # run epoch as an iteration over time steps
    def run_epoch(carry, _):
        carry, _ = jax.lax.scan(ovr_perceptron_step, carry, jnp.arange(N))
        return carry, None

    # run epochs with jax
    carry = (W, X, Y, M, d)
    carry, _ = jax.lax.scan(run_epoch, carry, None, length=epochs)

    # return results
    W, _, _, M, _ = carry
    return W, M

# compile with vmap and jit
ovr_perceptron = jit(vmap(ovr_perceptron, in_axes=(0, 0, 0, None)), static_argnames=["epochs"])


def ovr_evaluation_step(carry, t):
    """ make predictions for time t """
    
    W, train_X, test_X, P, d = carry

    # get kernel array
    test_x_t = test_X[t]
    g = perceptron_vmap_kernel(test_x_t , train_X, d)

    # predict
    predictions = jnp.dot(W, g)
    prediction_label = jnp.argmax(predictions)
    P = P.at[t].add(prediction_label)
    
    return (W, train_X, test_X, P, d), None


def make_confusion_matrix(carry_2, t):
    """ update confusion matrix for index t of the predictions """

    P, Y, C, digit_counts = carry_2

    # extract labels
    prediction = P[t].astype(int)
    truth = Y[t].astype(int)
    mistake = prediction != truth

    # get digit count for truth
    count = digit_counts[truth]

    # update confusion matrix if mistake
    C = jax.lax.cond(
        mistake,
        lambda c: c.at[truth, prediction].add(1 / count),
        lambda c: c,
        C
    )

    return (P, Y, C, digit_counts), None


def count_digit_entries(Y, digit):
    """ 
    Vmap for counting the number of times digit occurs in the true labels.
    Pass 1 Y array and an array of digit values to find counts for.
    """
    return jnp.sum(Y == digit)

count_digit_entries = vmap(count_digit_entries, in_axes=(None, 0))


def ovr_evaluation(
    W: jnp.ndarray, 
    train_X: jnp.ndarray,
    test_X: jnp.ndarray,
    Y: jnp.ndarray,
    d: jnp.ndarray,
):
    """ 
    One Vs Rest Evaluation
        Ran as a VMAP to batch all 20 runs.

    W: learned weights from ovr_perceptron
    X: training data
    Y: labels for training data
    d: set of params for the kernel function
    """

    K = 10

    # shapes
    N = Y.shape[0]
    train_N = train_X.shape[0]
    
    # init prediciton array
    P = jnp.zeros(N)
    Q
    # run epoch as an iteration over time steps
    def run_epoch(carry, _):
        carry, _ = jax.lax.scan(ovr_evaluation_step, carry, jnp.arange(N))
        return carry, None

    # run epochs with jax
    carry = (W, train_X, test_X, P, d)
    carry, _ = jax.lax.scan(run_epoch, carry, None, length=1)

    # extract predictions and calculate percentage error
    _, _, _, P, _ = carry
    mistakes = jnp.not_equal(P, Y)
    percentage_error = (mistakes.sum() / N) * 100
    
    # prep for confusion matrix calculation
    C = jnp.zeros((10, 10))
    digit_arr = jnp.arange(10)
    digit_counts = count_digit_entries(Y, digit_arr)

    # calculate confusion matrix
    carry_2 = (P, Y, C, digit_counts)
    carry_2, _ = jax.lax.scan(make_confusion_matrix, carry_2, jnp.arange(N))
    _, _, C, _ = carry_2

    # make C a percentage
    C = C * 100
    
    return percentage_error, mistakes, C

# compile with vmap and jit
ovr_evaluation = jit(vmap(ovr_evaluation))

