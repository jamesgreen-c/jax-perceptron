from core.models.one_versus_rest import ovr_perceptron, ovr_evaluation
import jax.numpy as jnp
from jax import vmap
import jax


def cross_val_step(_carry, param_index):
    """
    To write
    """

    test_X_fold, test_Y_fold, train_X, train_Y, errors, param_set = _carry
    train_N = train_Y.shape[0]

    param = param_set[param_index]
    
    # train the perceptron
    d_param_set = jnp.array([jnp.repeat(param, train_N)])
    train_X_a = jnp.array([train_X])
    train_Y_a = jnp.array([train_Y])
    weights, _ = ovr_perceptron(train_X_a, train_Y_a, d_param_set, 8)

    # evaluate the perceptron
    test_X = jnp.array([test_X_fold])
    test_Y = jnp.array([test_Y_fold])
    test_error, _, _ = ovr_evaluation(weights, train_X_a, test_X, test_Y, d_param_set)

    # store the test error and return the carry
    errors = errors.at[param_index].set(test_error[0])
    return (test_X_fold, test_Y_fold, train_X, train_Y, errors, param_set), None


def cross_val_results(test_X_fold, test_Y_fold, train_X, train_Y, param_set: jnp.ndarray):
    """
    
    test_X_fold: (_, 256)
    test_Y_fold: (_,)
    train_X: (_N, 256)
    train_Y: (_N, )
    param_set: in_axes = None, constant vector of parameter values to scan over

    return: We batch 5 fold permutations together resulting in a 
            return matrix of shape(5, 7). (5 test-folds, 7 errors for each d) 
    """

    errors = jnp.zeros(param_set.shape[0])
    N = train_Y.shape[0]
    
    # run over d
    carry = (test_X_fold, test_Y_fold, train_X, train_Y, errors, param_set)
    carry, _ = jax.lax.scan(cross_val_step, carry, np.arange(param_set.shape[0]))

    test_X_fold, test_Y_fold, train_X, train_Y, errors, param_set = carry
    return errors

cross_val_results = vmap(cross_val_results, in_axes=(0, 0, 0, 0, None))


def run_wrapper(
    test_X_folds,
    test_Y_folds,
    train_X_sets,
    train_Y_sets,
    train_X,
    train_Y,
    param_set
):
    """ 
    Wraps the cross_val_results in another vmapped function to batch all 20 runs simultaneously.
    5-fold cross validation

    test_errors will be (5, 7). Use to find best d and retrain perceptron using it.

    test_X_fold: (5, _, 256)
    test_Y_fold: (5, _,)
    train_X: (5, _N, 256)
    train_Y: (5, _N, )

    Returns the 20 weight matrices for the best d trained perceptrons
    """

    # get test errors
    test_errors = cross_val_results(test_X_folds, test_Y_folds, train_X_sets, train_Y_sets, param_set)

    # find the best d
    average_errors = jnp.mean(test_errors, axis=0)
    best_d = param_set[jnp.argmin(average_errors)] # index into parameter set at the given argmin location

    # retrain perceptron on all training data
    train_N = train_Y.shape[0]
    train_X = jnp.array([train_X])
    train_Y = jnp.array([train_Y])
    d_param_set = jnp.array([jnp.repeat(best_d, train_N)])
    weights, _ = ovr_perceptron(train_X, train_Y, d_param_set, 8)

    return weights[0], best_d, test_errors

compiled_cross_validation = jit(vmap(run_wrapper, in_axes=(0, 0, 0, 0, 0, 0, None)))

