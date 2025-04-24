# Kernelized Perceptron (OvO & OvR) in JAX

This project implements digit classification using kernelized Perceptron models in JAX. It supports both One-vs-One (OvO) and One-vs-Rest (OvR) multiclass classification schemes, with training and evaluation accelerated using `jax.jit` and `jax.vmap`.

## Features

- Kernelized Perceptron algorithm
- Multiclass classification via:
  - One-vs-One (OvO)
  - One-vs-Rest (OvR)
- JIT compilation for training and prediction
- Vectorized operations via `jax.vmap`
- Custom RBF and polynomial kernels
- Support for visualizing prediction accuracy

## Dependencies

- Python 3.8+
- [JAX](https://github.com/google/jax)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/) (optional, for plotting)

You can install the dependencies with:

```
pip install -r requirements.txt
```

Example `requirements.txt`:

```
jax
jaxlib
numpy
matplotlib
```


## Notes

- This implementation is for educational and experimental use.
- The training loop is fully vectorized and compiled, but not optimized for large-scale datasets yet.
