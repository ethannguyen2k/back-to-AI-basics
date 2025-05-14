# Back to AI Basics

This repository serves as a refresher guide for fundamental AI and neural network concepts. Use this as a quick reference whenever you need to revisit key machine learning concepts.

## Neural Network Fundamentals

All neural nets, no matter how fancy — are just clever stacks of [dot products](#the-basic-building-block-a-neuron) + [non-linearities](active_functions.ipynb) + [parameter updates](#the-training-process).

### The Basic Building Block: A Neuron

A single neuron (or unit in a layer) computes:

```
output = activation(w · x + b)
```

Where:
- `w · x` is a dot product between weights and input features
- `b` is a bias
- `activation()` is a non-linear function (like [ReLU, sigmoid, tanh](active_functions.ipynb), etc.)

### Scaling Up Neural Networks

- A layer does multiple dot products in parallel (matrix-vector or matrix-matrix multiplication).
- A deep neural network stacks these layers.
- Training adjusts the weights using gradients (derived from the chain rule—again, involving dot products in [backpropagation](packprop.ipynb)).

### Neural Network Variants

- **CNNs**: Dot products are applied through filters sliding over inputs (convolutions are still dot products over localized patches).
- **RNNs/LSTMs**: Apply dot products over sequences with shared weights.
- **Transformers**: Use dot products in the attention mechanism (e.g., Query · Key^T).
- **GNNs**: Nodes do dot products with neighbors' embeddings.
- **MLPs, autoencoders, GANs, etc.**: All rely on layers of dot products.

## The Training Process

1. **Forward pass**: Input data passes through the model to produce predictions
2. **Loss calculation**: Compare predictions to actual targets using a [loss function](loss.md)
3. **Backpropagation**: Calculate gradients of the loss with respect to model parameters using [backpropagation](packprop.ipynb)
4. **Parameter update**: Adjust weights and biases to minimize the loss using [optimizers](optimizers.md)

The goal of training is to minimize this loss function, which should lead to better predictions. And the ultimate goal is generalization, not just minimizing training loss, which can be achieved through proper [regularization](regularization.md).

## Contents of This Repository

- [**loss.md**](loss.md)/[**loss.ipynb**](loss.ipynb): Details about loss functions and their applications
- [**active_functions.ipynb**](active_functions.ipynb): Exploration of activation functions
- [**packprop.ipynb**](packprop.ipynb): Backpropagation algorithms and implementation details
- [**optimizers.md**](optimizers.md)/[**optimizers.ipynb**](optimizers.ipynb): Overview of optimization algorithms for neural networks
- [**regularization.md**](regularization.md): Guide to regularization techniques in both traditional machine learning and deep learning
- [**models/mlp_minst.ipynb**](models/mlp_minst.ipynb): Implementation of Multi-Layer Perceptron for MNIST digit classification

## Future Topics

This repository will continue to grow with additional topics including:
- Model architectures
- Practical implementation tips
- Hyperparameter tuning
- Transfer learning
- And more...

Feel free to explore the notebooks for detailed examples and implementations of these concepts.