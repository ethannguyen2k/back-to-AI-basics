# Optimization Algorithms in Machine Learning

## Abstract

This document provides a comprehensive overview of optimization algorithms used in machine learning and deep learning. It covers fundamental concepts, implementation details, and practical comparisons of common optimizers including SGD, Adam, RMSprop, and others. The guide includes PyTorch code examples, mathematical explanations, and recommendations for algorithm selection based on use cases.

## Table of Contents

- [Introduction](#introduction)
- [Optimizer Comparison](#optimizer-comparison)
- [Optimization Algorithms in Detail](#optimization-algorithms-in-detail)
  - [1. Stochastic Gradient Descent (SGD)](#1-stochastic-gradient-descent-sgd)
  - [2. Adam (Adaptive Moment Estimation)](#2-adam-adaptive-moment-estimation)
  - [3. RMSprop](#3-rmsprop)
  - [4. Adagrad](#4-adagrad)
  - [5. AdamW](#5-adamw)
  - [6. Adadelta](#6-adadelta)
  - [7. LBFGS (Limited-memory BFGS)](#7-lbfgs-limited-memory-bfgs)
  - [8. RAdam (Rectified Adam)](#8-radam-rectified-adam)
  - [9. Lookahead](#9-lookahead)
- [Choosing the Right Optimizer](#choosing-the-right-optimizer)
- [References](#references)

## Introduction

Optimization algorithms work to minimize or maximize functions to find the best solution (typically minimizing loss functions in ML).

Neural networks and other ML models involve finding the right values for thousands to millions of parameters. Without an efficient method to adjust these parameters, training would be practically impossible. The parameter space is too vast to search exhaustively or randomly.

They solve fundamental problems in ML training:

- **Efficiency**: Gradient-based methods provide a computationally feasible way to navigate high-dimensional parameter spaces
- **Local minima**: Advanced optimizers help avoid getting stuck in poor solutions
- **Convergence speed**: Different optimizers can dramatically reduce training time (from weeks to days or hours)
- **Generalization**: The right optimizer can lead to better-performing models on unseen data

In the ML training workflow, optimization algorithms are used in the final step to update weights and biases using gradients calculated during backpropagation. In code, optimizers are initialized once at the beginning of training setup and then applied repeatedly during each iteration of the training loop.

## Optimizer Comparison

| Optimizer | Learning Rate Adaptation | Momentum | Memory Requirements | Computation Cost | Best Use Cases |
|-----------|--------------------------|----------|---------------------|-----------------|----------------|
| SGD | No | Optional | Low | Low | When simplicity is needed; with proper tuning for generalization |
| Adam | Yes | Yes | Medium | Medium | General-purpose; default choice for many tasks |
| RMSprop | Yes | No | Medium | Medium | Non-stationary problems; RNNs |
| Adagrad | Yes | No | Medium | Medium | Sparse data; NLP tasks |
| AdamW | Yes | Yes | Medium | Medium | When regularization is important; transformer models |
| Adadelta | Yes | Implicit | Medium | Medium | When learning rate tuning is difficult |
| LBFGS | Yes | N/A | High | High | Small datasets; when exact optimization is needed |
| RAdam | Yes | Yes | Medium | Medium-High | When Adam shows convergence issues |
| Lookahead | Depends on base | Depends on base | Medium | Medium-High | To stabilize training of any base optimizer |

## Optimization Algorithms in Detail

### 1. Stochastic Gradient Descent (SGD)

**Core Mechanism**: Updates parameters using the formula: θ = θ - η∇J(θ), where η is the learning rate.

**Variants**:
- **Vanilla SGD**: Uses single samples or mini-batches instead of full dataset
- **SGD with Momentum**: Adds velocity term to smooth updates
  ```
  v = γv - η∇J(θ)
  θ = θ + v
  ```
  Where γ (typically 0.9) is the momentum coefficient.

**PyTorch Implementation**:
```python
import torch.optim as optim

# Basic SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum and weight decay
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
```

**Advantages**: Simple implementation, works well with sparse data
**Disadvantages**: Sensitive to feature scaling, can oscillate in ravines

### 2. Adam (Adaptive Moment Estimation)

**Core Mechanism**: Maintains adaptive learning rates using estimates of first moment (mean) and second moment (uncentered variance) of gradients.

**Algorithm**:
```
m = β₁m + (1-β₁)∇J(θ)           // Update biased first moment estimate
v = β₂v + (1-β₂)(∇J(θ))²        // Update biased second moment estimate
m̂ = m/(1-β₁ᵗ)                   // Bias-corrected first moment
v̂ = v/(1-β₂ᵗ)                   // Bias-corrected second moment
θ = θ - η·m̂/(√v̂ + ε)           // Update parameters
```

**PyTorch Implementation**:
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
```

**Hyperparameters**: Learning rate (η), exponential decay rates (β₁, β₂), and numerical stability term (ε)

**Advantages**: Works well in practice, adapts learning rate for each parameter, requires minimal tuning
**Disadvantages**: Can converge to suboptimal solutions on some problems

### 3. RMSprop

**Core Mechanism**: Adapts learning rates by dividing by a running average of squared gradients.

**Algorithm**:
```
v = βv + (1-β)(∇J(θ))²          // Accumulate squared gradients
θ = θ - η·∇J(θ)/(√v + ε)        // Update parameters
```

**PyTorch Implementation**:
```python
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-8, weight_decay=1e-5)
```

**Advantages**: Handles non-stationary objectives well, resolves Adagrad's radically diminishing learning rates
**Disadvantages**: Requires setting a good global learning rate

### 4. Adagrad

**Core Mechanism**: Adapts learning rates for each parameter based on historical gradients.

**Algorithm**:
```
G = G + (∇J(θ))²                // Accumulate squared gradients 
θ = θ - η·∇J(θ)/(√G + ε)        // Update parameters
```

**PyTorch Implementation**:
```python
optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=1e-5)
```

**Advantages**: Good for sparse data, eliminates manual learning rate tuning
**Disadvantages**: Aggressive learning rate decay can prematurely stop learning

### 5. AdamW

**Core Mechanism**: Decouples weight decay from gradient updates for better regularization.

**Algorithm**: Like Adam but with modified weight decay:
```
// Regular Adam update steps...
θ = θ - η(m̂/(√v̂ + ε) + λθ)     // λ is weight decay coefficient
```

**PyTorch Implementation**:
```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
```

**Advantages**: Better generalization than Adam, reduced overfitting
**Disadvantages**: Requires tuning both learning rate and weight decay

### 6. Adadelta

**Core Mechanism**: Eliminates need for a global learning rate by using the ratio of accumulated parameter updates to accumulated gradients.

**Algorithm**:
```
v = ρv + (1-ρ)(∇J(θ))²          // Accumulate squared gradients
Δθ = -√(Δx + ε)/√(v + ε)·∇J(θ)  // Compute update
Δx = ρΔx + (1-ρ)(Δθ)²           // Accumulate squared updates
θ = θ + Δθ                      // Update parameters
```

**PyTorch Implementation**:
```python
optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=1e-5)
```

**Advantages**: No need to set learning rate, robust to large gradients
**Disadvantages**: May converge slower than Adam in some cases

### 7. LBFGS (Limited-memory BFGS)

**Core Mechanism**: Approximates the Hessian matrix (second derivatives) to make more informed updates.

**Key Features**:
- Computes approximate curvature using historical gradients
- Uses line search to determine step size
- Requires full batch (not stochastic)

**PyTorch Implementation**:
```python
optimizer = optim.LBFGS(model.parameters(), lr=1, max_iter=20, history_size=100)
```

**Advantages**: Faster convergence with fewer iterations, handles ill-conditioned problems well
**Disadvantages**: More computation per step, requires more memory

### 8. RAdam (Rectified Adam)

**Core Mechanism**: Addresses Adam's convergence issues by incorporating a "rectification" term that adapts based on the variance of learning rates.

**Key Innovation**: Dynamically adjusts the adaptive learning rate based on variance estimation, providing warmup automatically.

**PyTorch Implementation**:
```python
# Using third-party implementation
from torch_optimizer import RAdam
optimizer = RAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
```

**Advantages**: Improves stability, reduces need for learning rate warmup
**Disadvantages**: Slightly more computation than Adam

### 9. Lookahead

**Core Mechanism**: A meta-optimizer that works with any base optimizer.

**Algorithm**:
1. Run k steps of the base optimizer (fast weights)
2. Take a step toward the new weights (slow weights)
   ```
   slow_weights = slow_weights + α(fast_weights - slow_weights)
   ```
3. Reset fast weights to slow weights
4. Repeat

**PyTorch Implementation**:
```python
# Using third-party implementation
from torch_optimizer import Lookahead
base_optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

**Advantages**: Improves convergence stability, reduces variance
**Disadvantages**: Additional computational overhead

## Choosing the Right Optimizer

When selecting an optimizer for your machine learning task, consider these guidelines:

1. **For starting out**: Begin with Adam as it works well across many problems with default hyperparameters
2. **For computer vision**:
   - Adam or SGD with momentum for CNNs
   - AdamW for vision transformers
3. **For NLP tasks**:
   - Adam/AdamW for transformer architectures
   - RMSprop for RNNs/LSTMs
4. **For reinforcement learning**: Adam or RMSprop
5. **When computational resources are limited**: SGD uses less memory
6. **When overfitting occurs**: Try AdamW with appropriate weight decay
7. **When Adam has convergence issues**: Try RAdam
8. **For fine-tuning pretrained models**: Lower learning rates with Adam or AdamW

The optimal choice often requires experimentation, as performance can vary based on model architecture, dataset characteristics, and specific task requirements.

## References

1. [Difference between RMSprop with momentum and Adam optimizers](https://datascience.stackexchange.com/questions/26792/difference-between-rmsprop-with-momentum-and-adam-optimizers)
2. [Gradient Descent Algorithm and its variants](https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/)
3. [PyTorch optimizer AdamW and Adam with weight decay](https://stackoverflow.com/questions/64621585/pytorch-optimizer-adamw-and-adam-with-weight-decay)
4. [Gradient Descent with Adadelta from scratch](https://machinelearningmastery.com/gradient-descent-with-adadelta-from-scratch/)
5. [How does the L-BFGS work?](https://stats.stackexchange.com/questions/284712/how-does-the-l-bfgs-work)
6. [RAdam: On the Variance of the Adaptive Learning Rate and Beyond](https://paperswithcode.com/method/radam)
7. [Lookahead Optimizer: k steps forward, 1 step back](https://github.com/michaelrzhang/lookahead)