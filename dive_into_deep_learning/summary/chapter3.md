# Chapter 3 Summary

## 3.1. Linear Regression

### 3.1.1. Basics

We assume that the relationship between the features $\mathbf{x}$ and the target $y$ is approximately linear. In other words, we assume the following.

$$
    y = \mathbf{w}^\top \mathbf{x} + b + \epsilon
    \quad \textrm{where} \quad
    \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

Using the `IID`(Independent and identically distributed random variables) assumption, `log-likelihood` is as follows.

$$
    -\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2
$$

Thus, calculating the `maximum likelihood estimators` is the same as minimizing the loss where the loss is given as follows.

$$
    l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2
    \quad \textrm{where} \quad
    \hat{y}^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} - b
$$

There is a closed-form solution for the linear regression case. However, we use `minibatch stochastic gradient descent` for general cases. The update rule is as follows.

$$
    (\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b)
$$

### 3.1.2. Vectorization for Speed

Vectorized code is much faster than for-loops.

## 3.2. Object-Oriented Design for Implementation

We use class inheritance to simplify making new models. For example, we inherit from `nn.Module` when we create a new model. This way, we only need to override the definition of `__init__` and `forward` for the model to work correctly.
