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

## 3.6. Generalization

### 3.6.1. Training Error and Generalization Error

Training error is an error on the training set. Generalization error is an error on the unseen data. Our objective is to minimize the Generalization error.

### 3.6.2. Underfitting or Overfitting?

Underfitting is when the training error is large and the generalization gap is small. The reason is that the model is too simple.

Overfitting is when the training error is small and the generalization gap is large. The reason is that the training set size is too small.

## 3.7. Weight Decay

Weight decay is a common way to mitigate overfitting.

### 3.7.1. Norms and Weight Decay

This method is also called $\ell_2$ regularization. This is because we add the $\ell_2$ norm of the weights to the loss function. The new loss function is as follows.

$$
    L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2
    \quad \textrm{where} \quad
    L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2
$$

The new update rule is quite simple.

$$\begin{aligned}
    \mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)
\end{aligned}$$

As we need to access the original weights when updating the weights, the additional computational cost by the weight decay is negligible.
