# Chapter 4 Summary

## 4.1. Softmax Regression

### 4.1.1. Classification

`multi-label classification` is a very common task in machine learning. We use `one-hot encoding` to represent such a classification.

For a task with $n$ different labels, we use $n$ different regression models. Each regression model predicts the probability of each label.

A probability must be in the range $[0, 1]$, and the probabilities must sum up to $1$. To achieve this, we use the `softmax` function.

$$
    \hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})
    \quad \textrm{where}\quad
    \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}
$$

As softmax calculation involves exponential, the calculation might overflow or underflow. This problem is mitigated by dividing the numerator and the denominator with the largest exponential term(before the actual exponential calculation). This method is called as `LogSumExp`.

### 4.1.2. Loss Function

The loss function `cross-entropy loss` is as follows.

$$
    \sum_{i=1}^n l(\mathbf{y^{(i)}}, \hat{\mathbf{y}}^{(i)})
    \quad \textrm{where}\quad
    l(\mathbf{y^{(i)}}, \hat{\mathbf{y}}^{(i)}) = - \sum_{j=1}^q y^{(i)}_j \log \hat{y}^{(i)}_j
$$
