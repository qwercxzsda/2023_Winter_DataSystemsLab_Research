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

## 4.4. Softmax Regression Implementation from Scratch

### 4.4.3. The Cross-Entropy Loss

```python
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

The result of the above code is `tensor([y_hat[0][0], y_hat[1][2]])`.

As a result, cross entropy $E[l(\mathbf{y^{(i)}}, \hat{\mathbf{y}}^{(i)})]$ is calculated as follows.

```python
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()
```

To be precise, $i^{th}$ term of `-torch.log(y_hat[list(range(len(y_hat))), y])` is the term $l(\mathbf{y^{(i)}}, \hat{\mathbf{y}}^{(i)}) = - \sum_{j=1}^q y^{(i)}_j \log \hat{y}^{(i)}_j$.
