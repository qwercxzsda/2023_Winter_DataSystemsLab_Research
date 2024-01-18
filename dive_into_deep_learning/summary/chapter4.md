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

## 4.6. Generalization in Classification

### 4.6.1. The Test Set

The `empirical error` of our classifier $f$ on $\mathcal{D}$ is simply the fraction of instances for which the prediction $f(\mathbf{x}^{(i)})$ disagrees with the true label $y^{(i)}$, and is given by the following expression:

$$
    \epsilon_\mathcal{D}(f) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)})
$$

By contrast, the `population error` is the expected fraction
of examples in the underlying population
(some distribution $P(X, Y)$  characterized
by probability density function $p(\mathbf{x},y)$)
for which our classifier disagrees
with the true label:

$$
    \epsilon(f)
    = E_{(\mathbf{x}, y) \sim P} \mathbf{1}(f(\mathbf{x}) \neq y) = \int\int \mathbf{1}(f(\mathbf{x}) \neq y) p(\mathbf{x}, y) \;d\mathbf{x} dy
$$

We want to know $\epsilon(f)$. However, this is impossible. Thus, we use $\epsilon_\mathcal{D}(f)$ instead. Mathematically, we can view $\epsilon_\mathcal{D}(f)$ as a statistical estimator of the population error $\epsilon(f)$.

Using statistics, we can derive that if we want to fit two standard deviations
in that range and thus be 95% confident that $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$, then we will need 10,000 samples with asymptotic assumptions, or 15,000 samples without any assumptions.

### 4.6.2. Test Set Reuse

You can't have separate test sets for every model you train. This problem relates to multiple hypothesis testing, which despite a vast literature in statistics, remains a persistent problem plaguing scientific research.

In practice, take care to create real test sets and consult them as infrequently as possible.

### 4.6.3. Statistical Learning Theory

To deal with the generalization problem, there is a concept called `uniform convergence`. Uniform convergence means 'with high probability, the empirical error rate for every classifier in the class $f\in\mathcal{F}$ will *simultaneously* converge to its true error rate'.

In other words, we seek a theoretical principle that would allow us to state that with probability at least $1-\delta$ (for some small $\delta$) no classifier's error rate $\epsilon(f)$ (among all classifiers in the class $\mathcal{F}$) will be misestimated by more than some small amount $\alpha$. Clearly, this does not hold for all model classes $\mathcal{F}$. The class of memorization machines is a counterexample.

There is a theory by Vapnik and Chervonenkis called `VC dimension` that can estimate the sample size needed to achieve uniform convergence(but the estimated sample size is much bigger than the actual size).
