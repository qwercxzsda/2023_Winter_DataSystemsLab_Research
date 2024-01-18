# Chapter 9 Summary

Recurrent neural networks (RNNs) are deep learning models that capture the dynamics of sequences via recurrent connections, which can be thought of as cycles in the network of nodes.

RNNs can be thought of as feedforward neural networks where each layer’s parameters (both conventional and recurrent) are shared across time steps.

## 9.1. Working with Sequences

Now, the inputs are an ordered list of feature vectors $\mathbf{x}_1, \dots, \mathbf{x}_T$, where each feature vector $\mathbf{x}_t$ is indexed by a time step $t \in \mathbb{Z}^+$ lying in $\mathbb{R}^d$.

These cases are possible when working with sequences:

1. sequentially structured input, fixed target
1. fixed input, sequentially structured target
1. sequentially structured input, sequentially structured target
    1. aligned: the input at each time step aligns with a corresponding target (e.g., part of speech tagging)
    1. unaligned: the input and target do not necessarily exhibit a step-for-step correspondence (e.g., machine translation).

### 9.1.1. Autoregressive Models

We are interested in knowing the probability distribution

$$
    P(x_t \mid x_{t-1}, \ldots, x_1)
$$

Or some statistic(s) of this distribution. e.g., the mean.

$$
    \mathbb{E}[(x_t \mid x_{t-1}, \ldots, x_1)]
$$

Such models that regress the value of a signal on the previous values of that same signal are naturally called `autoregressive models`.

There is one major problem: the number of inputs, $x_{t-1}, \ldots, x_1$ varies depending on $t$. There are a few strategies to resolve the issue.

1. Content ourselves to condition on some window of length $\tau$ and only use $x_{t-1}, \ldots, x_{t-\tau}$ observations.

    Now, the number of arguments is always the same, at least for $t > \tau$. Thus, we can use any linear model or deep network that requires fixed-length vectors as inputs.

1. Develop models that maintain some summary $h_t$ of the past observations and at the same time update $h_t$ in addition to the prediction $\hat{x}_t$.

    Now, the models must estimate not only $x_t$ with $\hat{x}_t = P(x_t \mid h_{t})$ but also updates of the form $h_t = g(h_{t-1}, x_{t-1})$. Since $h_t$ is never observed, these models are called `latent autoregressive models`.

To construct training data from historical data, one typically creates examples by sampling windows randomly.

In general, we do not expect time to stand still. However, we often assume that the dynamics, from which each subsequent observation is generated, do not change. Statisticians call dynamics that do not change `stationary`.

### 9.1.2. Sequence Models

Sometimes, especially when working with language, we wish to estimate the joint probability of an entire sequence.

$$
    P(x_1, \ldots, x_T) = P(x_1)
$$

The field of sequence modeling has been driven so much by natural language processing, that we often describe sequence models as "language models",
even when dealing with non-language data.

Language modeling gives us not only the capacity to `evaluate` likelihood, but also the ability to `sample` sequences, and even to optimize for the most likely sequences.

We can reduce language modeling to autoregressive prediction by decomposing the joint density  of a sequence $p(x_1, \ldots, x_T)$ into the product of conditional densities in a left-to-right fashion by applying the chain rule of probability:

$$
    P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}, \ldots, x_1)
$$

Note that if we are working with discrete signals such as words, then the autoregressive model must be a probabilistic classifier, outputting a full probability distribution over the vocabulary for whatever word will come next, given the leftwards context.

#### 9.1.2.1. Markov Models

For this model, we only use $\tau$ previous time steps, i.e., $x_{t-1}, \ldots, x_{t-\tau}$, rather than the entire sequence history $x_{t-1}, \ldots, x_1$.

If we only need $\tau$ steps for the prediction, we say that the sequence satisfies a `Markov condition`.

When $\tau = 1$, we say that the data is characterized by a first-order Markov model. When $\tau = k$, we say that the data is characterized
by a $k^{\textrm{th}}$-order Markov model.

When $\tau = 1$, following holds.

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}).$$

We often assume the Markov condition for simplicity. Even today's massive RNN- and Transformer-based language models seldom incorporate more than thousands of words of context.

With discrete data, we simply count the number of times
that each word has occurred in each context, producing
the relative frequency estimate of $P(x_t \mid x_{t-1})$.

#### 9.1.2.2 The Order of Decoding

We don't need to factorize the sequence $P(x_1, \ldots, x_T)$ left-to-right. In principle, there is nothing wrong with unfolding $P(x_1, \ldots, x_T)$ in reverse order. e.g.,

$$
    P(x_1, \ldots, x_T) = P(x_T) \prod_{t=T-1}^1 P(x_t \mid x_{t+1}, \ldots, x_T)
$$

However, there are many reasons to factorize text in the same direction in which we read it (left-to-right for most languages, but right-to-left for Arabic and Hebrew).

1. It is more natural.

   In everyday life, we observe a sequence and predict what comes next.

1. We can assign probabilities to arbitrarily long sequences using the same language model.

    Extending the probability over steps $1$ through $t$ into $1$ through $t+1$ is simple:
    $$
        P(x_{t+1}, \ldots, x_1) = P(x_{t}, \ldots, x_1) \cdot P(x_{t+1} \mid x_{t}, \ldots, x_1)
    $$

1. Predicting adjacent words is easier than predicting words at arbitrary other locations.

    This is true for all kinds of data. e.g., when the data is causally structured.

### 9.1.4. Prediction

For an observed sequence $x_1, \ldots, x_t$, predicting the output $\hat{x}_{t+k}$ at time step $t+k$ is called the $k$*-step-ahead prediction*.

As $k$ increases, errors accumulate and the quality of the prediction degrades, often dramatically.