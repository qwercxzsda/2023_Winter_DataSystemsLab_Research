# Performance and Resource Modeling in Highly-Concurrent OLTP Workloads (summary)

## ABSTRACT

Due to high degrees of concurrency, competition for resources, and complex interactions between transactions, resource prediction and performance analysis are difficult.

However, such analysis is a key component in understanding which queries are eating up the resources, and how their system would scale under load.

The framework introduced in the paper, called `DBSeer`, addresses this problem by employing statistical models that provide resource and performance analysis and prediction for highly concurrent OLTP workloads.

## 1. INTRODUCTION

In DBSeer, we develop two classes of models and compare their performance.

1. Black-box models

   Make minimal assumptions about the nature of the underlying system.

   Train statistical regression models to predict future performance based on past performance statistics.

   More general but less effective in making predictions outside of the range of inputs on which they were trained.

1. White-box models

   Take the major components of the underlying database system into account.

   Less general than black-box models (as they make assumptions about the nature of the database) but provide higher extrapolation(추론) power.

We make several contributions towards modeling transactional workload, including:

1. Resource Models

   We have developed white and black-box models for predicting different resources, including CPU, RAM, network, disk I/O, and lock contention.

   Our primary contribution here is a set of novel white-box models for predicting disk I/O and lock contention.

1. Extracting transaction types

   We have developed highly accurate clustering techniques, automatically extracting and summarizing “classes of similar transactions” from a query log, that allow us to accurately group similar transactions.

   This clustering is able, for example, to identify the 5 transaction classes in TPC-C, and the major query types in Wikipedia.

1. Evaluation

   We evaluate our models on a real database system, both using the well-known TPC-C benchmark and the real-life traces of Wikipedia, showing that we can predict the maximum throughput within 0-25% error.

   Additionally, we show that white-box models can avoid over-provisioning by at least 9× and predict disk I/O from 4× to 100× more accurately than simple black-box models when predicting resource utilization over a wide range of transaction rates.

## 2. SOLUTION OVERVIEW

In this paper, we focus on the problem of resource prediction. Given a set of transaction types (we describe our method for deriving these below) running at a certain rate (transactions per second, or TPS), with a certain mixture (fraction of each transaction type in the overall workload), the goal is to predict the CPU usage, disk I/O, minimum amount of RAM, network consumption, and time spent in lock contention.

### 2.1 DBSeer Overview

![Figure 1: Workflow in DBSeer.](images/Figure%201.%20Workflow%20in%20DBSeer..png)

DBSeer consists of the following steps, shown in Figure 1.

1. Collecting Logs.

   We observe a DBMS during normal operation, i.e., running without modification in its production state. We collect standard SQL query logs, as well as various DBMS and OS statistics (over periods of hours or days).

2. Preprocessing / Clustering.

   We align (by time) and join the various logs, and extract a set of transaction types to categorize the types/classes of transactions that the system runs.

3. Modeling.

   We build white- and black-box models to predict the resource utilization (CPU, RAM, disk I/O, Locks, etc.) of the system for different mixes and rates of transaction types.

All of our models accept a mixture $(f_1, ... f_J)$ and a target TPS $T$, where $f_i$ represents the fraction of the total transactions run from type $i$ and $J$ is the total number of types.

We can observe the system can answer what-if and attribution questions about never-seen-before mixtures and rates.

## 3. PREPROCESSING

### 3.1 Gathering the Log and Statistics

1. The SQL statements and transactions executed by the system (that will be clustered using our clustering mechanism).

2. The run-time(latency) of each transaction.

3. Aggregate OS stats, including per-core CPU usage, number of I/O reads and writes, number of outstanding asynchronous I/Os, total network packets and bytes transferred, number of page faults, number of context switches, CPU and I/O usage.

4. Global status variables from MySQL including the number of `SELECT`, `UPDATE`, `DELETE`, and `INSERT` commands executed, number of flushed and dirty pages, and the total lock wait-time.

As we are focused on non-intrusive logging, we do not collect any statistics that significantly slow down performance, such as fine-grained locking information.

### 3.2 Transaction Clustering

#### Extracting Transaction Summaries

`transaction summary` is defined as follows.

$
    [t_0(mode_1,table_1,n_1,t_1), \cdots, (mode_k, table_k, n_k, t_k)]
$

- $k$

  The number of tables accessed by the transaction (e.g., if it accesses table a, then table b, and then again table a, we have k = 3)

- $t_i$

  $t_0$ is the time between the BEGIN and the first SQL statement.

  $t_k$ is the time between the last SQL statement and the final `COMMIT`.
  
  For $1 \le i \lt k$, the time lag between the completion of the SQL statement causing the $i$’th access and the time that the $(i+1)$’th statement was issued. −1 when both the $i$’th and the $(i+1)$’th table accesses are caused by the same SQL statement.

- $table_i$

  The table accessed by the transaction.

- $mode_i$

  $w$ when accessing $table_i$ requires an exclusive lock (e.g. `DELETE`, `UPDATE`, `INSERT` or `SELECT...FOR UPDATE`).
  
  $r$ if the access requires a read-only/shared lock (e.g. general `SELECT`).

- $n_i$

  The approximate number of rows accessed from $table_i$.

All of this information can be obtained from the SQL logs except the number of rows read or written from each table, which we estimate using the query rewriting technique described in Section 3.3.

#### Learning Transaction Types

Given the transaction summaries, we use the extracted features and apply the DBSCAN clustering algorithm to group individual transactions based on their accesses.

The input data consists of one row per transaction. Each row contains a set of transaction features. The feature set consists of 2 attributes for each table in the database, one for the number of rows read from and one for the number of rows updated/inserted in each table (many of these features will be zero as most transactions do not access all the tables).

The output of this clustering is a workload summary that lists the major transaction types (along with representative queries) and provides their frequencies in the base workload.

### 3.3 Estimating Access Distributions

Our second use of the logs is to infer a rough probability distribution over all the pages in the database by access (read or write) and by transaction type—this is used in our locking and I/O prediction models to estimate conflict and update probabilities.

## 4. MODELING DISK I/O AND RAM

In this section, we present our white-box models for disk I/O and RAM provisioning.

Disk and memory are important aspects of performance in a database system. In a transactional database, such as MySQL, disk I/Os and RAM utilization are closely related to one another. The three main causes for disk I/Os are:

1. Logwrites

   needed to guarantee transactionality.

1. Dirty pages write backs (log-triggered data flushes)

   needed to bound recovery time, and allow transactionally-consistent reclamation of log files (all pages dirtied by transactions logged in the current log file need to be flushed to disk before the log file can be recycled).

1. I/Os due to cache misses (capacity misses)

   needed to read pages from disk that were not cached in the buffer pool, and possibly trigger eviction of dirty pages (which need to be written back to disk first).

These operations heavily depend on the size of the buffer pool: a smaller buffer pool leads to more I/Os.

### 4.1 Background on Disk I/O in a DBMS

Log writes are proportional to the rate of each transaction type in the load. Thus, log writes can be easily modeled with linear regression.

Therefore, in the rest of this section, we focus on writes that are due to dirty page write-backs, or “flushes”. Flushing happens for two main reasons.

1. capacity misses

   A new page is brought into the buffer pool and there are no free buffers, forcing an existing record to be flushed.

1. log-triggered data flushes

   The redo log file(s) is full and needs to be rotated or recycled.

`log-triggered data flushes` use multiple heuristics for performance reasons. In MySQL, this process is referred to as adaptive flushing of the buffer pool pages. Modeling the net effect of all these complex heuristics is a challenging task, and is one of our main contributions in this paper which is described next.

### 4.2 Disk Write Model

Different DBMSs use different heuristics for maintaining the balance between eagerly writing pages or lazily flushing them at the log rotation time.

In the following, we provide a simple analysis based on the conservation of flow that abstracts the internal details.

#### Probability of a page being dirtied

- $D$: the number of pages in the database

- $\tilde{f} = (f_1, \cdots f_l)$: the mixture of transactions

- $\tilde{p}_{write, i}$: the probability a transaction drawn from $\tilde{f}$ writes to the $i$’th page

- $\tilde{p}_{write}$: probability distribution over all the pages in the database.

For simplicity, we assume that a transaction only accesses one page. Thus, the expected number of pages dirtied by a single transaction is as follows.
$$
    \sum_{i=1}^D \tilde{p}_{write, i} = 1
$$

- $T_n$: the number of unique written(dirty) pages after executing $n$ transactions, given $\tilde{f}$, $\tilde{p}_{write}$

- $T_{n, i}$: the probability of the $i$’th page being written(dirtied)

Assuming that different transactions arrive independently of each other, we can write $T_n$ as follows.

$$
    T_n = \sum_{i=1}^D T_{n, i}
$$

As $(1 - \tilde{p}_{write, i})^n$ is the probability that the $i$’th page is not written by any of the $n$ transactions, we can write $T_{n, i}$ as follows.

$$
    T_{n, i} = 1 - (1 - \tilde{p}_{write, i})^n
$$

We will model the log rotation process using the above probabilities. At any point in time, every page falls into exactly one of these three categories.

1. $C_1$: a page is dirty and the first dirtying transaction (i.e., the transaction that made it dirty) is logged in the old log

1. $C_2$: a page is dirty and its first dirtying transaction is logged in the current (new) log

1. $C_3$: a page is still clean (i.e. is identical to its copy on the disk)

Q. old log와 current (new) log를 구분하는 기준은 무엇인가?

- $P_{1, i}, P_{2, i}, P_{3, i}$: the probability of the $i$’th page in the category $\{C_1, C_2, C_3\}$, respectively. For $\forall i \in \{1, \cdots, n\}$, $P_{1, i} + P_{2, i} + P_{3, i} = 1$

- $d_{1, t}, d_{2, t}, d_{3, t}$: the number of pages in the category $\{C_1, C_2, C_3\}$ at time $t$, respectively

- $L$: the maximum capacity of each log file

The following holds for log rotation.

1. The log needs to be rotated at least as often as every L transaction.

1. The log rotation can only happen at time $t$ if $d_{1,t} = 0$.

   Otherwise, a system crash could lead to data loss in any of the pages in category $C_1$.

1. If a log rotation happens at t, we will have $d_{2,t+1} = 0$.

   i.e., the new log will be empty at the beginning of time $t + 1$.

#### Abstracting the main idea behind MySQL’s adaptive flushing

- $F_t(n)$: the flushing rate chosen by adaptive flushing, given that the system is running $n$ transactions per second(TPS)

The main idea behind MySQL’s I/O heuristics, such as adaptive flushing is the following.

The flush rate (flow of pages out) should roughly match the rate at which pages are dirtied, such that at the time a log rotation happens, there will be no dirty pages waiting to be flushed.

- $l_t$: the current capacity of the new log at time $t$ $(0 \lt l_t \le L)$

The new log is expected to get full in $l_t / n$ seconds, over which $d_{1,t}$ pages need to be flushed back to disk.

i.e., we need to flush all the `old logs` before the `new log` gets full.

$$
    F_t(n)
    = \frac{d_{1, t}}{l_t / n}
    = \frac{d_{1, t} \cdot n}{l_t}
$$

#### Estimating flush rate (Monte-Carlo baseline)

- $F(n)$: the expected flush rate, given TPS of $n$ $(E[F_t(n)])$

We need to predict $F(n)$, without directly observing $d_{1, t}$ and $l_t$.

Monte-Carlo simulation is a way to estimate $d_{1, t}$ and $l_t$, but it is too slow to use in implementation.

#### Estimating flush rate (Iterative approach)

Q. What is log rotation?
A. old log -> 없어짐, new log -> old log 를 log rotation이라고 하는 것 같다.

Given that there is no log rotation at time $t$ and $t + 1$, we have:

$$
    l_{t + 1} = l_t - n
$$

$$
    d_{1, t + 1} = d_{1, t} - F_t(n)
$$

The number of dirty pages $(d_{1,t})$ can only decrease. This is because the old log is no longer being appended.

$$\begin{aligned}
    F_{t+1}(n)
    &= \frac{d_{1, t+1} \cdot n}{l_{t+1}}
    = \frac{(d_{1, t} - F_t(n)) \cdot n}{l_t - n}\\
    &= \frac{(d_{1, t} - \frac{d_{1, t} \cdot n}{l_t}) \cdot n}{l_t - n}
    = \frac{d_{1, t} \cdot n}{l_t}
    = F_t(n)
\end{aligned}$$

Thus, in the absence of log rotations, the following holds.

$$
    F_t(n) = \frac{d_{1, t_0} \cdot n}{L}
$$

Where $t_0$ is the time step immediately after any log rotation (i.e. log
was rotated at $t_0 − 1$) and $t$ is a time step before the next log rotation $(t_0 \le t \lt t_0 + \frac{L}{n})$.

This is because the new log is empty right after a log rotation $(l_{t_0} = L)$.

Since $n$ and $L$ are time-independent, we only need to estimate $E[d_{1,t}]$.

$$
   E[d_{1,t}] = \sum_{j=1}^D P_{1, j} \quad
   E[d_{2,t}] = \sum_{j=1}^D P_{2, j} \quad
   E[d_{3,t}] = \sum_{j=1}^D P_{3, j}
$$

Thus, we need to estimate $P_{1, j}$, $P_{2, j}$, and $P_{3, j}$. However, these variables are interdependent, since as pages are dirtied and flushed to disk, the probabilities change.

- $\{P_{i, j, t}\}$: the probability of the $j$’th page being in the category $C_i$ $(i = 1, 2, 3)$ at time $t$ $(t = 0, 1, 2, \cdots)$. i.e., three time series $\{P_{1, j, t}\}$, $\{P_{2, j, t}\}$, $\{P_{3, j, t}\}$.

For $m \ge 1$, the following holds.

$$
   P_{1, j, t + m} = P_{1, j, t} \cdot (1 - \frac{n}{L})^m
$$
$$
   P_{2, j, t + m} = 1 - P_{1, j, t + m} - P_{3, j, t + m}
$$
$$
   P_{3, j, t + m} =
      \left\{\begin{matrix}
      P_{3,j,t} \cdot (1-\frac{n}{L})^m
      + P_{1,j,t} \cdot \frac{n}{L} \cdot m \cdot \frac{(1-\frac{n}{L})^m}{1-\frac{n}{L}}
      \quad & \text{if } T_{n, j} = \frac{n}{L}
      \\
      \\
      P_{3,j,t} \cdot (1-T_{n,j})^m
      + P_{1,j,t} \cdot \frac{n}{L} \cdot \frac{(1-T_{n,j})^m - (1-\frac{n}{L})^m}{\frac{n}{L}-T_{n,j}}
      \quad & \text{if } T_{n, j} \ne \frac{n}{L}
      \end{matrix}\right.
$$

Our algorithm uses these equations by iteratively incrementing $t$, until they converge, thus estimating the values of $P_{1, i}$, $P_{2, i}$, and $P_{3, i}$.

Q. time series $\{P\}$가 dirtied, flushed에 따라 바뀌는데 어떻게 $t$가 증가함에 따라 수렴하는 거지?
A. rotate된 직후의 $P$만 고려하는 것 같다. 즉, 각 iteration마다 rotate가 되도록 time step을 증가시킨다고 생각할 수 있다.

#### 4.2.1 Algorithm

- line 2.4 ~ 2.6: estimate $P_{1, i}$, $P_{2, i}$, and $P_{3, i}$ using above equations
- line 2.3: estimate $F(n)$ using $P_{1, i}$
- $avgF$: estimator of $F(n)$

![Figure 2: Expected I/O (flush rate) prediction algorithm (without page clustering).](images/Figure%202.%20Expected%20IO%20(flush%20rate)%20prediction%20algorithm%20(without%20page%20clustering)..png)

#### Optimization, clustering similar pages

One major source of time and space complexity in the above algorithm is the total number of pages $D$. However, one can cluster the pages based on the $p_{write,i}$ values.

In the optimized version, we first cluster $D$ into $K(D,\epsilon)$ partitions (non-overlapping clusters) such that the $i$’th and $j$’th pages fall within the same partition if and only if $∣p_{write,i} − p_{write,j}∣ \le \epsilon$

In most cases, $\epsilon=0$ provides sufficient clustering.

Q. 어떻게 $p_{write,i}$를 알 수 있지?

### 4.3 Disk Reads and RAM Provisioning

We built a Monte-Carlo simulation of the buffer pool to estimate the capacity miss.

To estimate the miss rate for a database with $N$ pages of RAM, we allocate an $N$ element list $bp$. Then, we simply simulate `LRU` (or `LRU2` in the case of MySQL).

- $C_{read}$: number of read misses
- $C_{write}$: number of flushes (when dirty pages are evicted)

Using the page access distributions for each transaction type, we derive a `combined access distribution`.

- `combined access distribution`: represents the probability of each of the $D$ pages in the database being touched by the input mixture.

We then simulate accesses to these $D$ pages of the database by randomly selecting pages according to this combined distribution.

When a page is accessed, it is added to the head of $bp$ if it is not already present. Otherwise, it is moved to the head of the list. When a page is added a counter $C_{read}$ is incremented. If the access is a write, a bit on the page is set to mark it as dirty. If $bp$ already contains $N$ elements, the last element (tail) of $bp$ is removed, and if the dirty bit is set, a counter $C_{write}$ is incremented. This simulates the LRU cache eviction policy. We can then compute the number of page reads and flushes per second by dividing $C_{read}$ and $C_{write}$ by the TPS.

Combining our cache and log rotation models, our I/O model predicts the following, given that we are running $n$ TPS for a time period $t$.
- read: $C_{read} \cdot t \cdot n$ data pages
- write-back: $C_{write} \cdot t \cdot n + F_t(n)$ data pages, in addition to any sequential log I/O.

## 5. LOCK CONTENTION MODEL

Given a mixture of concurrent transactions, we want to predict the expected delay. We develop a model of two-phase locking (2PL) for this purpose.

### 5.1 A Summary of Thomasian’s 2PL Analysis

- $D_1, \cdots, D_n$: The set of non-overlapping regions that consists of the database.
- $C_1, \cdots, C_n$: The set of transaction classes.
- $K_j$: The number of locks required by a transaction in class $C_j$. Thus, the transaction takes $K_j + 1$ steps to complete.
- $S_{j,n}$: The mean processing times for the $n$’th step of a transaction of type $C_j$. The processing time is exponentially distributed.
- $g_{j,n,i}$: The probability that a transaction of class $C_j$ access the $i$’th region in the $n$’th step $(\sum_{i=1}^I g_{j,n,i} = 1)$.
- $U_{j,n}$: The mean delay incurred by a transaction from $C_j$ due to encountering a lock conflict at its $n$’th step.
