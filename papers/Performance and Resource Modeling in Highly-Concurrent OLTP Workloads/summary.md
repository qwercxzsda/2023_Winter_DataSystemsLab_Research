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
