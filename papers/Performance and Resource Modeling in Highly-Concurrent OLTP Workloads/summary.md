# Performance and Resource Modeling in Highly-Concurrent OLTP Workloads (summary)

## ABSTRACT

Due to high degrees of concurrency, competition for resources, and complex interactions between transactions, resource prediction and performance analysis are difficult.

However, such analysis is a key component in understanding which queries are eating up the resources, and how their system would scale under load.

The framework introduced in the paper, called `DBSeer`, addresses this problem by employing statistical models that provide resource and performance analysis and prediction for highly concurrent OLTP workloads.
