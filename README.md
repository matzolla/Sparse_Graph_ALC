# Sparse Graph agglomerative Likelihood Clustering

[Allassan Tchangmena](), [Lionel Yelibi]()

In this project, we propose a sparse aglomerative clustering algorithm, to cluster nodes from a graph with similar properties (base on their `weights`
and `edges`. The criterion used for clustering is a likelihood function inspired from [Giada and Marsili](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.63.061101) that has been relaxed to suit graph clustering problems. Given the egde $\epsilon$ and the weight $w$ of nodes in a graph. The likelihood function can be computed using the equation:

$$ Likelihood= \frac{1}{2}\sum_{n_{s}>1}\left[ln\left(\frac{\epsilon}{w}\right) + \left(\epsilon-1\right)ln\left(\frac{\epsilon^{2}-\epsilon}{\epsilon^{2}-w}\right)\right]$$

### Results

|  Before clustering                                               |    After clustering                            |
|---------------------------------------------------------------   |  --------------------------------------------- |
| ![Original](images/Original_graph.png)                           |        ![result](images/results.png)           |

### Setup

You can clone the github locally using the command
```

```
