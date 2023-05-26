# Sparse Graph agglomerative Likelihood Clustering

[Allassan Tchangmena](), [Lionel Yelibi]()

In this project, we propose a sparse aglomerative clustering algorithm, to cluster nodes from a graph with similar properties (base on their `weights`
and `edges`. The criterion used for clustering is a likelihood function inspired from [Giada and Marsili](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.63.061101) that has been relaxed to suit graph clustering problems. Given the egde $epsilon$ and the weight $w$ of nodes in a graph. The likelihood function can be computed using the equation:

$$ Likelihood= \frac{1}{2}\sum_{n_{s}>1}\left[ln\left(\frac{\epsilon}{w}\right)\right]$$
