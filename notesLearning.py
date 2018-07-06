"""
***** Unsupervised vs. Supervised Learning *****

Unsupervised: you don't give it any answers it can learn from.
> these algorithms try to find patterns in your data on their own
> these pattern discoverd for example in "clustering" are called latent variablkes
> examples: clustering movies via their properties and look by what mertrics they cluster together

Supervised: we have a set of answers the model can learn from
> Evaluation: two groups, a larger group of training data and test data that the model
  hasn't seen yet to evaluate how good the model predicts unseen data
> k
"""

"""
***** How does clustering work? *****

k-Means clustering:
> choose k random points in the attributes vector
> each of these "centroids" has data points that are nearest to it
> move the centroids to the middle of the data points that were previously nearest to it
> again, with the new centroid, assign data points of which the centroid is the nearest, to it
> the data points change their assigned centroid probably
> iterate this until the centroid don't move much anymore with each iteration
> this is because they converged into the middle point of data where this cluster has
  some kind of distance to another cluster
> this way, latent variables can be found

How large should k be?
> start with a low k
"""

"""
***** The concept of entropy *****

Definition:
> verbally, measure of "sameness" of dataset
> e.g. data set consists of of things ordered into classes. If most observations belong into one class, entropy is low
> but are there several groups/classes that are roughly the same size, the data set is more disordered and has therefore higher entropy
"""
