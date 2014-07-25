hadoop
======

This is a collection of machine learning apps meant to run on hadoop using the python library MRJob, created by Yelp.

K-Means
------------------------------------------------------
k-means is an unsupervised learning algorithm that clusters vectors by creating a centroid to represent the cluster.  The algorithm requires the number of clusters from the user.  It randomly chooses vectors to serve as the initial cluster centroids and calculates the distance from each centroid to all of the other vectors.  The vectors are then clustered into groups having the same closest centroid.  The vectors of each cluster are averaged to create a new centroid and the distances from these centroids to all vectors are then recalculated.  Again the vectors are clustered into groups with the closest centroid.  This process of averaging, calculating the distance, and reclustering is repeated until none of the vectors change clusters.

The algorithm is heavy dependent on the initial random centroids chosen.  It can find local minimum and not absolute minimum.  It needs to be run several times to determine if the correct absolute centroids were found.  The most frequent centroids returned can be considered the absolute centroids.  The algorithm assumes only 2 clusters but this can be easily changed in the code.  If the number of cluster are not known, the algorithm needs to be run for multiple to cluster numbers and an error rate needs to be calculated.  These error rates can be plotted against the number of clusters and best cluster number can be chosen by determining the cluster where the error begin the level off.

On hadoop, k-means is run by dispersing the vectors across the network and then copying the centroid file to each server while keeping not moving the vectors.  This maximizes the speed by reducing the amount of data that is transferred back and forth.  In MRJob, this is done by writing the centroids to disk after processing and reading them during the mapper_init process.  There is a fair amount of knowledge required to write this in MRJob.  Once the concepts of Map Reduce are understood, one needs to know how to run a sequence of jobs programmatically in MRJob.  One should have a good understanding of this process by reviewing both the MRJob documentation and this code.  This code works locally but there may be issue running it on Amazon EC2.

The Map Reduce Sequence:
  1. Count the Number of Vectors
  2. Determine the Initial Random Centroids
  3. Recalculate the Centroids (repeat) (map: determine nearest centroid, reduce: recalculate centroids)



Affinity Propagation
------------------------------------------------------
Affinity Propagation