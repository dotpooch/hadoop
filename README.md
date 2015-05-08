hadoop
======

This is a collection of machine learning apps meant to run on hadoop using the python library MRJob, created by Yelp.

K-Means
------------------------------------------------------
k-means is an unsupervised learning algorithm that clusters vectors by creating a centroid to represent the cluster.  The algorithm requires the number of clusters from the user.  It randomly chooses vectors to serve as the initial cluster centroids and calculates the distance from each centroid to all of the other vectors.  The vectors are then clustered into groups having the same closest centroid.  The vectors of each cluster are averaged to create a new centroid and the distances from these centroids to all vectors are then recalculated.  Again the vectors are clustered into groups with the closest centroid.  This process of averaging, calculating the distance, and reclustering is repeated until none of the vectors change clusters.

The algorithm is heavily dependent on the initial random centroids chosen.  It can sometimes find the local minimum and not absolute minimum.  It needs to be run several times to determine if the correct absolute centroids were found.  The most frequent centroids returned can be considered the absolute centroids.  This algorithm assumes only 2 clusters but this can be easily changed in the code.  If the number of clusters are not known, the algorithm needs to be run for multiple cluster numbers and an error rate needs to be calculated.  These error rates can be plotted against the number of clusters and best cluster number can be chosen by determining the cluster where the error begins the level off (aka the elbow).

On hadoop, k-means is run by dispersing the vectors across the network and then copying the centroid file to each server while keeping the vectors on their same severs.  This maximizes the speed by reducing the amount of data that is transferred back and forth.  In MRJob, this is done by writing the centroids to disk after processing and reading them during the mapper_init process.  There is a fair amount of knowledge required to write this in MRJob.  Once the concepts of Map Reduce are understood, one needs to know how to run a sequence of jobs programmatically in MRJob.  One should have a good understanding of this process by reviewing both the MRJob documentation and this code.  This code works locally but there may be issue running it on Amazon EC2.

The Map Reduce Sequence:
  1. Count the Number of Vectors
  2. Determine the Initial Random Centroids
  3. Recalculate the Centroids (repeat) (map: determine nearest centroid, reduce: recalculate centroids)



Affinity Propagation
------------------------------------------------------
Affinity propagation is an unsupervised learning algorithm that clusters vectors by approximating log probability that one vector can be represented by another vector in the corpus.  Unlike k-means which calculates and average of the points in each cluster, affinity propagation chooses an existing vector that can serve as the centroid of the cluster.  It, in fact, is not the centroid but the exemplar of the cluster and like the centroid can be treated as the compressed point which represents all the points in that cluster.  Also unlike k-means, affinity propogation does not require that the user predefine the number of clusters.  This can be very helpful when the data set and the number of clusters is large because the algorithm only needs to be run once.

The algorithm is slow and works by calculating the euchlidean distince between every point in the dataset and these values are assigned to the 'similarity' matrix.  This matrix is then modified by by replacing the diagonals with the median distance and adding a small amount of noise to all the off-diagonal values.  Three other matrices, 'availability','responsibility', and 'criterion' are also created and filled with zeros.  The 'similarity' and 'availability' are added together and these values are multipled by a dampening factor and added to update the 'responsibility' matrix. This 'responsibility' matrix, also multiplied by the dampening factor, then updates the 'availabilty' matrix.  The new 'responsibility' and new 'availbility' are added to create a 'criterion' maxtrix.  The column with the max for each row is determined and these columns represent the examplars.  This process is repeated until the examplars remain stable for a predefined number of iterations.

The formulas are rather complex but psuedo formulas give an accurate overview:

responsibility(row,column) = similarity(row,column) - max{(availability + similarity)(row) excluding(row,column)}

availability(k,k)          = max{0,positive responsibility(∑columns positive values excluding itself)}

availability(row,column)   = min(0,availability(k,k) + max(∑all positive elements of each column excluding in responsibility(row,column) if > 0, 0)

criterion                  = responsibility(row,column) + availability(row,column)

The dampening percentage is the learning rate of the algorithm.  If this too low, then the algorithm may overshoot the convergence point and oscillate.  The small amount of noise added to the similarity matrix eliminates outliers.  For example, this can happen if similarities are symmetric and two data points are isolated from the rest - then, there may be some difficulty deciding which of the two points should be the exemplar and this can lead to oscillations.
