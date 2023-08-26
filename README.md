# Fake/Real News Clustering

## Background
This project involves clustering analysis on both text and image data using various machine learning techniques. The primary goal is to explore the effectiveness of clustering algorithms in grouping similar data points together.

1. [Load Data and Initial Clustering](#Load-Data-and-Initial-Clustering)
2. [Visualizing Clusters](#Visualizing-Clusters)
3. [Performance Evaluation](#Performance-Evaluation)
4. [Finding Optimal K](#Finding-Optimal-K)
5. [PCA Followed by K-Means](#PCA-Followed-by-K-Means)
6. [MNIST Clustering](#MNIST-Clustering)
---

## Files
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
* Fake.csv
* True.csv



###1. Load Data and Initial Clustering <a name="load-data-and-initial-clustering"></a>

Here, we load text data from CSV files ("Fake.csv" and "True.csv"), perform text preprocessing, apply TF-IDF transformation, and cluster the data using K-Means algorithm. We label the data as "fake" or "real" and use TF-IDF features for clustering.

###2. Visualizing Clusters <a name="visualizing-clusters"></a>

We visualize the clusters created by K-Means using Principal Component Analysis (PCA) for dimensionality reduction. The scatter plots and 3D scatter plots display the clustering results and the positions of cluster centers.


###3. Performance Evaluation <a name="performance-evaluation"></a>

We evaluate the performance of clustering based on accuracy metrics. We calculate accuracy considering both cases where clusters match the labels and where clusters are opposite to the labels.

###4. Finding Optimal K <a name="finding-optimal-k"></a>

We explore finding the optimal number of clusters (k) using the Elbow method. We plot the Sum of Squared Distances (SSD) score as a function of the number of clusters and the Silhouette score to help determine the suitable value for k.

###5. PCA Followed by K-Means <a name="pca-followed-by-k-means"></a>

We apply PCA to the text data before performing K-Means clustering. We visualize the results of PCA followed by K-Means and analyze whether it affects the clustering outcome.

###6. MNIST Clustering <a name="mnist-clustering"></a>

In this section, we switch to image data using the MNIST dataset. We load, preprocess, and standardize the data for K-Means clustering. We evaluate the clustering performance by calculating accuracy and label majorities for each cluster.
