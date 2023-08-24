Text and Image Data Clustering

This project involves clustering analysis on both text and image data using various machine learning techniques. The primary goal is to explore the effectiveness of clustering algorithms in grouping similar data points together.
Table of Contents

    Imports
    Load Data and Initial Clustering
    Visualizing Clusters
    Performance Evaluation
    Finding Optimal K
    PCA Followed by K-Means
    MNIST Clustering

1. Imports <a name="imports"></a>

In this section, we import the necessary libraries and packages for our analysis. We use tools like numpy, pandas, matplotlib, and scikit-learn for data processing, visualization, and clustering.
2. Load Data and Initial Clustering <a name="load-data-and-initial-clustering"></a>

Here, we load text data from CSV files ("Fake.csv" and "True.csv"), perform text preprocessing, apply TF-IDF transformation, and cluster the data using K-Means algorithm. We label the data as "fake" or "real" and use TF-IDF features for clustering.
3. Visualizing Clusters <a name="visualizing-clusters"></a>

We visualize the clusters created by K-Means using Principal Component Analysis (PCA) for dimensionality reduction. The scatter plots and 3D scatter plots display the clustering results and the positions of cluster centers.
4. Performance Evaluation <a name="performance-evaluation"></a>

We evaluate the performance of clustering based on accuracy metrics. We calculate accuracy considering both cases where clusters match the labels and where clusters are opposite to the labels.
5. Finding Optimal K <a name="finding-optimal-k"></a>

We explore finding the optimal number of clusters (k) using the Elbow method. We plot the Sum of Squared Distances (SSD) score as a function of the number of clusters and the Silhouette score to help determine the suitable value for k.
6. PCA Followed by K-Means <a name="pca-followed-by-k-means"></a>

We apply PCA to the text data before performing K-Means clustering. We visualize the results of PCA followed by K-Means and analyze whether it affects the clustering outcome.
7. MNIST Clustering <a name="mnist-clustering"></a>

In this section, we switch to image data using the MNIST dataset. We load, preprocess, and standardize the data for K-Means clustering. We evaluate the clustering performance by calculating accuracy and label majorities for each cluster.
