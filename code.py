##### 1. Imports
from google.colab import files
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import matplotlib
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as PCA
from sklearn.metrics import silhouette_score as sil_func
import matplotlib.style as style

##### 2. Load data and cluster the samples into "fake" and "real"
fake = pd.read_csv('Fake.csv')
fake["label"] = 0
real = pd.read_csv('True.csv')
real["label"] = 1
all = fake.append(real)
all = all[["text", "label"]]
all.head()

# Applying tfidf on the text column of the dataset.
tfdata = tfidf(max_features = 1000, stop_words="english")
data = tfdata.fit_transform(all["text"])

# Creating scaler for standartization
scale = Scaler(with_mean=False)
# Fitting the scaler to our data
scale.fit(data)
# Standardizing our data
scaled_data = scale.transform(data)
print(scaled_data.shape)

# Creating KMeans with 2 clusters
kmeans = KMeans(2)
# Fitting the KMeans clusters to our data
k_mean_data = kmeans.fit_transform(scaled_data)
# Predicting the cluster for each point using our calculated centroids
prediction = kmeans.predict(scaled_data)

# extract the location of the cluster centers
centers = kmeans.cluster_centers_

k_mean_data = pd.DataFrame(k_mean_data)
# Getting feature names and creating DataFrame
col_name = list(tfdata.get_feature_names_out())
tf_idf = pd.DataFrame.sparse.from_spmatrix(scaled_data, columns = col_name)

##### 3. Visualize using scatter plot

# Create 3 component PCA
pca = PCA(n_components=3)
# Fit PCA to data and transform data accordingly
pca_tf_idf = pca.fit_transform(scaled_data.todense())
# Transform the centers as well
pca_centers = pca.transform(centers)
pca_centers

# pca_tf_idf = pd,DtaFrame(pca_tf_idf)
# Add first 2 PCA columns of data points to scatter
plt.scatter(x=pca_tf_idf[:,0], y=pca_tf_idf[:,1], c=all["label"], alpha=0.1)
plt.title("Document's tf idf  and k means centers mapped to PCA feature space")
plt.xlabel("PCA feature 1")
plt.ylabel("PCA feature 2")
# Add first 2 PCA columns of centroids to scatter
plt.scatter(x=pca_centers[:,0], y=pca_centers[:,1],  s=300, c = "r",edgecolors = "black", marker='*'   )
plt.legend()
plt.show()

# Create figure and ax for 3d scatter
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

# Add first 3 PCA columns of data points to scatter, make smaller and use alpha so centroids can be seen
ax.scatter(xs=pca_tf_idf[:,0], ys=pca_tf_idf[:,1], zs = pca_tf_idf[:,2], c=all["label"],  s=10, alpha = 0.03 )
# Add first 3 PCA columns of centroids to scatter, make larger to see through the data points
ax.scatter(xs=pca_centers[:,0], ys=pca_centers[:,1], zs = pca_centers[:,2],  s=300, c = "r",edgecolors = "black", marker='*', alpha=1  )
ax.legend()
plt.show()

##### 4. Performance Evaluation

# In lables 0 is fake and 1 is real
# Calculate accuracy If cluster is labeled the same as labels
print("If clusters are the same as labels:")
all["prediction"] = prediction
tp_tn = len(all[all["prediction"] == all["label"]])
accuracy1 = tp_tn/len(all)
print(accuracy1)
print()

# Calculate accuracy If cluster is labeled opposite of labels
tp_tn = len(all[all["prediction"] != all["label"]])
accuracy2 = tp_tn/len(all)
print("If clusters are opposite to labels:")
print(accuracy2)

##### 5. Find optimal k

ssd_score = []
sil_score = []
for num in range(2, 20):
  kmeans = KMeans(num)
  kmeans.fit_transform(scaled_data)
  # calculate the ssd score for this kmeans iterations
  ssd_score.append(kmeans.inertia_ )
  # calculate the sillhoutte score for this kmeans iterations
  sil_score.append(sil_func(scaled_data, kmeans.labels_ ,sample_size = 300))

style.use("fast")
plt.title("SSD score as a function of the number of clusters")
plt.plot(range(2, 20), ssd_score)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("ssd score")
plt.show()

style.use("fast")
plt.title("Silhouette score as a function of the number of clusters")
plt.plot(range(2, 20), sil_score)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette score")
plt.show()

#The optimal k for clustering according to our finding is 2. 
#The reason being that the silhouette score is the highest for k=2, 
#additionally we couldn't use the ssd distances graph to choose a better k since it has no "elbow".

##### 6.PCA then k-means

# Applying tfidf on the text column of the dataset.
tfdata = tfidf(max_features = 1000, stop_words="english")
data = tfdata.fit_transform(all["text"])

# Creating scaler for standartization
scale = Scaler(with_mean=False)
# Fitting the scaler to our data
scale.fit(data)
# Standardizing our data
scaled_data = scale.transform(data)
print(scaled_data.shape)

# Create 1000 component PCA
pca = PCA(n_components=1000)
# Fit PCA to data and transform data accordingly
pca_tf_idf = pca.fit_transform(scaled_data.todense())
# Transform the centers as well
# pca_centers = pca.transform(centers)
# pca_centers

# Creating KMeans with 2 clusters
kmeans = KMeans(2)
# Fitting the KMeans clusters to our data
k_mean_data = kmeans.fit_transform(pca_tf_idf)
# Predicting the cluster for each point using our calculated centroids
prediction = kmeans.predict(pca_tf_idf)

# pca_tf_idf = pd,DtaFrame(pca_tf_idf)
# Add first 2 PCA columns of data points to scatter
plt.scatter(x=pca_tf_idf[:,0], y=pca_tf_idf[:,1], c=all["label"]  )
# Add first 2 PCA columns of centroids to scatter
plt.scatter(x=kmeans.cluster_centers_[:,0], y=kmeans.cluster_centers_[:,1],  s=180, c = "r",edgecolors = "black", marker='*'   )
plt.show()

# Create figure and ax for 3d scatter
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

# Add first 3 PCA columns of data points to scatter, make smaller and use alpha so centroids can be seen
ax.scatter(xs=pca_tf_idf[:,0], ys=pca_tf_idf[:,1], zs = pca_tf_idf[:,2], c=all["label"],  s=10, alpha = 0.05 )
# Add first 3 PCA columns of centroids to scatter, make larger to see through the data points
ax.scatter(xs=kmeans.cluster_centers_[:,0], ys=kmeans.cluster_centers_[:,1], zs =kmeans.cluster_centers_[:,2],  s=300, c = "r",edgecolors = "black", marker='*', alpha=1  )
plt.show()

# In lables 0 is fake and 1 is real
# Calculate accuracy If cluster is labeled the same as labels
print("If clusters are the same as labels:")
all["prediction"] = kmeans.labels_
tp_tn = len(all[all["prediction"] == all["label"]])
accuracy1 = tp_tn/len(all)
print(accuracy1)
print()

# Calculate accuracy If cluster is labeled opposite of labels
tp_tn = len(all[all["prediction"] != all["label"]])
accuracy2 = tp_tn/len(all)
print("If clusters are opposite to labels:")
print(accuracy2)

'''
The accuracy in this approach is very similar to the accuracy in the previous approach. 
This is because k maens is an algorithms that clusters examples based on their distance to the closest 
cluster such that the average distances between each cluster to its assigned data is the lowest possible. 
When we modified the axes using pca features, it seems that the data points that were "similar" before the 
pca remained close to each other and therefore were assigned to the same cluster as before. 
'''

##### 7. MNIST clustering

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
mnist_data, labels = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
mnist_data = pd.DataFrame(mnist_data.reshape(70000, 28*28))
mnist_data

# Creating scaler for standartization
scale = Scaler(with_mean=False)
# Fitting the scaler to our data
scale.fit(mnist_data)
# Standardizing our data
scaled_data = scale.transform(mnist_data)
print(scaled_data.shape)

# Creating KMeans with 2 clusters
kmeans = KMeans(10)
# Fitting the KMeans clusters to our data
k_mean_data = kmeans.fit_transform(scaled_data)
# Predicting the cluster for each point using our calculated centroids
prediction = kmeans.predict(scaled_data)

# extract the location of the cluster centers
centers = kmeans.cluster_centers_

k_mean_data = pd.DataFrame(k_mean_data)
k_mean_data["pred"] = prediction
k_mean_data["label"] = labels

preds_majority = []
accuracys = []
for label in range(0, 10):
  cluster_data = k_mean_data[k_mean_data["pred"] == label]
  label_counts = cluster_data["label"].value_counts()
  # the majority label for the cluster
  preds_majority.append(label_counts.keys()[0])
  # calc accuracy
  accuracys.append(label_counts.max()/len(cluster_data))

for i in range(10):
  print(f"label for cluster {i} by majority voting: ", preds_majority[i])
  print(f"accuracy score for cluster {i} is {accuracys[i]}")
  print("")


ssd_score = []
sil_score = []
for num in range(2, 20):
  kmeans = KMeans(num)
  kmeans.fit_transform(scaled_data)
  ssd_score.append(kmeans.inertia_ )
  sil_score.append(sil_func(scaled_data, kmeans.labels_ ,sample_size = 300))

style.use("fast")
plt.title("SSD score as a function of the number of clusters")
plt.plot(range(2, 20), ssd_score)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("ssd score")
plt.show()

style.use("fast")
plt.title("Silhouette score as a function of the number of clusters")
plt.plot(range(2, 20), sil_score)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette score")
plt.show()

##Now I will use PCA before K means and plot again

# Create 3 component PCA
pca = PCA(n_components=2)
# Fit PCA to data and transform data accordingly
scaled_data = pca.fit_transform(scaled_data)
# Transform the centers as well

# Creating KMeans with 2 clusters
kmeans = KMeans(10)
# Fitting the KMeans clusters to our data
k_mean_data = kmeans.fit_transform(scaled_data)
# Predicting the cluster for each point using our calculated centroids
prediction = kmeans.predict(scaled_data)

k_mean_data = pd.DataFrame(k_mean_data)
k_mean_data["pred"] = prediction
k_mean_data["label"] = labels

preds_majority = []
accuracys = []
for label in range(0, 10):
  cluster_data = k_mean_data[k_mean_data["pred"] == label]
  label_counts = cluster_data["label"].value_counts()
  # the majority label for the cluster
  preds_majority.append(label_counts.keys()[0])
  # calc accuracy
  accuracys.append(label_counts.max()/len(cluster_data))

for i in range(10):
  print(f"label for cluster {i} by majority voting: ", preds_majority[i])
  print(f"accuracy score for cluster {i} is {accuracys[i]}")
  print("")


ssd_score = []
sil_score = []
for num in range(2, 20):
  kmeans = KMeans(num)
  kmeans.fit_transform(scaled_data)
  ssd_score.append(kmeans.inertia_ )
  sil_score.append(sil_func(scaled_data, kmeans.labels_ ,sample_size = 300))

style.use("fast")
plt.title("SSD score as a function of the number of clusters")
plt.plot(range(2, 20), ssd_score)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("ssd score")
plt.show()

style.use("fast")
plt.title("Silhouette score as a function of the number of clusters")
plt.plot(range(2, 20), sil_score)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette score")
plt.show()

