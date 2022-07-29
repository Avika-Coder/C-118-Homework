from google.colab import files
data_to_load=files.upload()

import pandas as pd
import csv
import plotly.express as px
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
df=pd.read_csv("stars.csv")
print(df.head())

graph= px.scatter(df,x="Size",y="Light")
graph.show()

from sklearn.cluster import KMeans
X=df.iloc[:,[0,1]].values
print(X)

WCSS=[]
for i in range(1,11):
  kMeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
  kMeans.fit(X)
  WCSS.append(kMeans.inertia_)
  
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), WCSS, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1')
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2')
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3')
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', label = 'Centroids',s=100,marker=',')
plt.grid(False)
plt.title('Clusters of Flowers')
plt.xlabel('Petal Size')
plt.ylabel('Sepal Size')
plt.legend()
plt.show()
