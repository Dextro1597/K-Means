import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('/home/dextro/Desktop/ML_FINAL/Codes/ML/Kmeans/DATAset.csv')
X=dataset.iloc[:,:].values
centers = np.array([[0.1,0.6],[0.3,0.2]])
print('Initial Centroids:\n', centers)
from sklearn.cluster import KMeans
model=KMeans(n_clusters=2, init=centers,n_init=1)
model.fit(X)
print('Labels', model.labels_)

#first question ans
print('P6 belongs to cluter', model.labels_[5])

#second question ans
print('Number of population around cluster 2:', np.count_nonzero(model.labels_== 1))
 
#third question ans
print('New Centroids:\n', model.cluster_centers_)

print('New vs old Centroids:\n', model.cluster_centers_ , "\n", centers)
