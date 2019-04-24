#Code by Ahad Patel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('/home/dextro/Desktop/ML_FINAL/Codes/ML/Kmeans/DATAset.csv')
X=dataset.iloc[:,:].values

#for plotting the points
x1=dataset.iloc[:,:-1].values
y1=dataset.iloc[:,-1].values
plt.scatter(x1,y1, color="red")
plt.title("Cluster Distribution")
plt.show()

centers = np.array([[0.1,0.6],[0.3,0.2]])
print('Initial Centroids:\n', centers)

#for plotting the intial centroids
C_x=np.array([0.1,0.3])
C_y=np.array([0.6,0.2])
plt.scatter(C_x[0],C_y[0], color="green")
plt.scatter(C_x[1],C_y[1], color="green")
plt.title('Centroids Initial')
plt.show()

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
finalcenters =np.array(model.cluster_centers_)
b=finalcenters[0]
d=finalcenters[1]
#print('B:', b , 'D:' ,d) 

print('New vs old Centroids:\n', model.cluster_centers_ ,'\n', centers)

#final graph with new centroids
plt.scatter(x1,y1, color="Black")
plt.scatter(b[0],b[1], color="Red")
plt.scatter(d[0],d[1], color="Red")
plt.title('Final Graph with new Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
