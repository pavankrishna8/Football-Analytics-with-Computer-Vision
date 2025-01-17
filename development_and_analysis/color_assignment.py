import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

image_path = "../output_videos/cropped_image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

#Take the top half of the image
top_half_image = image[0:int(image.shape[0]/2), :]
plt.imshow(top_half_image)
plt.show()

#Cluster the image into two clusters
 #Reshape the image into 2d array
image_2d = top_half_image.reshape(-1,3)

 #Performing K-Means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(image_2d)

 #Get the cluster labels
labels = kmeans.labels_

 #reshape the labels into the original image shape
clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

 #Display the clustered image
plt.imshow(clustered_image)
plt.show()

#Clusters are represented in either 0 or 1, since there are only 2 clusters
corner_clusters = [clustered_image[0,0],clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
print(non_player_cluster) #prints the cluster number assigned to non-player

#player cluster
player_cluster = 1-non_player_cluster
print(player_cluster) #prints the cluster number assigned to player

print(kmeans.cluster_centers_[player_cluster]) #prints the cluster color as RGB values

