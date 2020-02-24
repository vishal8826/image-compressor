from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = Image.open('me.jpg')
img_np = np.asarray(img)

pixels=img_np.reshape(img_np.shape[0]*img_np.shape[1], img_np.shape[2])

model = KMeans(n_clusters=16)
model.fit(pixels)

pixels_centroids = model.labels_
cluster_centers = model.cluster_centers_

final = np.zeros((pixels_centroids.shape[0],3))

for cluster_no in range(16):
    final[pixels_centroids==cluster_no]=cluster_centers[cluster_no]
    
comp_image = final.reshape(img_np.shape[0],img_np.shape[1],3)

comp_image = Image.fromarray(np.uint8(comp_image))
comp_image.save('me1.jpg')

fname = 'me.jpg'
fname1 = 'me1.jpg'
img1 = mpimg.imread(fname,0)
img2 = mpimg.imread(fname1,0)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,20))
ax1.imshow(img1)
ax2.imshow(img2)
