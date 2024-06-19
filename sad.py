import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load images
image_with_light = cv2.imread('IMG_0352.jpeg', cv2.IMREAD_GRAYSCALE)
image_without_light = cv2.imread('IMG_0353.jpeg', cv2.IMREAD_GRAYSCALE)

# Compute the absolute difference between the two images
difference_map = cv2.absdiff(image_with_light, image_without_light)

# Normalize the difference for visibility
difference_map = cv2.normalize(difference_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Flatten the difference map to use in sampling
flat_difference = difference_map.flatten()

# Convert differences to probabilities by normalizing to sum to 1
probabilities = flat_difference / np.sum(flat_difference)

# Sample points based on these probabilities
num_samples = 1000  # Number of points to sample
sampled_indices = np.random.choice(a=np.arange(probabilities.size), size=num_samples, p=probabilities)

# Convert flat indices to 2D coordinates
sampled_y, sampled_x = np.unravel_index(sampled_indices, shape=difference_map.shape)

plt.figure(figsize=(10, 10))
plt.imshow(image_with_light, cmap='gray')
plt.scatter(sampled_x, sampled_y, color='red', s=1)  # mark the sampled points
plt.title('Sampled Points Based on Lighting Difference')
plt.show()


# Extract intensity values from the original image at sampled points
sampled_intensities = image_with_light[sampled_y, sampled_x].reshape(-1, 1)

# Apply K-means clustering to segment based on sampled intensities
kmeans = KMeans(n_clusters=3, random_state=0).fit(sampled_intensities)
labels = kmeans.labels_

# Create an output image for visualizing the segmentation
segmentation_output = np.zeros(image_with_light.shape, dtype=np.uint8)
for label, y, x in zip(labels, sampled_y, sampled_x):
    segmentation_output[y, x] = (label + 1) * 85  # scale labels to 255 / 3

plt.figure(figsize=(10, 10))
plt.imshow(segmentation_output, cmap='gray')
plt.title('Segmentation Output')
plt.show()

cv2.imwrite('segmentation_result.png', segmentation_output)
