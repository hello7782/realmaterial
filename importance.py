import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
image_with_lighting = cv2.imread('sofaw.jpeg', cv2.IMREAD_GRAYSCALE)
image_without_lighting = cv2.imread('sofao.jpeg', cv2.IMREAD_GRAYSCALE)

# Feature extraction (e.g., edges)
edges = cv2.Canny(image_without_lighting, 100, 200)

# Compute importance weights (normalize edge values)
weights = edges / np.sum(edges)

# Importance sampling
num_samples = 1000
sampled_points = np.random.choice(np.arange(weights.size), size=num_samples, p=weights.flatten())

# Convert flat indices to 2D coordinates
sampled_coords = np.unravel_index(sampled_points, shape=weights.shape)

# Visualize sampled points
plt.figure(figsize=(10, 10))
plt.imshow(image_with_lighting, cmap='gray')
plt.scatter(sampled_coords[1], sampled_coords[0], color='red', s=1)
plt.title('Importance Sampled Points')
plt.show()
