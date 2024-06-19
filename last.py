import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
image_with_lighting = cv2.imread('IMG_0352.jpeg', cv2.IMREAD_GRAYSCALE)
image_without_lighting = cv2.imread('IMG_0353.jpeg', cv2.IMREAD_GRAYSCALE)

# Feature extraction (e.g., edges)
edges = cv2.Canny(image_without_lighting, 100, 200)

# Compute importance weights
edges_sum = np.sum(edges)
if edges_sum == 0:
    weights = np.zeros_like(edges, dtype=float)
else:
    weights = edges / edges_sum

# Ensure no NaN values in weights
weights = np.nan_to_num(weights)

# Normalize weights to ensure they sum to 1
weights_sum = np.sum(weights)
if weights_sum != 0:
    weights /= weights_sum
else:
    weights = np.ones_like(weights) / weights.size

# Importance sampling
num_samples = 1000
sampled_points = np.random.choice(np.arange(weights.size), size=num_samples, p=weights.flatten())

# Convert flat indices to 2D coordinates
sampled_coords = np.unravel_index(sampled_points, shape=weights.shape)

# Create importance map visualization
importance_map = np.zeros_like(image_with_lighting)
importance_map[sampled_coords] = 255  # Mark sampled points as white

# Save images
cv2.imwrite('luckw.jpeg', image_with_lighting)
cv2.imwrite('lucko.jpeg', importance_map)


# Display images for verification
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_with_lighting, cmap='gray')
plt.title('Image with Lighting')

plt.subplot(1, 2, 2)
plt.imshow(importance_map, cmap='gray')
plt.title('Importance Map')

plt.show()
