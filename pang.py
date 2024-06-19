import cv2
import numpy as np

# Load images
image_with_light = cv2.imread('with_light.jpg', cv2.IMREAD_GRAYSCALE)
image_without_light = cv2.imread('without_light.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the difference map
difference_map = cv2.absdiff(image_with_light, image_without_light)

# Normalize the difference map for better visualization and processing
difference_map_normalized = cv2.normalize(difference_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

# Compute importance weights from the normalized difference map
weights = difference_map_normalized.flatten().astype(np.float32)
weights /= np.sum(weights)  # Normalize weights to sum to 1

# Sample indices based on these importance weights
num_samples = 1000  # Number of points to sample
sampled_indices = np.random.choice(weights.size, size=num_samples, p=weights)

# Convert flat indices to 2D coordinates
sampled_coords = np.unravel_index(sampled_indices, shape=image_with_light.shape)

# Compute importance weights from the normalized difference map
weights = difference_map_normalized.flatten().astype(np.float32)
weights /= np.sum(weights)  # Normalize weights to sum to 1

# Sample indices based on these importance weights
num_samples = 1000  # Number of points to sample
sampled_indices = np.random.choice(weights.size, size=num_samples, p=weights)

# Convert flat indices to 2D coordinates
sampled_coords = np.unravel_index(sampled_indices, shape=image_with_light.shape)

# Save the segmentation result
cv2.imwrite('segmentation_output.jpg', segmentation_mask)
