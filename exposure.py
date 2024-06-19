import cv2
import numpy as np

# Load the original image and the reflectance map
original_img = cv2.imread('wilight.jpeg')
reflectance_map = cv2.imread('reflectance_map.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure the reflectance map has the same dimensions as the original image
reflectance_map = cv2.resize(reflectance_map, (original_img.shape[1], original_img.shape[0]))

# Convert the grayscale reflectance map to a 3-channel image
reflectance_map_3ch = cv2.cvtColor(reflectance_map, cv2.COLOR_GRAY2BGR)

# Convert images to float type for multiplication
original_img_float = original_img.astype(float)
reflectance_map_float = reflectance_map_3ch.astype(float) / 255

# Enhance the original image by multiplying with the reflectance map
enhanced_img = cv2.multiply(original_img_float, reflectance_map_float)

# Convert back to 8-bit for displaying
enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)

# Display the enhanced image
cv2.imshow('Enhanced Image', enhanced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
