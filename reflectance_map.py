import cv2
import numpy as np

# Load images
img_with_light = cv2.imread('handw.jpeg')
img_without_light = cv2.imread('hando.jpeg')

# Convert to grayscale
gray_light = cv2.cvtColor(img_with_light, cv2.COLOR_BGR2GRAY)
gray_dark = cv2.cvtColor(img_without_light, cv2.COLOR_BGR2GRAY)

# Avoid division by zero
gray_dark = np.where(gray_dark == 0, 1, gray_dark)

# Reflectance map estimation
reflectance_map = np.divide(gray_light.astype(float), gray_dark.astype(float))

# Normalize for display
reflectance_map_normalized = cv2.normalize(reflectance_map, None, 0, 255, cv2.NORM_MINMAX)

# Save or display the result
cv2.imwrite('hand.jpg', reflectance_map_normalized)

cv2.waitKey(0)
cv2.destroyAllWindows()
