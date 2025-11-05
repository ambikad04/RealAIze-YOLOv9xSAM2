import cv2
import numpy as np

# Create a blank image
img = np.zeros((640, 640, 3), dtype=np.uint8)

# Draw some shapes
# Rectangle
cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)  # Green rectangle

# Circle
cv2.circle(img, (450, 200), 100, (0, 0, 255), -1)  # Red circle

# Triangle
pts = np.array([[400, 400], [500, 500], [300, 500]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(img, [pts], (255, 0, 0))  # Blue triangle

# Save the image
cv2.imwrite('test_image.jpg', img)
print("Test image created: test_image.jpg") 