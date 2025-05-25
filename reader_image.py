import cv2
import os

image_path = os.path.abspath("input_image/sgugit.jpg")
print("Trying to read image from:", image_path)

image = cv2.imread(image_path)

if image is None:
    print("Image not found or failed to load.")
else:
    cv2.imshow("SSGUiT", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
