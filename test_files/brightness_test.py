from PIL import Image, ImageEnhance
import cv2
import numpy as np

def is_dark_red(pil_image):
    # Load the image
    # image = cv2.imread(image_path)
    
    # Convert the image to the HSV color space
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Convert the PIL image to OpenCV format
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for dark red color in HSV
    lower_dark_red = np.array([0, 100, 20])
    upper_dark_red = np.array([10, 255, 100])
    lower_dark_red2 = np.array([170, 100, 20])
    upper_dark_red2 = np.array([180, 255, 100])
    
    # Create masks for dark red color
    mask1 = cv2.inRange(hsv_image, lower_dark_red, upper_dark_red)
    mask2 = cv2.inRange(hsv_image, lower_dark_red2, upper_dark_red2)
    
    # Combine masks
    mask = mask1 + mask2
    
    # Calculate the percentage of dark red pixels
    dark_red_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    percentage_dark_red = (dark_red_pixels / total_pixels) * 100
    
    # Threshold for considering the image as dark red
    threshold = 35  # For example, 50% of the image should be dark red
    print(percentage_dark_red, percentage_dark_red > threshold)
    if percentage_dark_red > threshold:
        return True
    else:
        return False


# Load the uploaded image
image_path = "20221012-00388_Color_cutting0.png"
image = Image.open(image_path)

while not is_dark_red(image):
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    factor = 0.975  # Decrease brightness by 10%
    image = enhancer.enhance(factor)

# Save the new image
darkened_image_path = "darkened_image2.png"
# image.save(darkened_image_path)
# image.show()
image = np.array(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite(darkened_image_path, image)