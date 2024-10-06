import numpy as np
import cv2

def is_dark_red(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to the HSV color space
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
    threshold = 40  # For example, 50% of the image should be dark red
    
    if percentage_dark_red > threshold:
        return True, percentage_dark_red
    else:
        return False, percentage_dark_red


# Test the function
image_path = 'new.png'
is_dark_red_image, percentage = is_dark_red(image_path)
print(is_dark_red_image, percentage)
