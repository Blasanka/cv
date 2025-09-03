import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def setup_transformation():
    transform_params = np.zeros(256, dtype=np.uint8)
    
    x1, y1 = 0, 0
    x2, y2 = 50, 100
    for k in range(x1, x2 + 1):
        transform_params[k] = int(y1 + (y2 - y1) * (k - x1) / (x2 - x1))
    
    x1, y1 = 50, 100
    x2, y2 = 150, 150
    for k in range(x1, x2 + 1):
        transform_params[k] = int(y1 + (y2 - y1) * (k - x1) / (x2 - x1))
    
    x1, y1 = 150, 150
    x2, y2 = 150, 255
    transform_params[150] = 255
    
    for k in range(151, 256):
        transform_params[k] = 255
    
    return transform_params

def apply_transformation(image_path, params):
    img = cv2.imread(image_path)
    if img is None:
        # I used PIL to open image
        img = np.array(Image.open(image_path))
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
    
    transformed_img = cv2.LUT(gray_img, params)
    
    return gray_img, transformed_img


if __name__ == "__main__":
    intensity_transformation = setup_transformation()
    
    img_path = 'assets/emma.jpg'
    
    try:
        original, transformed = apply_transformation(img_path, intensity_transformation)
        
        # displaying the transformed and original image
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title('Original Emma')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(transformed, cmap='gray')
        plt.title('Transformed Emma')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        cv2.imwrite('assets/transformed_emma.jpg', transformed)
        
    except Exception as e:
        print(f"Error: {e}")