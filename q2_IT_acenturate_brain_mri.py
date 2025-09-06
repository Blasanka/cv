import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def apply_transformation(image_path, matter_params):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(image_path).convert('L'))
    transformed = cv2.LUT(img, matter_params)
    
    return img, transformed

def transform_white_matter():
    matter_params = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < 50:
            matter_params[i] = int(i * 0.3)
        elif i < 120:
            matter_params[i] = np.clip(int(50 + (i - 50) * 0.8), 0, 255)
        else:
            matter_params[i] = np.clip(int(106 + (i - 120) * 1.5), 0, 255)
            if matter_params[i] > 255:
                matter_params[i] = 255
    return matter_params

def transform_gray_matter():
    matter_params = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < 30:
            matter_params[i] = int(i * 0.2)
        elif i < 70:
            matter_params[i] = int(6 + (i - 30) * 1.2)
        elif i < 150:
            matter_params[i] = int(54 + (i - 70) * 2.0)
            if matter_params[i] > 255:
                matter_params[i] = 255
        else:
            matter_params[i] = np.clip(int(214 + (i - 150) * 0.4), 0, 255)
            if matter_params[i] > 255:
                matter_params[i] = 255
    return matter_params

def plot_transformation_curves():
    white_matter = transform_white_matter()

    gray_matter = transform_gray_matter()

    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    input_range = np.arange(256)
    
    axes[0].plot(input_range, white_matter, 'r-', linewidth=2, label='White Matter')
    axes[0].plot([0, 255], [0, 255], 'k--', alpha=0.5, label='Identity')
    axes[0].set_xlabel('Input Intensity')
    axes[0].set_ylabel('Output Intensity')
    axes[0].set_title('White Matter Transformation')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(0, 255)
    axes[0].set_ylim(0, 255)
    axes[1].plot(input_range, gray_matter, 'g-', linewidth=2, label='Gray Matter')
    axes[1].plot([0, 255], [0, 255], 'k--', alpha=0.5, label='Identity')
    axes[1].set_xlabel('Input Intensity')
    axes[1].set_ylabel('Output Intensity')
    axes[1].set_title('Gray Matter Transformation')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(0, 255)
    axes[1].set_ylim(0, 255)

    plt.tight_layout()
    plt.show()
    return white_matter, gray_matter

def display_brain_results(original, white, gray):
    _, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('Original Brain Image')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(white, cmap='gray')
    axes[0,1].set_title('White Matter')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(gray, cmap='gray')
    axes[1,0].set_title('Gray Matter')
    axes[1,0].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = 'assets/brain_proton_density_slice.png'
    white_matter, gray_matter = plot_transformation_curves()
    original, white_matter = apply_transformation(image_path, white_matter)
    _, gray_matter = apply_transformation(image_path, gray_matter)
    
    display_brain_results(original, white_matter, gray_matter)
    
    cv2.imwrite('assets/brain_white_matter_transformed.jpg', white_matter)
    cv2.imwrite('assets/brain_gray_matter_transformed.jpg', gray_matter)