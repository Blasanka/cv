import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('assets/einstein.png', cv2.IMREAD_GRAYSCALE)

sobel_x = np.array([[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]], dtype=np.float32)

sobel_x_cv2 = cv2.filter2D(img, cv2.CV_32F, sobel_x)
sobel_y_cv2 = cv2.filter2D(img, cv2.CV_32F, sobel_y)

gradient_magnitude_cv2 = np.sqrt(sobel_x_cv2**2 + sobel_y_cv2**2)

def custom_filter2d(image, kernel):
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(img_h):
        for j in range(img_w):
            roi = padded_img[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(roi * kernel)

    return output

sobel_x_custom = custom_filter2d(img.astype(np.float32), sobel_x)
sobel_y_custom = custom_filter2d(img.astype(np.float32), sobel_y)

gradient_magnitude_custom = np.sqrt(sobel_x_custom**2 + sobel_y_custom**2)

def separable_sobel_x(image):
    vertical_kernel = np.array([[1], [2], [1]], dtype=np.float32)
    temp = custom_filter2d(image, vertical_kernel)
    horizontal_kernel = np.array([[1, 0, -1]], dtype=np.float32)
    result = custom_filter2d(temp, horizontal_kernel)
    return result

def separable_sobel_y(image):
    vertical_kernel = np.array([[1], [0], [-1]], dtype=np.float32)
    temp = custom_filter2d(image, vertical_kernel)
    horizontal_kernel = np.array([[1, 2, 1]], dtype=np.float32)
    result = custom_filter2d(temp, horizontal_kernel)
    return result

sobel_x_separable = separable_sobel_x(img.astype(np.float32))
sobel_y_separable = separable_sobel_y(img.astype(np.float32))

gradient_magnitude_separable = np.sqrt(sobel_x_separable**2 + sobel_y_separable**2)
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')
axes[0, 1].imshow(sobel_x_cv2, cmap='gray')
axes[0, 1].set_title('Sobel X filter2D')
axes[0, 1].axis('off')
axes[0, 2].imshow(sobel_y_cv2, cmap='gray')
axes[0, 2].set_title('Sobel Y filter2D')
axes[0, 2].axis('off')
axes[0, 3].imshow(gradient_magnitude_cv2, cmap='gray')
axes[0, 3].set_title('Gradient Magnitude filter2D')
axes[0, 3].axis('off')

axes[1, 0].imshow(img, cmap='gray')
axes[1, 0].set_title('Original Image')
axes[1, 0].axis('off')
axes[1, 1].imshow(sobel_x_custom, cmap='gray')
axes[1, 1].set_title('Sobel X Custom')
axes[1, 1].axis('off')
axes[1, 2].imshow(sobel_y_custom, cmap='gray')
axes[1, 2].set_title('Sobel Y Custom')
axes[1, 2].axis('off')
axes[1, 3].imshow(gradient_magnitude_custom, cmap='gray')
axes[1, 3].set_title('Gradient Magnitude Custom')
axes[1, 3].axis('off')

axes[2, 0].imshow(img, cmap='gray')
axes[2, 0].set_title('Original Image')
axes[2, 0].axis('off')
axes[2, 1].imshow(sobel_x_separable, cmap='gray')
axes[2, 1].set_title('Sobel X Separable')
axes[2, 1].axis('off')
axes[2, 2].imshow(sobel_y_separable, cmap='gray')
axes[2, 2].set_title('Sobel Y Separable')
axes[2, 2].axis('off')
axes[2, 3].imshow(gradient_magnitude_separable, cmap='gray')
axes[2, 3].set_title('Gradient Magnitude Separable')
axes[2, 3].axis('off')

plt.tight_layout()
plt.show()