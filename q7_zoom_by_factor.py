import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def zoom_nearest_neighbor(image, zoom_factor):
    if len(image.shape) == 2:
        h, w = image.shape
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        zoomed = np.zeros((new_h, new_w), dtype=image.dtype)
        
        for i in range(new_h):
            for j in range(new_w):
                orig_i = int(i / zoom_factor)
                orig_j = int(j / zoom_factor)
                orig_i = min(orig_i, h - 1)
                orig_j = min(orig_j, w - 1)
                zoomed[i, j] = image[orig_i, orig_j]      
    else:
        h, w, c = image.shape
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        zoomed = np.zeros((new_h, new_w, c), dtype=image.dtype)
        
        for i in range(new_h):
            for j in range(new_w):
                orig_i = int(i / zoom_factor)
                orig_j = int(j / zoom_factor)
                orig_i = min(orig_i, h - 1)
                orig_j = min(orig_j, w - 1)
                zoomed[i, j] = image[orig_i, orig_j]
    
    return zoomed

def zoom_bilinear(image, zoom_factor):
    if len(image.shape) == 2:
        h, w = image.shape
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        zoomed = np.zeros((new_h, new_w), dtype=np.float32)
        
        for i in range(new_h):
            for j in range(new_w):
                orig_i = i / zoom_factor
                orig_j = j / zoom_factor
                i1, j1 = int(orig_i), int(orig_j)
                i2, j2 = min(i1 + 1, h - 1), min(j1 + 1, w - 1)
                di, dj = orig_i - i1, orig_j - j1
                top_left = image[i1, j1]
                top_right = image[i1, j2]
                bottom_left = image[i2, j1]
                bottom_right = image[i2, j2]
                top = top_left * (1 - dj) + top_right * dj
                bottom = bottom_left * (1 - dj) + bottom_right * dj
                zoomed[i, j] = top * (1 - di) + bottom * di       
    else:
        h, w, c = image.shape
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        zoomed = np.zeros((new_h, new_w, c), dtype=np.float32)
        
        for channel in range(c):
            for i in range(new_h):
                for j in range(new_w):
                    orig_i = i / zoom_factor
                    orig_j = j / zoom_factor
                    i1, j1 = int(orig_i), int(orig_j)
                    i2, j2 = min(i1 + 1, h - 1), min(j1 + 1, w - 1)
                    di, dj = orig_i - i1, orig_j - j1
                    top_left = image[i1, j1, channel]
                    top_right = image[i1, j2, channel]
                    bottom_left = image[i2, j1, channel]
                    bottom_right = image[i2, j2, channel]
                    top = top_left * (1 - dj) + top_right * dj
                    bottom = bottom_left * (1 - dj) + bottom_right * dj
                    zoomed[i, j, channel] = top * (1 - di) + bottom * di
    
    return zoomed.astype(image.dtype)

def compute_ssd(img1, img2):
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        if len(img1.shape) == 3:
            img1 = img1[:min_h, :min_w, :]
            img2 = img2[:min_h, :min_w, :]
        else:
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]

    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    ssd = np.sum((img1_f - img2_f) ** 2)
    if len(img1.shape) == 3:
        normalized_ssd = ssd / (img1.shape[0] * img1.shape[1] * img1.shape[2] * 255**2)
    else:
        normalized_ssd = ssd / (img1.shape[0] * img1.shape[1] * 255**2)
    return normalized_ssd

large_original1 = cv2.imread('assets/im01.png')
small_image1 = cv2.imread('assets/im01small.png')

large_original2 = cv2.imread('assets/im02.png')
small_image2 = cv2.imread('assets/im02small.png')

zoom_factor = 4.0

zoomed_nn1 = zoom_nearest_neighbor(small_image1, zoom_factor)
zoomed_nn2 = zoom_nearest_neighbor(small_image2, zoom_factor)

zoomed_bilinear1 = zoom_bilinear(small_image1, zoom_factor)
zoomed_bilinear2 = zoom_bilinear(small_image2, zoom_factor)

ssd_nn1 = compute_ssd(large_original1, zoomed_nn1)
ssd_nn2 = compute_ssd(large_original2, zoomed_nn2)

ssd_bilinear1 = compute_ssd(large_original1, zoomed_bilinear1)
ssd_bilinear2 = compute_ssd(large_original2, zoomed_bilinear2)

zoomed_cv2_nn1 = cv2.resize(small_image1, (small_image1.shape[1]*4, small_image1.shape[0]*4), 
                            interpolation=cv2.INTER_NEAREST)
zoomed_cv2_bilinear1 = cv2.resize(small_image1, (small_image1.shape[1]*4, small_image1.shape[0]*4), 
                                  interpolation=cv2.INTER_LINEAR)

ssd_cv2_nn1 = compute_ssd(large_original1, zoomed_cv2_nn1)
ssd_cv2_bilinear1 = compute_ssd(large_original1, zoomed_cv2_bilinear1)

fig, axes = plt.subplots(4, 4, figsize=(20, 20))

axes[0, 0].imshow(cv2.cvtColor(large_original1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original 1 (400x400)')
axes[0, 0].axis('off')
axes[0, 1].imshow(cv2.cvtColor(small_image1, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f'Small Image 1 {small_image1.shape}')
axes[0, 1].axis('off')
axes[0, 2].imshow(cv2.cvtColor(large_original2, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('Original 2 (400x400)')
axes[0, 2].axis('off')
axes[0, 3].imshow(cv2.cvtColor(small_image2, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title(f'Small Image 2 {small_image2.shape}')
axes[0, 3].axis('off')

axes[1, 0].imshow(cv2.cvtColor(zoomed_nn1, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f'Zoomed NN 1 {ssd_nn1:.4f}')
axes[1, 0].axis('off')
axes[1, 1].imshow(np.sum(np.abs(large_original1.astype(float) - zoomed_nn1.astype(float)), axis=2), cmap='hot')
axes[1, 1].set_title('Difference NN 1')
axes[1, 1].axis('off')
axes[1, 2].imshow(cv2.cvtColor(zoomed_nn2, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f'Zoomed NN 2 {ssd_nn2:.4f}')
axes[1, 2].axis('off')
axes[1, 3].imshow(np.sum(np.abs(large_original2.astype(float) - zoomed_nn2.astype(float)), axis=2), cmap='hot')
axes[1, 3].set_title('Difference NN 2')
axes[1, 3].axis('off')

axes[2, 0].imshow(cv2.cvtColor(zoomed_bilinear1, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title(f'Zoomed Bilinear 1 {ssd_bilinear1:.4f}')
axes[2, 0].axis('off')

axes[2, 1].imshow(np.sum(np.abs(large_original1.astype(float) - zoomed_bilinear1.astype(float)), axis=2), cmap='hot')
axes[2, 1].set_title('Difference Bilinear 1')
axes[2, 1].axis('off')
axes[2, 2].imshow(cv2.cvtColor(zoomed_bilinear2, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title(f'Zoomed Bilinear 2 {ssd_bilinear2:.4f}')
axes[2, 2].axis('off')
axes[2, 3].imshow(np.sum(np.abs(large_original2.astype(float) - zoomed_bilinear2.astype(float)), axis=2), cmap='hot')
axes[2, 3].set_title('Difference Bilinear 2')
axes[2, 3].axis('off')

axes[3, 0].imshow(cv2.cvtColor(zoomed_cv2_nn1, cv2.COLOR_BGR2RGB))
axes[3, 0].set_title(f'OpenCV NN 1 {ssd_cv2_nn1:.4f}')
axes[3, 0].axis('off')
axes[3, 1].imshow(cv2.cvtColor(zoomed_cv2_bilinear1, cv2.COLOR_BGR2RGB))
axes[3, 1].set_title(f'OpenCV Bilinear 1 {ssd_cv2_bilinear1:.4f}')
axes[3, 1].axis('off')

plt.tight_layout()
plt.show()