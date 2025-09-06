import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

def grabcut_segmentation(image, rect, iterations=5):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = np.zeros(img_bgr.shape[:2], np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')    
    kernel = np.ones((3, 3), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    
    foreground = image * mask2[:, :, np.newaxis]
    background_mask = 1 - mask2
    background = image * background_mask[:, :, np.newaxis]
    return mask2, foreground, background, rect

def create_enhanced_image(original, mask, blur_strength=15):
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    blurred = cv2.GaussianBlur(original, (blur_strength, blur_strength), 0)
    enhanced = np.zeros_like(original)
    enhanced[mask == 1] = original[mask == 1]
    enhanced[mask == 0] = blurred[mask == 0]

    blurred_bg = np.zeros_like(original)
    blurred_bg[mask == 0] = blurred[mask == 0]
    return enhanced, blurred_bg

def interactive_grabcut(image, rect):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 3, cv2.GC_EVAL)
    
    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
    return final_mask

image_path = 'assets/daisy.jpg'
flower_img = cv2.imread(image_path)
flower_img = cv2.cvtColor(flower_img, cv2.COLOR_BGR2RGB)
h, w = flower_img.shape[:2]
initial_rect = (w//4, h//4, w//2, h//2)

mask, foreground, background, rect_used = grabcut_segmentation(flower_img, initial_rect, iterations=8)
refined_mask = interactive_grabcut(flower_img, rect_used)
refined_foreground = flower_img * refined_mask[:, :, np.newaxis]
refined_background = flower_img * (1 - refined_mask)[:, :, np.newaxis]
enhanced_img, blurred_bg_only = create_enhanced_image(flower_img, refined_mask, blur_strength=21)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes[0, 0].imshow(flower_img)
axes[0, 0].set_title('Original image')
axes[0, 0].axis('off')

rect_patch = Rectangle((rect_used[0], rect_used[1]), rect_used[2], rect_used[3], 
                      linewidth=2, edgecolor='red', facecolor='none')
axes[0, 0].add_patch(rect_patch)
axes[0, 1].imshow(mask, cmap='gray')
axes[0, 1].set_title('Initial grabCut mask')
axes[0, 1].axis('off')
axes[0, 2].imshow(refined_mask, cmap='gray')
axes[0, 2].set_title('Refined segmentation mask')
axes[0, 2].axis('off')
axes[0, 3].imshow(refined_foreground)
axes[0, 3].set_title('Extracted foreground')
axes[0, 3].axis('off')

axes[1, 0].imshow(refined_background)
axes[1, 0].set_title('Extracted background')
axes[1, 0].axis('off')
axes[1, 1].imshow(flower_img)
axes[1, 1].set_title('Original image')
axes[1, 1].axis('off')
axes[1, 2].imshow(enhanced_img)
axes[1, 2].set_title('Enhanced image blurred background')
axes[1, 2].axis('off')
axes[1, 3].imshow(blurred_bg_only)
axes[1, 3].set_title('Blurred background only')
axes[1, 3].axis('off')

edge_mask = cv2.Canny((refined_mask * 255).astype(np.uint8), 50, 150)
edge_region = np.zeros_like(flower_img)
edge_pixels = np.where(edge_mask > 0)

dilated_edge = cv2.dilate(edge_mask, np.ones((15, 15), np.uint8), iterations=1)
edge_analysis_region = np.where(dilated_edge > 0)

axes[2, 0].imshow(edge_mask, cmap='gray')
axes[2, 0].set_title('Edge detection boundary')
axes[2, 0].axis('off')
axes[2, 1].imshow(flower_img[100:300, 100:300])
axes[2, 1].set_title('Original cropped')
axes[2, 1].axis('off')
axes[2, 2].imshow(enhanced_img[100:300, 100:300])
axes[2, 2].set_title('Enhanced cropped')
axes[2, 2].axis('off')

plt.tight_layout()
plt.show()