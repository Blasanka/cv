import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, segmentation
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt

original_image = None
processed_images = {}
        
def load_image(image_path):
    original_image = cv2.imread(image_path)
    if len(original_image.shape) == 3:
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = original_image.copy()
        
    return gray_image

def preprocess_gaussian_noise(image):
    kernel_size = 5
    sigma = 1.0
    
    denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    processed_images['gaussian_denoised'] = denoised
    return denoised

def preprocess_salt_pepper_noise(image):
    kernel_size = 5
    denoised = cv2.medianBlur(image, kernel_size)

    processed_images['saltpepper_denoised'] = denoised
    return denoised

def apply_otsu_segmentation(image):
    threshold_value, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images['otsu_segmented'] = binary_image
    return binary_image, threshold_value

def apply_morphological_operations(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_bool = closed > 0
    
    min_area = 100
    cleaned = morphology.remove_small_objects(binary_bool, min_size=min_area)
    filled = morphology.remove_small_holes(cleaned, area_threshold=50)

    morphological_result = (filled * 255).astype(np.uint8)
    processed_images['morphological'] = morphological_result
    return morphological_result

def count_connected_parts(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    num_grains = num_labels - 1
    min_area = 80
    max_area = 2000
    
    valid_grains = 0
    valid_labels = []
    grain_areas = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            valid_grains += 1
            valid_labels.append(i)
            grain_areas.append(area)
    
    colored_labels = np.zeros_like(binary_image)
    for label in valid_labels:
        colored_labels[labels == label] = 255
    
    processed_images['connected_parts'] = colored_labels
    processed_images['all_labels'] = labels
    return valid_grains, labels, stats, centroids, valid_labels

def watershed_segmentation(binary_image):
    dist_transform = distance_transform_edt(binary_image)
    local_max = peak_local_max(
        dist_transform, 
        min_distance=15, 
        threshold_abs=8
    )
    markers = np.zeros_like(binary_image, dtype=np.int32)
    for i, (row, col) in enumerate(local_max):
        markers[row, col] = i + 1
    
    watershed_labels = segmentation.watershed(-dist_transform, markers, mask=binary_image)
    watershed_grains = len(np.unique(watershed_labels)) - 1
    processed_images['watershed'] = watershed_labels
    return watershed_grains, watershed_labels

def visualize_results(image_a, image_b):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Rice Grain Counting', fontsize=16)
    
    axes[0,0].imshow(image_a, cmap='gray')
    axes[0,0].set_title('Image 8a: Gaussian noise')
    axes[0,0].axis('off')
    axes[0,1].imshow(image_b, cmap='gray')
    axes[0,1].set_title('Image 8b: Saltpepper noise')
    axes[0,1].axis('off')
    axes[0,2].imshow(processed_images['gaussian_denoised'], cmap='gray')
    axes[0,2].set_title('8a: Gaussian denoised')
    axes[0,2].axis('off')
    axes[0,3].imshow(processed_images['saltpepper_denoised'], cmap='gray')
    axes[0,3].set_title('8b: Salt-Pepper denoised')
    axes[0,3].axis('off')

    axes[1,0].imshow(processed_images['otsu_segmented'], cmap='gray')
    axes[1,0].set_title('Otsu segmentation')
    axes[1,0].axis('off')
    axes[1,1].imshow(processed_images['morphological'], cmap='gray')
    axes[1,1].set_title('Morphological cleaned')
    axes[1,1].axis('off')
    axes[1,2].imshow(processed_images['connected_parts'], cmap='gray')
    axes[1,2].set_title('Valid connected parts')
    axes[1,2].axis('off')
    axes[1,3].imshow(processed_images['watershed'], cmap='nipy_spectral')
    axes[1,3].set_title('Watershed segmentation')
    axes[1,3].axis('off')
    
    for i in range(4):
        for j in range(4):
            if i >= 2 and j >= 1:
                axes[i,j].axis('off')
            elif i >= 3:
                axes[i,j].axis('off')
    
    plt.tight_layout()
    plt.show()

def process_complete_flow(image_path_a, image_path_b):
    image_a = load_image(image_path_a)
    image_b = load_image(image_path_b)
    
    results = {}

    denoised_a = preprocess_gaussian_noise(image_a)
    binary_a, thresh_a = apply_otsu_segmentation(denoised_a)
    morphed_a = apply_morphological_operations(binary_a)
    count_a, labels_a, stats_a, centroids_a, valid_a = count_connected_parts(morphed_a)
    watershed_count_a, watershed_labels_a = watershed_segmentation(morphed_a)
    
    results['image_a'] = {
        'connected_parts': count_a,
        'watershed': watershed_count_a,
        'threshold': thresh_a
    }
    
    denoised_b = preprocess_salt_pepper_noise(image_b)
    binary_b, thresh_b = apply_otsu_segmentation(denoised_b)
    morphed_b = apply_morphological_operations(binary_b)
    count_b, labels_b, stats_b, centroids_b, valid_b = count_connected_parts(morphed_b)
    watershed_count_b, watershed_labels_b = watershed_segmentation(morphed_b)
    
    results['image_b'] = {
        'connected_parts': count_b,
        'watershed': watershed_count_b,
        'threshold': thresh_b
    }
    
    visualize_results(image_a, image_b)
    return results


rice_img = cv2.imread('assets/rice.png', cv2.IMREAD_GRAYSCALE)
noise_gaussian = np.random.normal(0, 25, rice_img.shape).astype(np.float32)
rice_gaussian = np.clip(rice_img.astype(np.float32) + noise_gaussian, 0, 255).astype(np.uint8)
rice_gausian_path = 'assets/rice_gaussian.jpg'
cv2.imwrite(rice_gausian_path, rice_gaussian)

img_saltpepper = rice_img.copy()
salt = np.random.rand(*rice_img.shape) < 0.05
pepper = np.random.rand(*rice_img.shape) < 0.05
img_saltpepper[salt] = 255
img_saltpepper[pepper] = 0
rice_saltpepper_path = 'assets/rice_saltpepper.jpg'
cv2.imwrite(rice_saltpepper_path, img_saltpepper)

process_complete_flow(rice_gausian_path, rice_saltpepper_path)