import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.color import rgb2hsv, rgb2lab
from scipy import ndimage

original_image = None
processed_images = {}
focal_length_mm = 8.0
camera_distance_mm = 480.0 
        
    
def segment_sapphires():
    hsv_mask = segment_hsv_color(original_image)
    lab_mask = segment_lab_color(original_image)
    blue_mask = segment_blue_channel(original_image)
    otsu_mask = segment_otsu(original_image)
    kmeans_mask = segment_kmeans(original_image)
    combined_mask = combine_masks([hsv_mask, lab_mask, blue_mask, otsu_mask, kmeans_mask])
    
    processed_images.update({
        'hsv_mask': hsv_mask,
        'lab_mask': lab_mask, 
        'blue_mask': blue_mask,
        'otsu_mask': otsu_mask,
        'kmeans_mask': kmeans_mask,
        'combined_mask': combined_mask
    })
    
    return combined_mask

def segment_hsv_color(image):
    hsv = rgb2hsv(image)
    
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    blue_mask = ((h >= 0.5) & (h <= 0.75) &  (s >= 0.3) & (s <= 1.0) & (v >= 0.2) & (v <= 1.0))
    
    return blue_mask.astype(np.uint8) * 255

def segment_lab_color(image):
    lab = rgb2lab(image)
    
    l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]

    l_norm = (l - l.min()) / (l.max() - l.min())
    b_norm = (b - b.min()) / (b.max() - b.min())
    
    blue_mask = (b_norm < 0.4) & (l_norm > 0.2)
    return blue_mask.astype(np.uint8) * 255

def segment_blue_channel(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    blue_dominant = (b > r) & (b > g) & (b > 100)
    return blue_dominant.astype(np.uint8) * 255

def segment_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_mask

def segment_kmeans(image, k=3):
    data = image.reshape((-1, 3)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    labels = labels.reshape(image.shape[:2])
    blue_intensities = [centers[i][2] for i in range(k)]
    blue_cluster = np.argmax(blue_intensities)
    
    return (labels == blue_cluster).astype(np.uint8) * 255

def combine_masks(masks, threshold=0.6):
    mask_stack = np.stack([mask/255 for mask in masks], axis=-1)
    vote_count = np.sum(mask_stack, axis=-1)
    min_votes = len(masks) * threshold
    combined = (vote_count >= min_votes).astype(np.uint8) * 255
    return combined

def fill_holes_morphological(binary_mask):
    if binary_mask.dtype != bool:
        binary_mask = binary_mask > 0
    
    filled_simple = ndimage.binary_fill_holes(binary_mask)
    kernel = morphology.disk(5)
    filled_closing = morphology.closing(binary_mask, kernel)
    seed = morphology.erosion(binary_mask, morphology.disk(3))
    filled_reconstruction = morphology.reconstruction(seed, binary_mask)
    filled_remove_holes = morphology.remove_small_holes(binary_mask, area_threshold=100)
    final_filled = filled_remove_holes | filled_simple
    result = (final_filled * 255).astype(np.uint8)
    
    processed_images.update({
        'filled_simple': (filled_simple * 255).astype(np.uint8),
        'filled_closing': (filled_closing * 255).astype(np.uint8),
        'filled_reconstruction': (filled_reconstruction * 255).astype(np.uint8),
        'filled_final': result
    })
    
    return result

def analyze_connected_features(binary_mask):
    if binary_mask.dtype == bool:
        binary_mask = (binary_mask * 255).astype(np.uint8)
    elif binary_mask.max() <= 1:
        binary_mask = (binary_mask * 255).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    
    features_details = []
    for i in range(1, num_labels):
        area_pixels = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        centroid_x, centroid_y = centroids[i]
        
        features_details.append({
            'label': i,
            'area_pixels': area_pixels,
            'bbox': (x, y, width, height),
            'centroid': (centroid_x, centroid_y)
        })
    
    features_details.sort(key=lambda x: x['area_pixels'], reverse=True)
    processed_images['feature_labels'] = labels
    processed_images['features_details'] = features_details
    return features_details

def compute_areas(features_details):
    sensor_pixel_size_um = 3.5
    sensor_pixel_size_mm = sensor_pixel_size_um / 1000
    
    magnification = focal_length_mm / camera_distance_mm
    
    object_pixel_size_mm = sensor_pixel_size_mm / magnification
    object_pixel_size_mm2 = object_pixel_size_mm ** 2 
    
    actual_areas = []
    for i, comp in enumerate(features_details):
        area_pixels = comp['area_pixels']
        area_mm2 = area_pixels * object_pixel_size_mm2
        
        actual_areas.append({
            'component_id': i + 1,
            'area_pixels': area_pixels,
            'area_mm2': area_mm2,
            'centroid': comp['centroid'],
            'bbox': comp['bbox']
        })
    
    return actual_areas

def display_analysis():
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Sapphire segmentation and analysis pipeline', fontsize=16)
    
    axes[0,0].imshow(original_image)
    axes[0,0].set_title('Original sapphire image')
    axes[0,0].axis('off')
    axes[0,1].imshow(processed_images['hsv_mask'], cmap='gray')
    axes[0,1].set_title('HSV color segmentation')
    axes[0,1].axis('off')
    axes[0,2].imshow(processed_images['blue_mask'], cmap='gray')
    axes[0,2].set_title('Blue channel segmentation')
    axes[0,2].axis('off')
    axes[0,3].imshow(processed_images['kmeans_mask'], cmap='gray')
    axes[0,3].set_title('K-means segmentation')
    axes[0,3].axis('off')

    axes[1,0].imshow(processed_images['combined_mask'], cmap='gray')
    axes[1,0].set_title('Combined segmentation')
    axes[1,0].axis('off')
    axes[1,1].imshow(processed_images['filled_final'], cmap='gray')
    axes[1,1].set_title('Holes filled')
    axes[1,1].axis('off')
    axes[1,2].imshow(processed_images['feature_labels'], cmap='nipy_spectral')
    axes[1,2].set_title('Connected features')
    axes[1,2].axis('off')

    result_img = original_image.copy()
    for comp in processed_images['features_details']:
        x, y, w, h = comp['bbox']
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cx, cy = comp['centroid']
        cv2.circle(result_img, (int(cx), int(cy)), 5, (0, 255, 0), -1)
    
    axes[1,3].imshow(result_img)
    axes[1,3].set_title('Final Detection Results')
    axes[1,3].axis('off')
    
    for i in range(2, 3):
        for j in range(4):
            axes[i,j].axis('off')
    
    plt.tight_layout()
    plt.show()


original_image = cv2.imread('assets/sapphire.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

binary_mask = segment_sapphires()
filled_mask = fill_holes_morphological(binary_mask)
features_details = analyze_connected_features(filled_mask)
actual_areas = compute_areas(features_details)

display_analysis()