import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

def gaussian_vibrance_transform(x, a, sigma=70):
    gaussian_term = 128 * np.exp(-((x - 128)**2) / (2 * sigma**2))
    enhanced = x + a * gaussian_term
    return np.clip(enhanced, 0, 255)

def to_hsv_manual(rgb_image):
    rgb_normalized = rgb_image.astype(np.float32) / 255.0
    r, g, b = rgb_normalized[:,:,0], rgb_normalized[:,:,1], rgb_normalized[:,:,2]
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    delta = max_val - min_val
    v = max_val
    s = np.where(max_val == 0, 0, delta / max_val)
    h = np.zeros_like(max_val)

    mask_r = (max_val == r) & (delta != 0)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)

    mask_g = (max_val == g) & (delta != 0)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    
    mask_b = (max_val == b) & (delta != 0)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    
    return h, s, v

def to_rgb_manual(h, s, v):
    h = h / 60.0
    c = v * s
    x = c * (1 - np.abs((h % 2) - 1))
    m = v - c
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
    
    mask0 = (h >= 0) & (h < 1)
    mask1 = (h >= 1) & (h < 2)
    mask2 = (h >= 2) & (h < 3)
    mask3 = (h >= 3) & (h < 4)
    mask4 = (h >= 4) & (h < 5)
    mask5 = (h >= 5) & (h < 6)
    
    r[mask0], g[mask0], b[mask0] = c[mask0], x[mask0], 0
    r[mask1], g[mask1], b[mask1] = x[mask1], c[mask1], 0
    r[mask2], g[mask2], b[mask2] = 0, c[mask2], x[mask2]
    r[mask3], g[mask3], b[mask3] = 0, x[mask3], c[mask3]
    r[mask4], g[mask4], b[mask4] = x[mask4], 0, c[mask4]
    r[mask5], g[mask5], b[mask5] = c[mask5], 0, x[mask5]
    
    rgb = np.stack([r + m, g + m, b + m], axis=2)
    return (rgb * 255).astype(np.uint8)

def split_hsv_planes(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, s, v = to_hsv_manual(img)
    return img, h, s, v

def apply_vibrance_enhancement(image_path, a_values):
    original_img, h, s, v = split_hsv_planes(image_path)
    s_255 = (s * 255).astype(np.uint8)
    
    enhancement = {}
    for a in a_values:
        s_impr = gaussian_vibrance_transform(s_255, a)
        s_impr = s_impr / 255.0
        impr_rgb = to_rgb_manual(h, s_impr, v)

        enhancement[a] = {
            'impr_rgb': impr_rgb,
            's_original': s,
            's_impr': s_impr,
            's_original': s_255,
            's_impr': s_impr
        }
    return original_img, h, s, v, enhancement

def plot_transformation_curve(a_values, sigma=70):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.linspace(0, 255, 256)
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, a in enumerate(a_values):
        y = gaussian_vibrance_transform(x, a, sigma)
        ax1.plot(x, y, color=colors[i], linewidth=2, label=f'a = {a}')
    
    ax1.plot(x, x, 'k--', alpha=0.5, label='Identity')
    ax1.set_xlabel('Input saturation')
    ax1.set_ylabel('Output saturation')
    ax1.set_title('Vibrance enhanced transformation curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 255)
    ax1.set_ylim(0, 255)
    
    gaussian_component = 128 * np.exp(-((x - 128)**2) / (2 * sigma**2))
    ax2.plot(x, gaussian_component, 'red', linewidth=2, label='Gaussian component')
    ax2.set_xlabel('Input Intensity')
    ax2.set_ylabel('Gaussian enhancement factor')
    ax2.set_title('Gaussian enhancement component')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 255)
    
    plt.tight_layout()
    plt.show()

def display_hsv_planes(original_img, h, s, v):
    _, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].imshow(original_img)
    axes[0,0].set_title('Original rgb image')
    axes[0,0].axis('off')
    
    h_display = np.ones_like(h)
    hue_image = to_rgb_manual(h, h_display, h_display)
    axes[0,1].imshow(hue_image)
    axes[0,1].set_title('Hue Plane')
    axes[0,1].axis('off')
    axes[1,0].imshow(s, cmap='gray')
    axes[1,0].set_title('Saturation Plane')
    axes[1,0].axis('off')
    axes[1,1].imshow(v, cmap='gray')
    axes[1,1].set_title('Value Plane')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_vibrance_results(original_img, results, a_values):
    n_results = len(a_values) + 1 
    _, axes = plt.subplots(2, n_results, figsize=(4*n_results, 8))

    axes[0,0].imshow(original_img)
    axes[0,0].set_title('Original image')
    axes[0,0].axis('off')
    axes[1,0].imshow(results[a_values[0]]['s_original'], cmap='gray')
    axes[1,0].set_title('Original saturation')
    axes[1,0].axis('off')
    
    for i, a in enumerate(a_values, 1):
        axes[0,i].imshow(results[a]['impr_rgb'])
        axes[0,i].set_title(f'Enhanced a={a}')
        axes[0,i].axis('off')
        axes[1,i].imshow(results[a]['s_impr'], cmap='gray')
        axes[1,i].set_title(f'Enhanced saturation a= {a}')
        axes[1,i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_saturation_histograms(results, a_values):
    _, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = ['black', 'blue', 'green', 'red']
    for i, a in enumerate([None] + a_values):
        s_data = results[a]['s_impr']
        title = f'Enhanced saturation histogram a= {a}'
        color = colors[i]     
        axes[i].hist(s_data.flatten(), bins=50, alpha=0.7, color=color, density=True)
        axes[i].set_title(title)
        axes[i].set_xlabel('Saturation Value')
        axes[i].set_ylabel('Density')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 255)
    
    plt.tight_layout()
    plt.show()

def calculate_vibrance_metrics(original_img, impr_img):
    orig_lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
    enh_lab = cv2.cvtColor(impr_img, cv2.COLOR_RGB2LAB)
    delta_e = np.sqrt(np.sum((orig_lab.astype(float) - enh_lab.astype(float))**2, axis=2))
    mean_delta_e = np.mean(delta_e)
    
    orig_hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
    enh_hsv = cv2.cvtColor(impr_img, cv2.COLOR_RGB2HSV)
    sat_increase = np.mean(enh_hsv[:,:,1]) - np.mean(orig_hsv[:,:,1])
    mse = np.mean((original_img.astype(float) - impr_img.astype(float))**2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    return {
        'delta_e': mean_delta_e,
        'saturation_increase': sat_increase,
        'psnr': psnr
    }

def find_optimal_a_value(original_img, results):
    best_scores = {}
    for a in results.keys():
        enhanced = results[a]['impr_rgb']
        metrics = calculate_vibrance_metrics(original_img, enhanced)
        saturation_score = min(metrics['saturation_increase'] * 10, 10)
        color_preservation_score = max(0, 10 - metrics['delta_e'] / 5)
        quality_score = min(metrics['psnr'] / 5, 10)
        score = (saturation_score * 0.5 + 
                        color_preservation_score * 0.3 + 
                        quality_score * 0.2)
        
        best_scores[a] = {
            'score': score,
        }
        print(f"a= {a:.1f}:")
    
    optimal_a = max(best_scores.keys(), key=lambda x: best_scores[x]['score'])
    print(f"Optimal a value= {optimal_a}")

    Image.fromarray(optimal_a['impr_rgb']).save(f'vibrance_{optimal_a['impr_rgb']}.jpg')

if __name__ == "__main__":
    image_path = 'assets/spider.png'
    
    a_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    original_img, h, s, v, results = apply_vibrance_enhancement(image_path, a_values)
    
    plot_transformation_curve(a_values)
    display_hsv_planes(original_img, h, s, v)
    compare_vibrance_results(original_img, results, [0.3, 0.5, 0.7])
    plot_saturation_histograms(results, [0.3, 0.5, 0.7])
    find_optimal_a_value(original_img, results)