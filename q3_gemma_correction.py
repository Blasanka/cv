import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage import color
import seaborn as sns

def to_lab(img):
    img_normalized = img.astype(np.float64) / 255.0
    lab_img = color.rgb2lab(img_normalized)
    return lab_img

def to_rgb(lab_img):
    rgb_img = color.lab2rgb(lab_img)
    rgb_img = np.clip(rgb_img * 255, 0, 255).astype(np.uint8)
    return rgb_img

def gamma_correction(l_channel, gamma):
    l_normalized = l_channel / 100.0
    corrected_normalized = np.power(l_normalized, 1.0/gamma)
    l_corrected = corrected_normalized * 100.0
    return l_corrected

def check_gamma_values(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lab_image = to_lab(img)
    l_channel = lab_image[:, :, 0]
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]

    gamma_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    results = {}
    
    for gamma in gamma_values:
        l_corrected = gamma_correction(l_channel, gamma)
        lab_corrected = np.stack([l_corrected, a_channel, b_channel], axis=2)
        rgb_corrected = to_rgb(lab_corrected)
        results[gamma] = {
            'l_original': l_channel,
            'l_corrected': l_corrected,
            'lab_corrected': lab_corrected,
            'rgb_corrected': rgb_corrected,
        }
    return img, lab_image, results

def plot_gamma_comparison(original_img, results, selected_gammas):
    _, axes = plt.subplots(2, len(selected_gammas), figsize=(16, 8))
    
    for i, gamma in enumerate(selected_gammas):
        if gamma == 1.0:
            axes[0, i].imshow(original_img)
            axes[0, i].set_title(f'Original y={gamma}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(results[gamma]['l_original'], cmap='gray')
            axes[1, i].set_title(f'L Channel y={gamma}')
            axes[1, i].axis('off')
        else:
            axes[0, i].imshow(results[gamma]['rgb_corrected'])
            axes[0, i].set_title(f'Corrected y={gamma}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(results[gamma]['l_corrected'], cmap='gray')
            axes[1, i].set_title(f'L Channel y={gamma}')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_histograms_comparison(results, selected_gammas):
    _, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'red', 'purple']
    for i, gamma in enumerate(selected_gammas):
        if gamma == 1.0:
            l_value = results[gamma]['l_original']
            title = f'Original L histogram y= {gamma}'
        else:
            l_value = results[gamma]['l_corrected']
            title = f'Corrected L histogram y= {gamma}'
        
        axes[i].hist(l_value.flatten(), bins=50, alpha=0.7, color=colors[i], range=(0, 100), density=True)
        axes[i].set_title(title)
        axes[i].set_xlabel('L Value')
        axes[i].set_ylabel('Density')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 100)
    
    plt.tight_layout()
    plt.show()

def plot_gamma_curve(gamma_values):
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 1, 256)
    colors = ['red', 'black', 'blue', 'green']    

    for i, gamma in enumerate(gamma_values):
        y = np.power(x, 1.0/gamma)
        label = f'y= {gamma}'
        if gamma == 1.0:
            label += ' original'
        plt.plot(x, y, color=colors[i], linewidth=2, label=label)
    
    plt.xlabel('Input lightness normalized')
    plt.ylabel('Output lightness normalized')
    plt.title('Gamma correction curves in lab color space')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    plt.show()

if __name__ == "__main__":
    img_path = 'assets/highlights_and_shadows.jpg'
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lab_img = to_lab(img)
    l_channel = lab_img[:,:,0]
    a_channel = lab_img[:,:,1]
    b_channel = lab_img[:,:,2]

    results = {}
    
    gamma_test_values = [0.5, 1.0, 1.5, 2.2]
    gamma_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.2, 2.5]
    plot_gamma_curve(gamma_test_values)
    
    for gamma in gamma_values:
        l_corrected = gamma_correction(l_channel, gamma)
        lab_corrected = np.stack([l_corrected, a_channel, b_channel], axis=2)
        rgb_corrected = to_rgb(lab_corrected)
        results[gamma] = {
            'l_original': l_channel,
            'l_corrected': l_corrected,
            'lab_corrected': lab_corrected,
            'rgb_corrected': rgb_corrected,
        }

    plot_gamma_comparison(img, results, gamma_test_values)
    plot_histograms_comparison(results, gamma_test_values)

    gamma = 2.2
    optimal_result = results[gamma]
    corrected_pil = Image.fromarray(optimal_result['rgb_corrected'])
    corrected_pil.save(f'gamma_corrected_image_gamma_{gamma}.jpg')