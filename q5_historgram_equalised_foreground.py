import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('assets/jeniffer.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
hue = img_hsv[:, :, 0]
saturation = img_hsv[:, :, 1]
value = img_hsv[:, :, 2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(hue, cmap='gray')
axes[0].set_title('Hue Plane')
axes[0].axis('off')

axes[1].imshow(saturation, cmap='gray')
axes[1].set_title('Saturation Plane')
axes[1].axis('off')

axes[2].imshow(value, cmap='gray')
axes[2].set_title('Value Plane')
axes[2].axis('off')

plt.tight_layout()
plt.show()

threshold_value = 30
_, mask = cv2.threshold(saturation, threshold_value, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
foreground_masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
foreground_gray = cv2.cvtColor(foreground_masked, cv2.COLOR_RGB2GRAY)
foreground_pixels = foreground_gray[mask > 0]

hist_original = cv2.calcHist([foreground_pixels], [0], None, [256], [0, 256])
cumsum = np.cumsum(hist_original)
cumsum_masked = np.ma.masked_equal(cumsum, 0)
cumsum_normalized = (cumsum_masked - cumsum_masked.min()) * 255 / (cumsum_masked.max() - cumsum_masked.min())
cumsum_normalized = np.ma.filled(cumsum_normalized, 0).astype('uint8')
equalized_foreground = cumsum_normalized[foreground_gray]

result = img_rgb.copy()
result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

for i in range(3):
    channel = result[:, :, i]
    scaling_factor = np.mean(channel[mask > 0]) / np.mean(equalized_foreground[mask > 0]) if np.mean(equalized_foreground[mask > 0]) > 0 else 1
    channel[mask > 0] = np.clip(equalized_foreground[mask > 0] * scaling_factor, 0, 255)
    result[:, :, i] = channel

background_mask = cv2.bitwise_not(mask)
background = cv2.bitwise_and(img_rgb, img_rgb, mask=background_mask)
final_result = cv2.add(background, cv2.bitwise_and(result, result, mask=mask))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(hue, cmap='gray')
axes[0, 0].set_title('Hue Plane')
axes[0, 0].axis('off')

axes[0, 1].imshow(saturation, cmap='gray')
axes[0, 1].set_title('Saturation Plane')
axes[0, 1].axis('off')

axes[0, 2].imshow(value, cmap='gray')
axes[0, 2].set_title('Value Plane')
axes[0, 2].axis('off')

axes[1, 0].imshow(mask, cmap='gray')
axes[1, 0].set_title('Foreground Mask')
axes[1, 0].axis('off')

axes[1, 1].imshow(img_rgb)
axes[1, 1].set_title('Original Image')
axes[1, 1].axis('off')

axes[1, 2].imshow(final_result)
axes[1, 2].set_title('Histogram equalized foreground')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(foreground_pixels, bins=256, alpha=0.7, color='blue', label='Original')
plt.title('Original foreground histogram')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
equalized_pixels = equalized_foreground[mask > 0]
plt.hist(equalized_pixels, bins=256, alpha=0.7, color='red', label='Equalized')
plt.title('Equalized foreground histogram')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()