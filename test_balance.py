import cv2
import numpy as np
import matplotlib.pyplot as plt

# 自動調整白平衡的函數
def auto_white_balance(image):
    # 將圖像轉換為浮點型進行計算
    image_float = image.astype(np.float32)
    
    # 計算每個通道 (B, G, R) 的平均值
    avg_b = np.mean(image_float[:, :, 0])
    avg_g = np.mean(image_float[:, :, 1])
    avg_r = np.mean(image_float[:, :, 2])
    
    # 根據紅色與藍色通道的平均值判斷冷暖色
    if avg_r > avg_b:
        print("Image is warm, applying cool adjustment...")
        # 偏暖色，增強藍色通道
        correction_factor = avg_r / avg_b
        image_float[:, :, 0] *= correction_factor
    else:
        print("Image is cool, applying warm adjustment...")
        # 偏冷色，增強紅色通道
        correction_factor = avg_b / avg_r
        image_float[:, :, 2] *= correction_factor
    
    # 確保像素值保持在 [0, 255] 範圍內
    image_float = np.clip(image_float, 0, 255).astype(np.uint8)
    
    return image_float

# 載入圖像
image_path = 'your_image_path_here.jpg'  # 請替換為你的圖像路徑
image = cv2.imread(image_path)

# 進行白平衡調整
balanced_image = auto_white_balance(image)

# 將調整後的圖像轉換為 RGB 以便顯示
balanced_image_rgb = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2RGB)

# 顯示原始圖像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

# 顯示白平衡調整後的圖像
plt.subplot(1, 2, 2)
plt.imshow(balanced_image_rgb)
plt.title("White Balanced Image")
plt.axis('off')

plt.show()

# 保存調整後的圖像
cv2.imwrite('balanced_image.jpg', balanced_image)
