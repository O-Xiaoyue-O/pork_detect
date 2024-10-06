import numpy as np
import cv2

# RGB 到 Lab 的轉換函數
def rgb_to_lab(rgb):
    # 將 RGB 值轉換為浮點數
    rgb = rgb.astype('float32') / 255.0
    
    # 將 RGB 值轉換為 Lab 值
    lab = cv2.cvtColor(rgb.reshape(1, 1, 3), cv2.COLOR_RGB2Lab)
    
    # 提取 Lab 值中的 L 分量並返回
    return lab[0, 0, 0]

# 讀取 24 位彩色圖片
image = cv2.imread('result.png')

# 將圖片轉換為 numpy 陣列
image_array = np.array(image)

# 初始化 L 分量的總和和有效像素計數
l_sum = 0
valid_pixel_count = 0

# 遍歷圖片的每個像素
for i in range(image_array.shape[0]):
    for j in range(image_array.shape[1]):
        # 提取每個像素的 RGB 值
        rgb = image_array[i, j]
        
        # 檢查是否為黑色像素
        if np.array_equal(rgb, [0, 0, 0]):
            continue  # 跳過黑色像素
        
        # 將 RGB 值轉換為 Lab 值
        lab = rgb_to_lab(rgb)
        
        # 累加 L 分量的值
        l_sum += lab 
        
        # 增加有效像素計數
        valid_pixel_count += 1

# 計算 L 分量的平均值
if valid_pixel_count > 0:
    l_mean = l_sum / valid_pixel_count
else:
    l_mean = 0  # 如果沒有有效像素，設為 0

print("L 分量的平均值：", l_mean)
