import cv2
import numpy as np

# 讀取影像
image_path = '20221012-00388_Color_cutting0.png'
image = cv2.imread(image_path)

# 初始化GrabCut相關變量
mask = np.zeros(image.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 定義矩形區域，根據圖片內容進行調整
rect = (10, 10, image.shape[1]-10, image.shape[0]-10)

# 執行GrabCut
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 修改mask，將確定為前景和可能為前景的區域設為1，其餘設為0
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 將mask應用到影像上
result = image * mask2[:, :, np.newaxis]

# 轉為灰度圖以便進行輪廓檢測
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# 找到輪廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 創建一個黑色影像來繪製縮小後的輪廓
shrinked_contour_image = np.zeros_like(gray)

# 繪製縮小後的輪廓
for contour in contours:
    shrink_factor = 0.95  # 縮小比例
    M = cv2.moments(contour)
    if M['m00'] == 0:
        continue
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    contour = contour - [cx, cy]
    contour = contour * shrink_factor
    contour = contour + [cx, cy]
    contour = contour.astype(np.int32)
    
    cv2.drawContours(shrinked_contour_image, [contour], -1, (255), thickness=cv2.FILLED)

# 將結果與原始影像結合以顯示縮小後的主體
final_result = cv2.bitwise_and(image, image, mask=shrinked_contour_image)

# 找到非黑色的邊界來進行裁剪
y_indices, x_indices = np.where(shrinked_contour_image == 255)
if len(x_indices) > 0 and len(y_indices) > 0:
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    cropped_result = final_result[y_min:y_max+1, x_min:x_max+1]
else:
    cropped_result = final_result

# 保存結果
output_path_cropped = 'eroded_result_cropped.png'
cv2.imwrite(output_path_cropped, cropped_result)