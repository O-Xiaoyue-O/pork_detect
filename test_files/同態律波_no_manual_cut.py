import cv2
import numpy as np
import os
from PIL import Image

oil_img = cv2.imread('darkened_image2.png',cv2.IMREAD_GRAYSCALE)
mask = cv2.imread('darkened_image2.png')
x,y=oil_img.shape
print(oil_img.shape)

for i in range(x):
    for j in range(y):
        if oil_img[i,j]>110:
            oil_img[i,j]=255
        else:
            oil_img[i,j]=0
black = 0
white = 0
for i in range(x):
    for j in range(y):
        if oil_img[i,j]==0:
            black+=1
        else:
            white+=1
#print("白色:",white)
#print("黑色:",black)
# 讀取二值化後的遮罩
img_mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

# 找到黑色區域，將黑色區域標記為255，其餘區域標記為0
ret, black_area = cv2.threshold(img_mask_gray, 1, 255, cv2.THRESH_BINARY_INV)

# 計算黑色區域的面積
black_area_size = cv2.countNonZero(black_area)

print("里肌肉面積:", black_area_size, " 像素")

rate1 = white/(x*y)
rate2 = black/(x*y)
print("油花占比:",round(rate1*100,2),'%')
#print("黑色占比:",round(rate2*100,2),'%')
cv2.imwrite("new_result.jpg", oil_img)