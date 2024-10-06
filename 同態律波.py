import cv2
import numpy as np
import os
from PIL import Image


def get_area_points(img):
    count = 1
    a = []
    b = []
    Points = []
    Points_list = []

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)

            print('顏色' + str(img[y, x, 0]) + ' ' + str(img[y, x, 1]) + ' ' + str(img[y, x, 2]))
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
            cv2.imshow("{}.jpg".format(count), img)
            Points.append([a[-1], b[-1]])
            print(a[-1], b[-1])

    while True:
        cv2.namedWindow("{}.jpg".format(count), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("{}.jpg".format(count), 960, 540)

        cv2.setMouseCallback("{}.jpg".format(count), on_EVENT_LBUTTONDOWN)

        cv2.imshow("{}.jpg".format(count), img)

        flag = cv2.waitKey()

        if flag == 32:
            if Points:
                Points_list.append(Points)
            Points = []
        if flag == 13:
            if Points:
                Points_list.append(Points)
            break
    return Points_list

def GetArea(img1):
    Mask1 = np.zeros(img1.shape, dtype='uint8')
    Pts_list = get_area_points(img1)
    
    if not Pts_list:
        print("未選擇任何點。")
        return None

    for Pts in Pts_list:
        Pts = np.array(Pts)
        cv2.fillPoly(Mask1, [Pts], (255, 255, 255))

    return Mask1

#同態濾波
def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape

    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)

    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M**2 + N**2)
    Z = (rh - r1) * (1 - np.exp(-c * (D**2 / d0**2))) + r1
    dst_fftshift = Z * gray_fftshift

    dst_fftshift = (h - 1) * dst_fftshift + 1
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)

    dst = np.uint8(np.clip(dst, 0, 255))
    return dst

path = "20221012-00388_Color.jpg"
if os.path.isfile(path):
    print("path {} is existence.".format(path))
    im = Image.open(path)
    img = np.array(im)
    print(im, img.shape)
    img_new = homomorphic_filter(img)
    print("new img shape is {}".format(img_new.shape))
    cv2.imwrite("new.png", img_new)

#同態濾波

if __name__ == '__main__':
    imgPath = "new.png"
    SavePath = "new_mask.png"
    isSave = True

    img = cv2.imread(imgPath)
    Mask = GetArea(img.copy())

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('mask', 960, 540)
    cv2.imshow('mask', Mask)
    cv2.waitKey()

    if isSave:
        cv2.imwrite(SavePath, Mask)
        print(f"圖像已保存到{Mask}")

    Mask = cv2.cvtColor(Mask, cv2.COLOR_RGB2BGR)

    img_mask = cv2.bitwise_and(img, img, mask=Mask[:,:,2])
    cv2.namedWindow('img_mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img_mask', 960, 540)
    cv2.imshow('img_mask', img_mask)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite("result.png", img_mask)

oil_img = cv2.imread('result.png',cv2.IMREAD_GRAYSCALE)
# Step 2: Find contours on the mask
contours, _ = cv2.findContours(Mask[:,:,2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 3: Get bounding box of the largest contour
x, y, w, h = cv2.boundingRect(contours[0])

# Step 4: Crop the original image using the bounding box coordinates
oil_img = oil_img[y:y+h, x:x+w]

# Save or display the cropped image
# cv2.imwrite('cropped_image.jpg', cropped_image)
cv2.imshow('Cropped Image', oil_img)
cv2.waitKey(0)
# cv2.destroyAllWindows()
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
img_mask_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)

# 找到黑色區域，將黑色區域標記為255，其餘區域標記為0
ret, black_area = cv2.threshold(img_mask_gray, 1, 255, cv2.THRESH_BINARY_INV)

# 計算黑色區域的面積
black_area_size = cv2.countNonZero(black_area)

print("里肌肉面積:", black_area_size, " 像素")

rate1 = white/(x*y)
rate2 = black/(x*y)
print("油花占比:",round(rate1*100,2),'%')
#print("黑色占比:",round(rate2*100,2),'%')
cv2.imwrite("new_result.png", oil_img)