import numpy as np
import cv2
 
# region 輔助函數
# RGB2XYZ空間的系數矩陣
M = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])
 
# im_channel取值範圍：[0,1]
def f(im_channel):
    return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931
 
def anti_f(im_channel):
    return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
# endregion
 
# region RGB 轉 Lab
# 像素值RGB轉XYZ空間，pixel格式:(B,G,R)
# 返回XYZ空間下的值
def __rgb2xyz__(pixel):
    b, g, r = pixel[0], pixel[1], pixel[2]
    rgb = np.array([r, g, b])
    XYZ = np.dot(M, rgb.T)
    XYZ = XYZ / 255.0
    return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)
 
def __xyz2lab__(xyz):
    F_XYZ = [f(x) for x in xyz]
    L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
    a = 500 * (F_XYZ[0] - F_XYZ[1])
    b = 200 * (F_XYZ[1] - F_XYZ[2])
    return (L, a, b)
 
def RGB2Lab(pixel):
    xyz = __rgb2xyz__(pixel)
    Lab = __xyz2lab__(xyz)
    return Lab
 
# endregion
 
# region Lab 轉 RGB
def __lab2xyz__(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = Lab[1] / 500.0 + fY
    fZ = fY - Lab[2] / 200.0
 
    x = anti_f(fX)
    y = anti_f(fY)
    z = anti_f(fZ)
 
    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883
 
    return (x, y, z)
 
def __xyz2rgb(xyz):
    xyz = np.array(xyz)
    xyz = xyz * 255
    rgb = np.dot(np.linalg.inv(M), xyz.T)
    rgb = np.uint8(np.clip(rgb, 0, 255))
    return rgb
 
def Lab2RGB(Lab):
    xyz = __lab2xyz__(Lab)
    rgb = __xyz2rgb(xyz)
    return rgb
# endregion
 
if __name__ == '__main__':
    img = cv2.imread('20221012-00388_Color_cutting1.png')
    w, h, _ = img.shape
    lab = np.zeros((w, h, 3), dtype=np.uint8)
    l_channel = np.zeros((w, h), dtype=np.uint8)
    l_sum = 0  # 用于累加L分量
    for i in range(w):
        for j in range(h):
            Lab = RGB2Lab(img[i, j])
            lab[i, j] = (Lab[0], Lab[1], Lab[2])
            l_channel[i, j] = Lab[0]
            l_sum += Lab[0]# 累加L分量
    l_mean = l_sum / (w * h)  # 計算平均值
   

    # 直接保存只包含L分量的圖像
    cv2.imwrite('result_l_channel.png', l_channel)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (10, 30) 
    font_scale = 1
    color = (255, 255, 255)  # 文本顏色，白色
    thickness = 2
    img_with_text = cv2.putText(img.copy(), f'L Mean: {l_mean:.2f}', org, font, font_scale, color, thickness, cv2.LINE_AA)


    # 直接保存Lab圖像
    cv2.imwrite('result_lab.png', lab)
    cv2.imwrite('result_with_text.png', img_with_text)
    print("L分量的平均值：", l_mean)
