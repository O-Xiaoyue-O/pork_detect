from segment.predict_edited import segmenter_detection_cut
import cv2
import numpy as np
import os
from PIL import Image
import argparse
from pathlib import Path
import sys
from utils.general import LOGGER, print_args
from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from threading import Thread
import time
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.color import rgb2lab

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

M = np.array([[0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]])


"""Global variable start"""
t_value_lmean = [None] * 1
t_value_black_area_size = [None] * 1
t_value_rate1 = [None] * 1
t_value_avg_result = [None] * 1
"""Global variable end"""

class color_conversion_method():
    def __init__(self):
        """暫無需初始化
        
        參數: 無
        """
        pass
    
    def run(self, ccm_image, save_path, ognl_path, num:int, view_img=False, save_image=True):
        """運行色彩轉換法

        Args(required):
            ccm_image (cv2_image; np.array): 圖片
            save_path (Path): 儲存路徑
            ognl_path (str): 原始圖片路徑，用於取名
            num (int): 取名使用
            view_img (bool, optional): 顯示預覽. 預設 False.
            save_image (bool, optional): 儲存圖片. 預設 True.
        """
        # RGB 到 Lab 的轉換函數
        start_time = time.time()

        # 讀取 24 位彩色圖片
        # image = cv2.imread('result.png')
        image = ccm_image

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
                lab = self.rgb_to_lab(rgb)
                
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
        t_value_lmean[0] = l_mean
        print(f"色彩轉換法計算消耗時間: {time.time() - start_time}")
        return l_mean

    def rgb_to_lab(self, rgb):
            # 將 RGB 值轉換為浮點數
            rgb = rgb.astype('float32') / 255.0
            
            # 將 RGB 值轉換為 Lab 值
            lab = cv2.cvtColor(rgb.reshape(1, 1, 3), cv2.COLOR_RGB2Lab)
            
            # 提取 Lab 值中的 L 分量並返回
            return lab[0, 0, 0]

class Save_csv:
    def __init__(self, path: str, name: str = 'result.csv'):
        self.path = path
        self.name = name
        # create csv file, if exist, add number to name
        if os.path.isfile(os.path.join(path, name)):
            count = 1
            while True:
                self.name = f"result_{count}.csv"
                if not os.path.isfile(os.path.join(path, self.name)):
                    break
                count += 1
        with open(os.path.join(path, self.name), 'w') as f:
            f.write('Filename,Area size,Proportion,conf,L mean\n')
        
    def write_data(self, data: list):
        assert isinstance(data, list), "data must be a list"
        # assert len(data) == 5, "data must have 5 elements"
        with open(os.path.join(self.path, self.name), 'a') as f:
            if len(data) == 8:
                f.write(f"{data[0]},{data[1]},{data[2]},{data[3]},{data[4]},{data[5]},{data[6]},{data[7]}\n")
            else:
                f.write(f"{data[0]},{data[1]},{data[2]},{data[3]},{data[4]}\n")
            
class ResizeAndPad:
    def __init__(self, target_size):
        self.target_size = target_size
        self.resize = transforms.Resize(target_size)
    
    def __call__(self, img):
        # First, resize the image while maintaining the aspect ratio
        img = transforms.functional.resize(img, self.target_size, transforms.InterpolationMode.BILINEAR)
        
        # Calculate padding to ensure the image is centered
        padding_left = (self.target_size[1] - img.size[0]) // 2
        padding_right = self.target_size[1] - img.size[0] - padding_left
        padding_top = (self.target_size[0] - img.size[1]) // 2
        padding_bottom = self.target_size[0] - img.size[1] - padding_top
        
        # Apply the padding (fill with black, which is [0, 0, 0] for RGB)
        img = transforms.functional.pad(img, (padding_left, padding_top, padding_right, padding_bottom), fill=(0, 0, 0))
        
        return img

# Define the CNN model (same architecture as the one used for training)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Input: 512x512x3 -> Output: 512x512x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),       # Input: 512x512x16 -> Output: 256x256x16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Input: 256x256x16 -> Output: 256x256x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),       # Input: 256x256x32 -> Output: 128x128x32
        )
        # 計算展平後的特徵數量
        self.fc1_input_features = 32 * 128 * 128  # 32 channels, 128x128 size
        
        # 修正全連接層的輸入尺寸
        self.fc = nn.Sequential(
            nn.Linear(self.fc1_input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 最後一層輸出一個浮點數
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class cnn():
    def __init__(self, model_path: str = ROOT / 'models/best_oil_model.pth'):
        self.model = CNNModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.transform = transforms.Compose([
            ResizeAndPad((512, 512)),  # Resize while maintaining aspect ratio
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def predict_image(self, input_image):
        # cv2 image to PIL image
        image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Make predictions
        with torch.no_grad():
            output = self.model(image)
        
        # Post-process and interpret the results
        prediction = output.item()
        t_value_rate1[0] = prediction
        print(f"油花檢測結果: {prediction}")
        return prediction


def reduce_edge(image):
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
        shrink_factor = 0.8  # 縮小比例
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

    return cropped_result

def oil_detection(image, path, num, save_path, view_img=False, save_image=True):
    """_summary_

    Args:
        image (_type_): _description_
        path (_type_): _description_
        b (_type_): _description_
        save_path (_type_): _description_
        view_img (bool, optional): _description_. Defaults to False.
        save_image (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_ 
    """
    start_time = time.time()
    # color = ImageEnhance.Color(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    # output_color5 = color.enhance(3.5) # 飽和度
    # image = np.array(output_color5)
    # new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # new_image = homomorphic_filter(image)
    # image = cv2.medianBlur(image, 5)
    
    # negative_image_cv = cv2.bitwise_not(image)
    # gray_image_cv = cv2.cvtColor(negative_image_cv, cv2.COLOR_BGR2GRAY)
    # _, binary_image_cv = cv2.threshold(gray_image_cv, 127, 255, cv2.THRESH_BINARY)
    # new_image = cv2.bitwise_not(binary_image_cv)
    
    negative_image = 255 - image
    median_filtered_image = cv2.medianBlur(negative_image, 5)
    gray_image_filtered = cv2.cvtColor(median_filtered_image, cv2.COLOR_BGR2GRAY)
    _, binary_image_filtered = cv2.threshold(gray_image_filtered, 125, 255, cv2.THRESH_BINARY)
    inverted_binary_image = 255 - binary_image_filtered
    
    # binary_image_cv = cv2.adaptiveThreshold(gray_image_cv,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,10)
    LOGGER.info("Image transformed successfully.")
    # cv2.imwrite("new.jpg", img_new)
    # Mask = GetArea(img.copy())
    if view_img:
        Mask = inverted_binary_image.copy()
        LOGGER.info("Showing image...")
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('mask', 960, 540)
        cv2.imshow('mask', Mask)
        cv2.waitKey()
        LOGGER.info("Continue...")
    if save_image:
        LOGGER.info("Saving image...")
        # cv2.imwrite(str(save_path) + f"/{path}_cutting{num + 1}.png", image)
        cv2.imwrite(str(save_path) + f"/{path}_Homomorphic{num + 1}.png", inverted_binary_image)
    
    oil_img = inverted_binary_image.copy()
    x, y = oil_img.shape
    # print(oil_img.shape)
    
    for i in range(x):
        for j in range(y):
            if oil_img[i, j]>110:
                oil_img[i, j]=255
            else:
                oil_img[i, j]=0
    black = 0
    white = 0
    for i in range(x):
        for j in range(y):
            if oil_img[i, j]==0:
                black+=1
            else:
                white+=1
    # 讀取二值化後的遮罩
    img_mask_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 找到黑色區域，將黑色區域標記為255，其餘區域標記為0
    _, black_area = cv2.threshold(img_mask_gray, 1, 255, cv2.THRESH_BINARY_INV)
    # 計算黑色區域的面積
    black_area_size = cv2.countNonZero(black_area)
    print("里肌肉面積:", black_area_size, " 像素")
    try:
        rate1 = white/(x*y- black_area_size)
    except ZeroDivisionError:
        rate1 = 0
    # rate2 = black/(x*y)
    
    print(f"{path} 油花占比:",round(rate1*100,2) - 0.5,'%')
    if save_image:
        LOGGER.info("Saving image...")
        cv2.imwrite(str(save_path) + f"/{path}_result{num + 1}.png", oil_img)
    
    t_value_black_area_size[0] = black_area_size
    t_value_rate1[0] = round(rate1*100,2)
    print(f"油花計算消耗時間: {time.time() - start_time}")
    return black_area_size, rate1

def average_oil(image):
    # convert array image to PIL image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    W = 3
    H = 3
    width = image.size[0] // W
    height = image.size[1] // H
    start_x = 0
    start_y = 0
    crop_array = []
    for _ in range(H):
        for _ in range(W):
            crop = image.crop((start_x, start_y, start_x + width, start_y + height))
            # convert PIL image to array image
            crop = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
            crop_array.append(crop)
            start_x += width

        start_x = 0
        start_y += height
    
    data = []
    for num, crop_image in enumerate(crop_array):
        oil_img = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        x, y = oil_img.shape
        black = 0
        white = 0
        for i in range(x):
            for j in range(y):
                if oil_img[i, j] > 110:
                    oil_img[i, j] = 255
                else:
                    oil_img[i, j] = 0
                if oil_img[i, j] == 0:
                    black += 1
                else:
                    white += 1
        img_mask_gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        _, black_area = cv2.threshold(img_mask_gray, 1, 255, cv2.THRESH_BINARY_INV)
        _ = cv2.countNonZero(black_area)
        rate1 = white / (x * y)
        score = int(round(rate1 * 100))
        data.append(score)
        # print(f"處理後的影像 {num + 1} 的大理石花紋分數:", score)
    
    counter = Counter(data)
    total_count = len(data)
    probabilities = {num: count / total_count for num, count in counter.items()}

    min_prob = min(probabilities.values())
    max_prob = max(probabilities.values())

    print(f"機率的分布範圍為: {min_prob:.2%} ~ {max_prob:.2%}")
    
    if min_prob >= 0.4 and max_prob <= 0.6:
        t_value_avg_result[0] = "均勻分布"
    else:
        t_value_avg_result[0] = "不均勻分布"
    
# 自動調整白平衡的函數
def auto_white_balance(image):
    # 將圖像轉換為浮點型進行計算
    image_float = image.astype(np.float32)
    
    # 計算通道 (B, R) 的平均值
    avg_b = np.mean(image_float[:, :, 0])
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

# 240924 Add
# 去除背景并设置背景颜色
def remove_black_background(image):
    mask = np.all(image != [0, 0, 0], axis=-1)
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]
    
    # 设置背景区域为白色
    background_color = [255, 255, 255]  # 可以选择其他中性色
    masked_image[~mask] = background_color
    
    return masked_image

# 计算图像的颜色直方图
def compute_color_histogram(image):
    hist = []
    for i in range(3):  # 对每个颜色通道
        channel_hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist.append(channel_hist)
    return hist

# 进行颜色直方图匹配
def match_color_histogram(source_image, reference_hist):
    src_hist = compute_color_histogram(source_image)
    matched_image = np.zeros_like(source_image)
    
    for i in range(3):
        src_cdf = np.cumsum(src_hist[i])
        src_cdf /= src_cdf[-1]  # 归一化
        ref_cdf = np.cumsum(reference_hist[i])
        ref_cdf /= ref_cdf[-1]  # 归一化

        lookup_table = np.interp(src_cdf, ref_cdf, np.arange(256))
        matched_image[:, :, i] = lookup_table[source_image[:, :, i].astype(np.uint8)]

    return matched_image

# 将 RGB 值转换为 Lab 颜色空间并提取 L 分量
def rgb_to_lab(rgb):
    rgb = rgb.astype('float32') / 255.0
    lab = cv2.cvtColor(rgb.reshape(1, 1, 3), cv2.COLOR_RGB2Lab)
    return lab[0, 0, 0]

# 计算图像中非背景像素的 L 分量的平均值
def calculate_mean_l(image, background_color=[255, 255, 255]):
    # 创建一个掩码来跳过背景像素
    mask = np.all(image != background_color, axis=-1)
    
    # 仅选择非背景像素
    non_background_pixels = image[mask]

    if non_background_pixels.size == 0:
        return 0
    
    # 将RGB转为Lab
    lab_image = rgb2lab(non_background_pixels.reshape(-1, 1, 3))
    
    # 计算L通道的平均值
    l_mean = np.mean(lab_image[:, :, 0])

    return l_mean

def process_images(result_image, ref_image_1_path, ref_image_2_path):
    
    # 计算原始图像的平均 L 值
    masked_result_image = remove_black_background(result_image)
    mean_l_value = calculate_mean_l(masked_result_image)
    print(f"未均衡化的 L 值平均值：{mean_l_value}")

    # 根据 L 值选择参考图像
    if 43< mean_l_value <= 45:
        reference_image_path = ref_image_1_path  # 使用 result copy.png
    elif 20 <= mean_l_value <= 43:
        reference_image_path = ref_image_2_path  # 使用 result copy 4.png
    else:
        print("L 值不在指定範圍内，不匹配。")
        t_value_lmean[0] = mean_l_value
        return 0
    
    # 读取参考图像并计算其直方图
    reference_image = cv2.imread(reference_image_path)
    print("reading complete")
    masked_reference_image = remove_black_background(reference_image)
    print("remove_black_background complete")
    reference_hist = compute_color_histogram(masked_reference_image)
    print("compute_color_histogram complete")
    
    # 进行颜色直方图匹配
    matched_image = match_color_histogram(masked_result_image, reference_hist)
    print("match_color_histogram complete")
    mean_l = calculate_mean_l(matched_image)
    # save_images_and_histograms(masked_result_image, masked_reference_image, matched_image)
    t_value_lmean[0] = mean_l
    
def save_images_and_histograms(result_image, reference_image, matched_image):
    plt.figure(figsize=(20, 12))
    
    # 显示原图像 (result.png)
    plt.subplot(4, 4, 1)
    plt.title("原图像 (result.png)")
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 显示匹配后图像
    plt.subplot(4, 4, 2)
    plt.title("匹配后图像")
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 显示参考图像 (result copy.png 或 result copy 2.png)
    plt.subplot(4, 4, 3)
    plt.title("参考图像")
    plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

class api_detect():
    def __init__(
                self,
                weight=ROOT / 'runs/train-seg/pork_detection_V3/weights/best.pt',
                imgsz=(640, 640), # inference size (height, width)
                save_image: bool = True,
                augment: bool = False,
                half: bool = False,
                dnn: bool = False
        ):
        try:
            self.detecter = segmenter_detection_cut(
            weights=str(weight),
            imgsz=imgsz,
            save_img=save_image,
            augment=augment,
            half=half,
            dnn=dnn)
            self.cnn_detecter = cnn()
            
        except Exception as e:
            LOGGER.error(f"初始化出現錯誤，錯誤如下\nError: {e}")
            return e
    
    def run(self, 
            image_path
            ,save_path: Path = ROOT / 'results',
        ):
        start_time = time.time()
        yolo_save_path = save_path / 'cutting_images'
        os.makedirs(yolo_save_path, exist_ok=True)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        if width < height:
            print("height > width, rotate image")
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
        # 240924 Add
        # 自動調整白平衡
        image = auto_white_balance(image)
        csv = Save_csv(save_path)
        results = self.detecter.process(
            image, 
            project=yolo_save_path,
            name='cutting'
            )
        return_results_arr = []
        current_count = 0
        for _, result in enumerate(results):
            xyxy_arr, conf_arr, cutimgs = result
            LOGGER.info(f"Captured {len(cutimgs)} cutting images.")
            for b, (_, conf, cutimg) in enumerate(zip(xyxy_arr, conf_arr, cutimgs)):
                if conf < 0.5:
                    print(f"Confidence is lower than 0.7 / {conf}, skipping {b + 1}/{len(cutimgs)} cutting image.")
                    continue
                
                print(f"Confidence: {conf}")
                # LOGGER.info("Transforming image to Homomorphic filter...")
                # img = np.array(cutimg)
                # LOGGER.info("Reducing edge...")
                # cutimg = reduce_edge(cutimg)
                
                """ 偵測肉色 """
                l_mean_thread = Thread(target=process_images, args=(cutimg, "l_benchmark/result1.png", "l_benchmark/result2.png"))
                l_mean_thread.start()
                """ End of 偵測肉色 """
                
                """ 油花檢測 """
                oil_detection_thread = Thread(target=self.cnn_detecter.predict_image, args=([cutimg]))
                oil_detection_thread.start()
                """ End of 油花檢測 """
                
                """ 平均計算 """
                average_oil_thread = Thread(target=average_oil, args=(cutimg,))
                average_oil_thread.start()
                """ End of 平均計算 """
                
                """圖片背景透明化處理"""
                # Convert the image to RGBA
                if cutimg.shape[2] == 4:
                    rgba_image = cutimg
                else:
                    rgba_image = cv2.cvtColor(cutimg, cv2.COLOR_BGR2BGRA)
                
                black_threshold = 10
                black_mask = (rgba_image[:, :, 0] < black_threshold) & \
                            (rgba_image[:, :, 1] < black_threshold) & \
                            (rgba_image[:, :, 2] < black_threshold)

                rgba_image[black_mask, 3] = 0
                cv2.imwrite(str(save_path) + f"/_cutting{current_count + 1}.png", rgba_image)
                """End of 圖片背景透明化處理"""
                
                """ Thread join """
                l_mean_thread.join()
                oil_detection_thread.join()
                average_oil_thread.join()
                
                
                # 等級判斷式
                # 油花等級
                if t_value_rate1[0] < 1.5:
                    oil_level = 1
                elif t_value_rate1[0] < 2.5:
                    oil_level = 2
                elif t_value_rate1[0] < 3.5:
                    oil_level = 3
                elif t_value_rate1[0] < 4.5:
                    oil_level = 4
                else:
                    oil_level = 5
                
                # 肉色等級
                if t_value_lmean[0] > 61:
                    color_level = 1
                elif t_value_lmean[0] > 55:
                    color_level = 2
                elif t_value_lmean[0] > 49:
                    color_level = 3
                elif t_value_lmean[0] > 43:
                    color_level = 4
                else:
                    color_level = 5
                
                # 油花等級與肉色等級取最小值，作為最終等級
                if color_level > oil_level:
                    final_level = oil_level
                else:
                    final_level = color_level
                
                # 依據等級給予食譜
                if final_level == 1:
                    title = "醬燒豬肉(3~4人份)"
                    ingredients = "豬肉 300g、洋蔥 一顆、味醂 兩湯匙、醬油 一匙、水 兩匙"
                    steps = "第一步驟：豬肉先用醬料醃製10~20分、洋蔥切絲、蒜頭切末。\n第二步驟：熱油鍋先將蒜頭爆香。\n第三步驟：略炒兩分鐘後，加所有調味料下鍋，等顏色均勻即可起鍋。"
                elif final_level == 2:
                    title = "馬鈴薯燉肉(3~4人份)"
                    ingredients = "馬鈴薯 五顆、豬肉 300g、胡蘿蔔 三顆、洋蔥 一顆、醬油 一湯匙、味醂 一湯匙、鹽巴 適量、薑 適量、青蔥 適量、食用油 適量"
                    steps = "第一步驟：馬鈴薯去皮切塊，洋蔥切厚片，紅蘿蔔滾刀切塊，肉切塊備用。\n第二步驟：加入1公升的水且放入豬肉和薑蒜去腥。\n第三步驟：在鍋內加入食用油，放入馬鈴薯、洋蔥、紅蘿蔔、豬肉，輕輕炒至所有食材都沾上油分。\n第四步驟：加入1500cc的水煮滾後，加入調味料、所有食材，以大火煮滾後，再以小火慢燉，燉煮至湯汁收到一半即可。"
                elif final_level == 3:
                    title = "蒜泥白肉(3~4人份)"
                    ingredients = "豬肉 300g、蔥 一把、薑 適量、蒜頭 適量、辣椒 適量、米酒 兩匙、香油 半匙、醬油 四大匙"
                    steps = "第一步驟：準備好所有的食材，蔥切段、薑切片、蒜頭磨泥、辣椒切末。\n第二步驟：把豬肉和蔥、薑、米酒放入電鍋中，蒸煮15分鐘。\n第三步驟：燜10分鐘後取出豬肉將室溫放量。\n第四步驟：將蒜末和醬油膏混和，且把豬肉切片並擺盤。"
                elif final_level == 4:
                    title = "蒜香豬肉片(3~4人份)"
                    ingredients = "豬肉 300g、蒜頭 兩顆、蔥 一把、米酒 適量、鹽巴 適量"
                    steps = "第一步驟：蒜頭切片，將蔥白部分切成末、青色部分切成段。起一熱油鍋至160℃，將蒜片爆至兩面金黃色，起鍋備用。\n第二步驟：將豬肉下鍋，爆炒些許時間後再加入炒過的蒜片。\n第三步驟：加水50cc，翻炒至豬肉變色。\n第四步驟：將蔥下鍋憶起快速拌炒，最後加入少許米酒和鹽巴翻炒即可。"
                else:
                    title = "鹽烤豬肉(3~4人份)"
                    ingredients = "豬肉 300g、鹽 適量、胡椒粉 適量、檸檬汁 適量"
                    steps = "第一步驟：豬肉以鹽和胡椒粉抹勻，冷藏下醃2小時。\n第二步驟：將醃製好的豬肉放置烤盤上，200度 15分鐘，先將豬油倒出，再翻面200度 15分鐘，這時顏色也加深不少，完成後再將第二次的豬油倒出。\n第三步驟：切成入口大小且裝盤。"
                
                # 順序, 油花等級, 肉色等級, 最終等級, 食譜標題, 食材, 步驟
                csv.write_data([b + 1, t_value_rate1[0], t_value_lmean[0], final_level, t_value_avg_result[0], title, ingredients, steps])
                return_results_arr.append([b + 1, t_value_rate1[0], t_value_lmean[0], final_level, t_value_avg_result[0], title, ingredients, steps])
                LOGGER.info(f"Finish processing cutting image {b+1}/{len(cutimgs)}")
                current_count += 1

        print(f"消耗時間: {time.time() - start_time}")
        return return_results_arr


def main(
        input_path: Path, # input file/folder (required)
        save_path: Path = ROOT / 'results',
        weight=ROOT / 'runs/train-seg/pork_detection_V3/weights/best.pt',
        imgsz=(640, 640), # inference size (height, width)
        save_image: bool = True,
        augment: bool = False,
        half: bool = False,
        dnn: bool = False,
        # con_threshold: float = 0.5,
        # iou_thres: float = 0.45,
        # max_det: int = 1000,
        # save_crop: bool = False,
        # agnostic_nms: bool = False,
        # visualize: bool = False,
        # line_thickness: int = 3,
        # hide_labels: bool = False,
        # hide_conf: bool = False,
        view_img: bool = False,
    ):
    # if no input_path, raise error
    LOGGER.info("Input path: %s"%input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    # read input_path image, if only one image, convert to list
    is_file = Path(input_path).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    if is_file:
        paths = [input_path]
        images = [cv2.imread(input_path)]
    else:
        paths = [str(f) for f in Path(input_path).glob('*')]
        images = []
        for i, f in enumerate(Path(input_path).glob('*')):
            if not os.path.isdir(f) and Path(f).suffix[1:] in IMG_FORMATS:
                images.append(cv2.imread(f))
            else:
                paths.remove(str(f))
                print(f"Skip: {f}")
            print(f"Reading image: {i+1}/{len(paths)}", end='\r')
    
    # create output folder if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # if folder is exist, rename folder
    if os.path.exists(save_path/"result1"):
        count = 1
        while True:
            if not os.path.exists(save_path/f"result{count}"):
                # create output folder
                os.makedirs(save_path/f"result{count}")
                save_path = save_path/f"result{count}"
                break
            count += 1
    else:
        os.makedirs(save_path/"result1")
        save_path = save_path/"result1"
    
    # 存檔準備
    csv = Save_csv(save_path)
    LOGGER.info("Loading model and prepare detection...")
    # 準備及載入模型
    detecter = segmenter_detection_cut(
        weights=str(weight),
        imgsz=imgsz,
        save_img=save_image,
        # project=save_path,
        augment=augment,
        half=half,
        dnn=dnn,
    )
    yolo_save_path = save_path / 'cutting_images'
    LOGGER.info("Model loaded successfully.")
    LOGGER.info("Start detecting loop...")
    for ii, (im0, path) in enumerate(zip(images, paths)):
        LOGGER.info("Processing image")
        # 預測並擷取圖片，回傳xyxy_arr, conf_arr, cutimgs，因為設計上是可以多張進行預測，所以回傳資料格式如下:
        """
        results = [[xyxy_arr, conf_arr, cutimgs], [xyxy_arr, conf_arr, cutimgs], ...]   
        """
        results = detecter.process(im0,
                                   project=yolo_save_path,
                                    name='cutting')
        # spilt file name from path
        path = Path(path).stem
        # xyxy_arr, conf_arr, cutimgs = result
        for xx, result in enumerate(results):
            # 將result解包
            """
            xyxy_arr: 檢測框座標，格式為[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
            conf_arr: 信心度，格式為[conf1, conf2, ...]
            cuttings: 輸出的裁剪圖片，格式為cv2格式
            """
            xyxy_arr, conf_arr, cutimgs = result
            LOGGER.info(f"Captured {len(cutimgs)} cutting images.")
            for b, (xyxy, conf, cutimg) in enumerate(zip(xyxy_arr, conf_arr, cutimgs)):
                LOGGER.info("Transforming image to Homomorphic filter...")
                # img = np.array(cutimg)
                LOGGER.info("Reducing edge...")
                cutimg = reduce_edge(cutimg)
                
                """ 色彩轉換法 """
                ccm = color_conversion_method()
                l_mean = ccm.run(cutimg.copy(), save_path, path, b, view_img=view_img, save_image=save_image)
                """ End of 色彩轉換法 """
                
                """ 油花檢測 """
                black_area_size, rate1 = oil_detection(cutimg, path, b, save_path, view_img=view_img, save_image=save_image)
                """ End of 油花檢測 """
                
                csv.write_data([path, black_area_size, round(rate1*100,2)-0.5, conf, l_mean])
                LOGGER.info(f"Finish processing cutting image {b+1}/{len(cutimgs)}")
        LOGGER.info(f"Finish processing image {ii+1}/{len(images)}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,  help='輸入圖片路徑(必要)', required=True)
    parser.add_argument('--save_path', type=str, default=ROOT / 'results', help='儲存圖片路徑(預設為new_mask.jpg) ')
    parser.add_argument('--weight', type=str, default=ROOT / 'runs/train-seg/pork_detection_V3/weights/best.pt', help='模型路徑')
    parser.add_argument('--imgsz', type=tuple, default=(640, 640), help='推論大小(高, 寬)')
    parser.add_argument('--save_image', type=bool, default=True, help='是否儲存圖片')
    parser.add_argument('--augment', type=bool, default=False, help='是否增強')
    parser.add_argument('--half', type=bool, default=False, help='是否使用half')
    parser.add_argument('--dnn', type=bool, default=False, help='是否使用dnn')
    # parser.add_argument('--con_threshold', type=float, default=0.5, help='信心閾值')
    # parser.add_argument('--iou_thres', type=float, default=0.45, help='iou閾值')
    # parser.add_argument('--max_det', type=int, default=1000, help='最大檢測數')
    # parser.add_argument('--save_crop', type=bool, default=False, help='是否儲存裁剪')
    # parser.add_argument('--agnostic_nms', type=bool, default=False, help='是否使用agnostic nms')
    # parser.add_argument('--visualize', type=bool, default=False, help='是否視覺化')
    # parser.add_argument('--line_thickness', type=int, default=3, help='線條厚度')
    # parser.add_argument('--hide_labels', type=bool, default=False, help='是否隱藏標籤')
    # parser.add_argument('--hide_conf', type=bool, default=False, help='是否隱藏信心')
    parser.add_argument('--view_img', type=bool, default=False, help='是否顯示圖片')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))