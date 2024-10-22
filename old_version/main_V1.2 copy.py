"""
Update:
1. 新增色彩轉換法
"""
from segment.predict_edited import segmenter_detection_cut
import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import argparse
from pathlib import Path
import sys
from utils.general import LOGGER, print_args
from utils.dataloaders import IMG_FORMATS, VID_FORMATS

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

M = np.array([[0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]])

class color_conversion_method():
    # im_channel取值範圍：[0,1]
    def f(self, im_channel):
        return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931

    def anti_f(self, im_channel):
        return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
    # endregion

    # region RGB 轉 Lab
    # 像素值RGB轉XYZ空間，pixel格式:(B,G,R)
    # 返回XYZ空間下的值
    def __rgb2xyz__(self, pixel):
        b, g, r = pixel[0], pixel[1], pixel[2]
        rgb = np.array([r, g, b])
        XYZ = np.dot(M, rgb.T)
        XYZ = XYZ / 255.0
        return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)

    def __xyz2lab__(self, xyz):
        F_XYZ = [self.f(x) for x in xyz]
        L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
        a = 500 * (F_XYZ[0] - F_XYZ[1])
        b = 200 * (F_XYZ[1] - F_XYZ[2])
        return (L, a, b)

    def RGB2Lab(self, pixel):
        xyz = self.__rgb2xyz__(pixel)
        Lab = self.__xyz2lab__(xyz)
        return Lab

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
        assert len(data) == 5, "data must have 5 elements"
        with open(os.path.join(self.path, self.name), 'a') as f:
            f.write(f"{data[0]},{data[1]},{data[2]},{data[3]},{data[4]}\n")


def is_dark_red(pil_image):
    # Load the image
    # image = cv2.imread(image_path)
    
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for dark red color in HSV
    lower_dark_red = np.array([0, 100, 20])
    upper_dark_red = np.array([10, 255, 100])
    lower_dark_red2 = np.array([170, 100, 20])
    upper_dark_red2 = np.array([180, 255, 100])
    
    # Create masks for dark red color
    mask1 = cv2.inRange(hsv_image, lower_dark_red, upper_dark_red)
    mask2 = cv2.inRange(hsv_image, lower_dark_red2, upper_dark_red2)
    
    # Combine masks
    mask = mask1 + mask2
    
    # Calculate the percentage of dark red pixels
    dark_red_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    percentage_dark_red = (dark_red_pixels / total_pixels) * 100
    
    # Threshold for considering the image as dark red
    threshold = 35  # For example, 50% of the image should be dark red
    # print(percentage_dark_red, percentage_dark_red > threshold)
    if percentage_dark_red > threshold:
        return True
    else:
        return False

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

    return cropped_result


def main(
        input_path: Path, # input file/folder (required)
        save_path: Path = ROOT / 'results',
        weight=ROOT / 'runs/train-seg/pork_detection_V2/weights/best.pt',
        imgsz=(640, 640), # inference size (height, width)
        save_image: bool = True,
        augment: bool = False,
        half: bool = False,
        dnn: bool = False,
        con_threshold: float = 0.5,
        iou_thres: float = 0.45,
        max_det: int = 1000,
        save_crop: bool = False,
        agnostic_nms: bool = False,
        visualize: bool = False,
        line_thickness: int = 3,
        hide_labels: bool = False,
        hide_conf: bool = False,
        view_img: bool = False,
    ):
    # if no input_path, raise error
    LOGGER.info("Input path: %s"%input_path)
    if not input_path:
        raise ValueError("input_path is required")
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
        augment=augment,
        half=half,
        dnn=dnn,
    )
    LOGGER.info("Model loaded successfully.")
    LOGGER.info("Start detecting loop...")
    for ii, (im0, path) in enumerate(zip(images, paths)):
        LOGGER.info("Processing image")
        # 預測並擷取圖片，回傳xyxy_arr, conf_arr, cutimgs，因為設計上是可以多張進行預測，所以回傳資料格式如下:
        """
        results = [[xyxy_arr, conf_arr, cutimgs], [xyxy_arr, conf_arr, cutimgs], ...]   
        """
        results = detecter.process(im0, path)
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
                # print(im0, img.shape)
                # convert cv2 to PIL
                # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # while not is_dark_red(img):
                #     # Adjust brightness
                #     enhancer = ImageEnhance.Brightness(img)
                #     factor = 0.975  # Decrease brightness by 97.5%
                #     img = enhancer.enhance(factor)
                # # convert PIL to cv2
                # img = np.array(img)
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                """ 色彩轉換法 """
                ccm_image = cutimg.copy()
                # convert to rgb
                ccm_image = cv2.cvtColor(ccm_image, cv2.COLOR_BGR2RGB)
                w, h, _ = ccm_image.shape
                lab = np.zeros((w, h, 3), dtype=np.uint)
                l_channel = np.zeros((w, h), dtype=np.uint8)
                l_sum = 0  # 用于累加L分量
                skip_count = 0
                skip_white_count = 0
                for i in range(w):
                    for j in range(h):
                        rgb_lab = color_conversion_method()
                        Lab = rgb_lab.RGB2Lab(ccm_image[i, j])
                        if Lab[0] == 0 and Lab[1] == 0 and Lab[2] == 0:
                            skip_count += 1
                            continue
                        if ccm_image[i, j][0] >= 200 and ccm_image[i, j][1] >= 110 and ccm_image[i, j][2] >= 90 and skip_white_count < 100:
                            skip_white_count += 1
                            continue
                        lab[i, j] = (Lab[0], Lab[1], Lab[2])
                        l_channel[i, j] = Lab[0]
                        l_sum += Lab[0]# 累加L分量
                l_mean = l_sum / (w * h - skip_count - skip_white_count)   # 計算平均值
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (10, 30) 
                font_scale = 1
                text_color = (255, 255, 255)  # 文本顏色，白色
                thickness = 2
                img_with_text = cv2.putText(ccm_image.copy(), f'L Mean: {l_mean:.2f}', org, font, font_scale, text_color, thickness, cv2.LINE_AA)


                # 直接保存Lab圖像
                if save_image:
                    LOGGER.info("Saving image...")
                    cv2.imwrite(str(save_path) + f'/{path}_result_1_channel_{b}.png', l_channel)
                    cv2.imwrite('result_lab.png', lab)
                    cv2.imwrite('result_with_text.png', img_with_text)
                
                if view_img:
                    LOGGER.info("Showing lab image...")
                    cv2.namedWindow('lab', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('lab', 960, 540)
                    cv2.imshow('lab', lab)
                    cv2.waitKey()
                    LOGGER.info("Showing image with text...")
                    cv2.namedWindow('img_with_text', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('img_with_text', 960, 540)
                    cv2.imshow('img_with_text', img_with_text)
                    LOGGER.info("Continue...")

                print("L分量的平均值：", l_mean)

                """ 油花檢測 """
                img = cutimg.copy()
                color = ImageEnhance.Color(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                output_color5 = color.enhance(3)
                img = np.array(output_color5)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_new = homomorphic_filter(img)
                print("new img shape is {}".format(img_new.shape))
                LOGGER.info("Image transformed successfully.")
                # cv2.imwrite("new.jpg", img_new)
                # Mask = GetArea(img.copy())
                if view_img:
                    Mask = img.copy()
                    LOGGER.info("Showing image...")
                    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('mask', 960, 540)
                    cv2.imshow('mask', Mask)
                    cv2.waitKey()
                    LOGGER.info("Continue...")
                if save_image:
                    LOGGER.info("Saving image...")
                    cv2.imwrite(str(save_path) + f"/{path}_cutting{b}.png", img)
                    cv2.imwrite(str(save_path) + f"/{path}_Homomorphic{b}.png", img_new)
                
                oil_img = img_new.copy()
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
                img_mask_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 找到黑色區域，將黑色區域標記為255，其餘區域標記為0
                _, black_area = cv2.threshold(img_mask_gray, 1, 255, cv2.THRESH_BINARY_INV)
                # 計算黑色區域的面積
                black_area_size = cv2.countNonZero(black_area)
                print("里肌肉面積:", black_area_size, " 像素")
                rate1 = white/(x*y)
                # rate2 = black/(x*y)
                print(f"{path} 油花占比:",round(rate1*100,2),'%')
                csv.write_data([path, black_area_size, round(rate1*100,2), conf, l_mean])
                #print("黑色占比:",round(rate2*100,2),'%')
                if save_image:
                    LOGGER.info("Saving image...")
                    cv2.imwrite(str(save_path) + f"/{path}_result{b}.png", oil_img)
                # cv2.imwrite("new_result.jpg", oil_img)
                LOGGER.info(f"Finish processing cutting image {b+1}/{len(cutimgs)}")
        LOGGER.info(f"Finish processing image {ii+1}/{len(images)}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,  help='輸入圖片路徑(必要)', required=True)
    parser.add_argument('--save_path', type=str, default=ROOT / 'results', help='儲存圖片路徑(預設為new_mask.jpg) ')
    parser.add_argument('--weight', type=str, default=ROOT / 'runs/train-seg/pork_detection_V2/weights/best.pt', help='模型路徑')
    parser.add_argument('--imgsz', type=tuple, default=(640, 640), help='推論大小(高, 寬)')
    parser.add_argument('--save_image', type=bool, default=True, help='是否儲存圖片')
    parser.add_argument('--augment', type=bool, default=False, help='是否增強')
    parser.add_argument('--half', type=bool, default=False, help='是否使用half')
    parser.add_argument('--dnn', type=bool, default=False, help='是否使用dnn')
    parser.add_argument('--con_threshold', type=float, default=0.5, help='信心閾值')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='iou閾值')
    parser.add_argument('--max_det', type=int, default=1000, help='最大檢測數')
    parser.add_argument('--save_crop', type=bool, default=False, help='是否儲存裁剪')
    parser.add_argument('--agnostic_nms', type=bool, default=False, help='是否使用agnostic nms')
    parser.add_argument('--visualize', type=bool, default=False, help='是否視覺化')
    parser.add_argument('--line_thickness', type=int, default=3, help='線條厚度')
    parser.add_argument('--hide_labels', type=bool, default=False, help='是否隱藏標籤')
    parser.add_argument('--hide_conf', type=bool, default=False, help='是否隱藏信心')
    parser.add_argument('--view_img', type=bool, default=False, help='是否顯示圖片')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))