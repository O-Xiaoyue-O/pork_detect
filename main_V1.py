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
            f.write('Filename,Area size,Proportion,conf\n')
        
    def write_data(self, data: list):
        assert isinstance(data, list), "data must be a list"
        assert len(data) == 4, "data must have 3 elements"
        with open(os.path.join(self.path, self.name), 'a') as f:
            f.write(f"{data[0]},{data[1]},{data[2]},{data[3]}\n")


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
    print(percentage_dark_red, percentage_dark_red > threshold)
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
    for i, (im0, path) in enumerate(zip(images, paths)):
        LOGGER.info("Processing image")
        # 預測並擷取圖片，回傳xyxy_arr, conf_arr, cutimgs，因為設計上是可以多張進行預測，所以回傳資料格式如下:
        """
        results = [[xyxy_arr, conf_arr, cutimgs], [xyxy_arr, conf_arr, cutimgs], ...]   
        """
        results = detecter.process(im0, path)
        # spilt file name from path
        path = Path(path).stem
        # xyxy_arr, conf_arr, cutimgs = result
        for result in results:
            # 將result解包
            """
            xyxy_arr: 檢測框座標，格式為[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
            conf_arr: 信心度，格式為[conf1, conf2, ...]
            cuttings: 輸出的裁剪圖片，格式為cv2格式
            """
            xyxy_arr, conf_arr, cutimgs = result
            LOGGER.info(f"Captured {len(cutimgs)} cutting images.")
            for x, (xyxy, conf, cutimg) in enumerate(zip(xyxy_arr, conf_arr, cutimgs)):
                LOGGER.info("Transforming image to Homomorphic filter...")
                # img = np.array(cutimg)
                # 其餘照"同態濾波.py"，將部分搬過來
                img = cutimg.copy()
                # print(im0, img.shape)
                # convert cv2 to PIL
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                while not is_dark_red(img):
                    # Adjust brightness
                    enhancer = ImageEnhance.Brightness(img)
                    factor = 0.975  # Decrease brightness by 97.5%
                    img = enhancer.enhance(factor)
                # convert PIL to cv2
                img = np.array(img)
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
                    cv2.imwrite(str(save_path) + f"/{path}_cutting{i}.png", cutimg)
                    cv2.imwrite(str(save_path) + f"/{path}_Homomorphic{i}.png", img_new)
                
                # Mask = cv2.cvtColor(Mask, cv2.COLOR_RGB2BGR)
                # img_mask = cv2.bitwise_and(img, img, mask=Mask[:,:,2])
                # oil_img = img_mask.copy()
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
                ret, black_area = cv2.threshold(img_mask_gray, 1, 255, cv2.THRESH_BINARY_INV)

                # 計算黑色區域的面積
                black_area_size = cv2.countNonZero(black_area)

                print("里肌肉面積:", black_area_size, " 像素")

                rate1 = white/(x*y)
                # rate2 = black/(x*y)
                print(f"{path} 油花占比:",round(rate1*100,2),'%')
                csv.write_data([path, black_area_size, round(rate1*100,2), conf])
                #print("黑色占比:",round(rate2*100,2),'%')
                if save_image:
                    LOGGER.info("Saving image...")
                    cv2.imwrite(str(save_path) + f"/{path}_result{i}.png", oil_img)
                # cv2.imwrite("new_result.jpg", oil_img)
                LOGGER.info(f"Finish processing cutting image {x+1}/{len(cutimgs)}")
        LOGGER.info(f"Finish processing image {i+1}/{len(images)}")

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