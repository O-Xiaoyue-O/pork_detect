import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

def check_if_image_loaded_by_cv2(image):
    if not isinstance(image, np.ndarray):
        return False
    if image.ndim < 2:
        return False
    if image.ndim == 3 and image.shape[2] != 3:
        return False
    if image is None:
        return False
    return True

@smart_inference_mode()
class segmenter_detection_cut:
    def __init__(self,
        weights=ROOT / 'runs/train-seg/gelan-c-seg7/weights/best.pt',  # model.pt path(s)# source=r'C:\Users\aiotl\Documents\pork_cut\yolo_dataset\YOLODataset_seg\images\train',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_img=True,  # save results to *.png
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        # self.source = str(source)
        self.save_img = save_img
        self.augment = augment

        # Load model
        device = select_device(device)
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        # dataset = LoadImages(self.source, img_size=self.imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
        self.seen, self.dt = 0, (Profile(), Profile(), Profile())
        self.count = 0


    def process(self, 
            im0, 
            path=None,
            conf_thres=0.4,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            save_crop=False,  # save cropped prediction boxes
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            visualize=False,  # visualize features
            line_thickness=3,  # bounding box thickness (pixels)
            project=ROOT / 'runs/pork_cut-seg',  # save results to project/name
            name='pork_cut',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            retina_masks=False,
        ):
        # Directories
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'crops').mkdir(parents=True, exist_ok=True)  # make dir
        
        if not check_if_image_loaded_by_cv2(im0):
            raise ValueError("The image is not loaded by cv2.imread() or not loaded.")
        self.count += 1
        path = f"image.png" if path is None else path
        s = f"image: {self.count}"
        
        im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = self.model(im, augment=self.augment, visualize=visualize)[:2]
        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions
        result_array = []
        for i, det in enumerate(pred):  # per image
            self.seen += 1
            p, im0 = path, im0.copy()
            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # im.jpga
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
            if len(det):
                masks = process_mask(proto[-1][i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True) # HWC  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Mask cutting
                # print(masks)
                maskeds = []
                for num, mask in enumerate(masks):
                    mask_raw = mask.cpu().data.numpy()
                    mask_3channel = cv2.merge((mask_raw, mask_raw, mask_raw))
                    h2, w2, _ = im0.shape
                    mask = cv2.resize(mask_3channel, (w2, h2))
                    
                    lower_black = np.array([0,0,0])
                    upper_black = np.array([0,0,1])
                    mask = cv2.inRange(mask, lower_black, upper_black)
                    mask = cv2.bitwise_not(mask)
                    
                    kernel_size = 50  # 腐蝕大小
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask = cv2.erode(mask, kernel, iterations=1)
                    
                    masked = cv2.bitwise_and(im0, im0, mask=mask)
                    
                    # 去除多餘的黑邊
                    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    larget_contours = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(larget_contours)
                    cropped_image = masked[y:y+h, x:x+w]
                    # save result to array
                    maskeds.append(cropped_image)
                    if self.save_img:
                        save_cut_path = str(self.save_dir / 'crops' / p.stem) + f"_cut{num + 1}.png"  # im.jpg
                        cv2.imwrite(save_cut_path, cropped_image)
                
                # Mask plotting
                annotator.masks(masks,
                                colors=[colors(x, True) for x in det[:, 5]],
                                im_gpu=None if retina_masks else im[i])
                # save array
                xyxy_arr, conf_arr = [], []
                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    # save to array
                    xyxy_arr.append(xyxy), conf_arr.append(conf)
                    if self.save_img or save_crop:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                    if save_crop:
                        save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)
                        
                # save result to array
                result_array.append([xyxy_arr, conf_arr, maskeds])
            
            # Stream results
            im0 = annotator.result()
            # Save results (image with detections)
            if self.save_img:
                cv2.imwrite(save_path, im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.save_img:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        return result_array


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'C:\Users\aiotl\Documents\pork_cut\yolov9\runs\train-seg\gelan-c-seg7\weights\best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=r'C:\Users\aiotl\Documents\pork_cut\yolo_dataset\YOLODataset_seg\images\train', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


# def main(opt):
#     check_requirements(exclude=('tensorboard', 'thop'))
#     pre_load_model(**vars(opt))


if __name__ == "__main__":
    LOGGER.error("This script is not meant to be run directly. Pass")
