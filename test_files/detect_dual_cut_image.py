import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders_edit import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

def check_if_image_loaded_by_cv2(image):
    if not isinstance(image, np.ndarray):
        return False
    if image.ndim < 2:
        return False
    if image.ndim == 3 and image.shape[2] != 3:
        return False
    return True

def run_detect(image):
    if not check_if_image_loaded_by_cv2(image):
        LOGGER.error("No input image found.")
        return None
    # setting
    weights = ROOT / r'runs\train\yolov9-c10\weights\best.pt'  # model path or triton URL
    imgsz = (640, 640)  # inference size (height, width)
    conf_thres = 0.5  # confidence threshold
    xyxy_array, conf_array = run(weights=weights, source=image, imgsz=imgsz, conf_thres=conf_thres)
    xyxy_array = list(map(lambda x: list(map(int, x)), xyxy_array))
    print(xyxy_array)
    for xyxy in xyxy_array:
        # cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
        roi = image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        random_name = str(np.random.randint(0, 1000000))
        cv2.imshow("image", roi)
        cv2.imwrite(f"roi_{random_name}.png", roi)
        cv2.waitKey(0)
    
    
@smart_inference_mode()
def run(
        weights=ROOT / r'runs\train\yolov9-c10\weights\best.pt',  # model path or triton URL
        source=None,  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    if source is None:
        LOGGER.error("No input image found.")
        return None
    im0s = source.copy()
    path, s = "no_path", "image size:"
    # source = str(source)
    # save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    im = letterbox(im0s, imgsz, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    
    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, dt = 0, (Profile(), Profile(), Profile())
    # for path, im, im0s, vid_cap, s in dataset:
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    # Inference
    with dt[1]:
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        pred = pred[0][1]
    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        p, im0 = path, im0s.copy()
        p = Path(p)  # to Path
        # save_path = str(save_dir / p.name)  # im.jpg
        # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            xyxy_arr, conf_arr = [], []
            # Write results
            for *xyxy, conf, _ in reversed(det): # last is cls
                xyxy_arr.append(xyxy)
                conf_arr.append(conf)
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                #     with open(f'{txt_path}.txt', 'a') as f:
                #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                # if save_img or save_crop or view_img:  # Add bbox to image
                #     c = int(cls)  # integer class
                #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                #     annotator.box_label(xyxy, label, color=colors(c, True))
                # if save_crop:
                #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
        else:
            print("No object detected.")
            return None
        
        return xyxy_arr, conf_arr
        # Stream results
        # im0 = annotator.result()
        # if view_img:
        #     if platform.system() == 'Linux' and p not in windows:
        #         windows.append(p)
        #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        #     cv2.imshow(str(p), im0)
        #     cv2.waitKey(1)  # 1 millisecond
        # # Save results (image with detections)
        # if save_img:
        #     if dataset.mode == 'image':
        #         cv2.imwrite(save_path, im0)
        #     else:  # 'video' or 'stream'
        #         if vid_path[i] != save_path:  # new video
        #             vid_path[i] = save_path
        #             if isinstance(vid_writer[i], cv2.VideoWriter):
        #                 vid_writer[i].release()  # release previous video writer
        #             if vid_cap:  # video
        #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #             else:  # stream
        #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
        #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
        #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #         vid_writer[i].write(im0)
    # Print time (inference-only)
    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)




def main():
    # check_requirements(exclude=('tensorboard', 'thop'))
    input_image = cv2.imread(r'C:\Users\aiotl\Documents\pork_cut\dataset\train\images\image1.png')
    run_detect(image=input_image)

if __name__ == "__main__":
    main()
