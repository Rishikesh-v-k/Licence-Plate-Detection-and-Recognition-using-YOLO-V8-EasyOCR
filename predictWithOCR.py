# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import easyocr
import cv2
from omegaconf import DictConfig
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

def getOCR(im, coors):
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im = im[y:h, x:w]
    conf = 0.2

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    ocr = ""

    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) > 1 and len(result) > 6 and result[2] > conf:
            ocr = result[1]

    return str(ocr)

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        # Initialize CSV file for storing license plate information
        csv_filename = f"license_plate_info_{frame}.csv"
        csv_filepath = str(self.save_dir / csv_filename)
        with open(csv_filepath, 'w') as csv_file:
            csv_file.write("License Plate, Timestamp, Location\n")

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = None if self.args.hide_labels else (
                self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
            ocr = getOCR(im0, xyxy)
            if ocr != "":
                label = ocr

                # Append license plate information to CSV file
                with open(csv_filepath, 'a') as csv_file:
                    csv_file.write(f"{label}, {self.dataset.timestamp}, {self.dataset.location}\n")

            self.annotator.box_label(xyxy, label, color=colors(c, True))

        return log_string

@hydra.main(config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg: DictConfig):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"

    # Process two videos simultaneously
    video_paths = cfg.source

    for video_path in video_paths:
        cfg.source = video_path
        predictor = DetectionPredictor(cfg)
        predictor()

if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    # Add video paths as command line arguments
    predict()

