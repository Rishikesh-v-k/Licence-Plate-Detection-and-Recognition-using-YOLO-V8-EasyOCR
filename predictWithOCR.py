# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import easyocr
import cv2
import re
from omegaconf import DictConfig
from typing import List
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from datetime import datetime, timezone, timedelta
import csv

# Define the missing variable gn
gn = torch.tensor([1.], device='cuda:0')

def getOCR(im, coors):
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im_cropped = im[y:h, x:w]
    conf_threshold = 0.2

    # Use RGB image for OCR instead of converting to grayscale
    gray = cv2.cvtColor(im_cropped, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    ocr = ""

    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) > 1 and len(result) > 6 and result[2] > conf_threshold:
            ocr = result[1]

    # Clean the detected license plate by removing unwanted characters
    ocr = ''.join(char for char in ocr if char.isalnum())

    return ocr

def is_valid_license_plate(license_plate):
    # Define the regular expression pattern for Indian license plates
    pattern = r'^[A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,3}\s?[0-9]{4}$'

    # Check if the license plate string matches the pattern
    if re.match(pattern, license_plate):
        return True
    else:
        return False

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        # Apply temporal smoothing here if needed

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]

        self.seen += 1
        im0 = im0.copy()
        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string

        indian_timezone = timezone(timedelta(hours=5, minutes=30))
        timestamp = datetime.now(indian_timezone).strftime("%Y-%m-%d %H:%M:%S")

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:
                c = int(cls)
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                ocr = getOCR(im0, xyxy)
                if ocr != "":
                    label = ocr

                # Check if the detected license plate is valid for Indian format
                if label and is_valid_license_plate(label):
                    # Save license plate and timestamp in the CSV file
                    if label not in license_plates_timestamps:
                        license_plates_timestamps[label] = ["nill", "nill"]

                    if video_index == 0:
                        license_plates_timestamps[label][0] = timestamp
                    else:
                        license_plates_timestamps[label][1] = timestamp

                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string

@hydra.main(config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg: DictConfig):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"

    video_paths = cfg.source
    global video_index
    video_index = 0

    for video_path in video_paths:
        cfg.source = video_path
        predictor = DetectionPredictor(cfg)
        predictor()
        video_index += 1

    # Calculate the average speed and save to CSV
    csv_filename = "license_plate_info.csv"
    if cfg.project is None:
        csv_filepath = csv_filename
    else:
        csv_filepath = str(cfg.project / csv_filename)
    distance = 10  # Assuming a known distance of 10 km between the two points

    with open(csv_filepath, 'w', newline='') as csv_file:
        fieldnames = ["License Plate", "Timestamp1", "Timestamp2", "Speed(km/h)"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for license_plate, timestamps in license_plates_timestamps.items():
            timestamp1, timestamp2 = timestamps

            if timestamp1 != "nill" and timestamp2 != "nill":
                time1 = datetime.strptime(timestamp1, "%Y-%m-%d %H:%M:%S")
                time2 = datetime.strptime(timestamp2, "%Y-%m-%d %H:%M:%S")
                time_diff = time2 - time1
                time_diff_seconds = time_diff.total_seconds()
                speed = distance / (time_diff_seconds / 3600)  # Convert to km/h
                writer.writerow({"License Plate": license_plate, "Timestamp1": timestamp1, "Timestamp2": timestamp2, "Speed(km/h)": f"{speed:.2f}"})
            else:
                writer.writerow({"License Plate": license_plate, "Timestamp1": timestamp1, "Timestamp2": timestamp2, "Speed(km/h)": "nill"})

if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    license_plates_timestamps = {}  # Dictionary to store license plates and timestamps
    predict()  
