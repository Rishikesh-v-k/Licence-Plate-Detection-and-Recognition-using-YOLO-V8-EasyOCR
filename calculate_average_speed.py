# Modified Ultralytics YOLO with EasyOCR for Number Plate Recognition 🚀, GPL-3.0 license

import hydra
import torch
import easyocr
import cv2
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import pandas as pd
import time

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
        if len(results) > 1 and len(results[1]) > 6 and results[2] > conf:
            ocr = result[1]

    return str(ocr)

class DetectionPredictor(BasePredictor):
    def __init__(self, cfg, output_csv):
        super(DetectionPredictor, self).__init__(cfg)
        self.output_csv = output_csv

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img, timestamp):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds, timestamp

    def write_results(self, idx, preds, batch, timestamp):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string

        timestamp_ms = int(time.time() * 1000)
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        results = []
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = None if self.args.hide_labels else (
                self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
            ocr = getOCR(im0, xyxy)
            if ocr != "":
                label = ocr
                results.append({'label': label, 'timestamp': timestamp_ms})

            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        if len(results) > 0:
            df = pd.DataFrame(results)
            df.to_csv(self.output_csv, mode='a', header=False, index=False)

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/best.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = [
        '/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/point_a_video.mp4',
        '/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/point_b_video.mp4'
    ]
    output_csv = 'number_plate_results.csv'
    
    with open(output_csv, 'w') as f:
        f.write("label,timestamp\n")  # Write CSV header
    
    predictor1 = DetectionPredictor(cfg, output_csv)
    predictor2 = DetectionPredictor(cfg, output_csv)
    
    predictor1()
    predictor2()

if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    predict()