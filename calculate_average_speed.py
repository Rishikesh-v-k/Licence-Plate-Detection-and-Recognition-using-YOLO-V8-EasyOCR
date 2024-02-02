# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import easyocr
import cv2
from datetime import datetime
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors

# Use 'best.pt' as the model file
BEST_MODEL_PATH = "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/best.pt"

# Define the paths for the video samples
POINT_A_VIDEO_PATH = "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/point_a_video.mp4"
POINT_B_VIDEO_PATH = "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/point_b_video.mp4"

# Missing Functions - Define getOCR and save_one_box
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
        if len(results) > 1 and len(result[1]) > 6 and result[2] > conf:
            ocr = result[1]

    return str(ocr)

def save_one_box(xyxy, im, file, BGR=True):
    xyxy = [int(xy) for xy in xyxy]
    im = im[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    if BGR:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(file), im)

class SpeedCalculation:
    def __init__(self, distance):
        self.distance = distance
        self.vehicle_records = {}

    def record_entry(self, plate, point):
        now = datetime.now()
        self.vehicle_records[plate] = {'entry_time': now, 'entry_point': point}

    def record_exit(self, plate, point):
        if plate in self.vehicle_records:
            self.vehicle_records[plate]['exit_time'] = datetime.now()
            self.vehicle_records[plate]['exit_point'] = point

    def calculate_speed(self, plate):
        if plate in self.vehicle_records and 'exit_time' in self.vehicle_records[plate]:
            entry_time = self.vehicle_records[plate]['entry_time']
            exit_time = self.vehicle_records[plate]['exit_time']
            elapsed_time = (exit_time - entry_time).total_seconds()

            if elapsed_time > 0:
                average_speed = self.distance / elapsed_time
                entry_point = self.vehicle_records[plate]['entry_point']
                exit_point = self.vehicle_records[plate]['exit_point']
                return {
                    'plate': plate,
                    'average_speed': average_speed,
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_point': entry_point,
                    'exit_point': exit_point
                }
        return None

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

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                ocr = getOCR(im0, xyxy)

                if ocr != "":
                    label = ocr
                    # Record entry time for the vehicle with the recognized license plate
                    speed_calculator.record_entry(ocr, point='Point A')

                self.annotator.box_label(xyxy, label, color=colors(c, True))

            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string

# Instantiate SpeedCalculation class with the known distance between points A and B
DISTANCE_AB = 100  # Replace with the actual distance in meters
speed_calculator = SpeedCalculation(DISTANCE_AB)

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = BEST_MODEL_PATH  # Use the 'best.pt' model
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = [POINT_A_VIDEO_PATH, POINT_B_VIDEO_PATH]  # List of video sources
    predictor = DetectionPredictor(cfg)
    predictor()

    # After processing, calculate and print average speeds
    for plate in speed_calculator.vehicle_records.keys():
        average_speed_info = speed_calculator.calculate_speed(plate)
        if average_speed_info is not None:
            print(f"Vehicle with license plate {average_speed_info['plate']}:\n"
                  f" - Average Speed: {average_speed_info['average_speed']} m/s\n"
                  f" - Entry Time: {average_speed_info['entry_time']}\n"
                  f" - Exit Time: {average_speed_info['exit_time']}\n"
                  f" - Entry Point: {average_speed_info['entry_point']}\n"
                  f" - Exit Point: {average_speed_info['exit_point']}\n")

if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    predict()
