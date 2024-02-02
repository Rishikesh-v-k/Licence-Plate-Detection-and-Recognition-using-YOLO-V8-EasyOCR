# Ultralytics Enhanced Speed Monitoring System 🚗💨, GPL-3.0 license

import hydra
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from datetime import datetime
import cv2

# Define constants for the system
DISTANCE_AB = 100.0 # Distance between Point A and Point B in meters
SPEED_LIMIT = 60 # Speed limit in km/h

class AverageSpeedCalculator(BasePredictor):

  def __init__(self, cfg, point, video_path):
    super().__init__(cfg)
    self.first_timestamps = {} # Dictionary to store the timestamp of each vehicle entering Point A or B
    self.timestamps = {'Point_A': {}, 'Point_B': {}} # Store timestamps for both points
    self.point = point
    self.video_path = video_path
    self.cap = cv2.VideoCapture(str(video_path))

  def get_annotator(self, img):
    return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

  def preprocess(self, img):
    img = torch.from_numpy(img).to(self.model.device)
    img = img.half() if self.model.fp16 else img.float() # uint8 to fp16/32
    img /= 255 # 0 - 255 to 0.0 - 1.0
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
      im = im[None] # expand for batch dim
    self.seen += 1
    im0 = im0.copy()
    if self.webcam: # batch_size >= 1
      log_string += f'{idx}: '
      frame = self.dataset.count
    else:
      frame = getattr(self.dataset, 'frame', 0)

    self.data_path = p
    self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
    log_string += '%gx%g ' % im.shape[2:] # print string
    self.annotator = self.get_annotator(im0)

    det = preds[idx]
    self.all_outputs.append(det)
    if len(det) == 0:
      return log_string

    for c in det[:, 5].unique():
      n = (det[:, 5] == c).sum() # detections per class
      log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

    # write
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] # normalization gain whwh
    for *xyxy, conf, cls in reversed(det):
      if self.args.save_txt: # Write to file
        xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() # normalized xywh
        line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh) # label format
        with open(f'{self.txt_path}.txt', 'a') as f:
          f.write(('%g ' * len(line)).rstrip() % line + '\n')

      if self.args.save or self.args.save_crop or self.args.show: # Add bbox to image
        c = int(cls) # integer class
        label = None if self.args.hide_labels else (
          self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
        speed = self.calculate_speed(xyxy)
        label += f', Speed: {speed:.2f} km/h'
        self.annotator.box_label(xyxy, label, color=colors(c, True))

      if self.args.save_crop:
        imc = im0.copy()
        save_one_box(xyxy,
               imc,
               file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
               BGR=True)

    return log_string

  def calculate_speed(self, coordinates):
    vehicle_id = hash(coordinates.tobytes())
    timestamp = datetime.now()

    if vehicle_id not in self.first_timestamps:
      self.first_timestamps[vehicle_id] = timestamp
      self.timestamps[self.point][vehicle_id] = timestamp
      return 0.0 # Speed is 0 until the vehicle enters Point A or B

    time_difference = (timestamp - self.timestamps[self.point][vehicle_id]).total_seconds() # in seconds
    average_speed = (DISTANCE_AB / 1000) / (time_difference / 3600) # converting meters to kilometers and seconds to hours

    return average_speed


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def calculate_average_speed(cfg):
  cfg.model = cfg.model or "yolov8n.pt"
  cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2) # check image size
  cfg.source_a = cfg.source_a if cfg.source_a is not None else ROOT / "assets" / "point_a_video.mp4"
  cfg.source_b = cfg.source_b if cfg.source_b is not None else ROOT / "assets" / "point_b_video.mp4"
   
  predictor_a = AverageSpeedCalculator(cfg, 'Point_A', cfg.source_a)
  predictor_b = AverageSpeedCalculator(cfg, 'Point_B', cfg.source_b)

  while True:
    # Process frames from Point A and Point B concurrently
    _, frame_a = predictor_a.cap.read()
    _, frame_b = predictor_b.cap.read()

    if frame_a is None and frame_b is None:
      break

    if frame_a is not None:
      predictor_a.write_results(idx=0, preds=predictor_a.postprocess([], frame_a, frame_a), batch=([], frame_a, frame_a))

    if frame_b is not None:
      predictor_b.write_results(idx=0, preds=predictor_b.postprocess([], frame_b, frame_b), batch=([], frame_b, frame_b))

  # Release video capture objects
  predictor_a.cap.release()
  predictor_b.cap.release()


if __name__ == "__main__":
  calculate_average_speed()