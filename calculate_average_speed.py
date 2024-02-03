import hydra
import torch
import easyocr
import cv2
import time
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# Global variables
DISTANCE_AB = 100  # Replace with the actual known distance between Point A and Point B

def get_ocr(im, coors):
    # Function to extract license plate using EasyOCR
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im = im[y:h, x:w]
    conf_threshold = 0.2

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    ocr = ""

    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) > 1 and len(result) > 6 and result[2] > conf_threshold:
            ocr = result[1]

    return str(ocr)

# Modified DetectionPredictor class
class SpeedCalculationPredictor(BasePredictor):

    def __init__(self, cfg, video_b):
        super().__init__(cfg)
        self.video_b = video_b
        self.timestamps_a = []
        self.timestamps_b = []

    def get_annotator(self, img):
        # Function to initialize annotator for visualization (if needed)
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        # Preprocessing function for each frame
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        # Postprocessing function to handle detections
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
        # Function to handle the output for each frame
        p, im, im0 = batch
        log_string = ""
        
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        det = preds[idx]

        # Matching and average speed calculation
        if len(det) == 0:
            return log_string

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # Matched vehicles processing
        matched_vehicles = []  # List to store information about matched vehicles

        for *xyxy, conf, cls in reversed(det):
            # License plate extraction
            ocr = get_ocr(im0, xyxy)

            if ocr != "":
                # Process matched vehicles
                if ocr not in matched_vehicles:
                    matched_vehicles.append(ocr)

                    # Measure time difference and calculate speed
                    if self.webcam:
                        timestamps = self.timestamps_a if idx == 0 else self.timestamps_b
                        timestamps.append(time.time())
                        if len(timestamps) > 1:
                            time_difference = timestamps[-1] - timestamps[-2]
                            average_speed = DISTANCE_AB / time_difference
                            print(f"Vehicle {ocr} Average Speed: {average_speed} units/time")
                            
        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def calculate_average_speed(cfg):
    # Function to calculate average speed
    cfg.model = cfg.model or "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/best.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)

    video_a_path = "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/point_a_video.mp4"  # Replace with actual path
    video_b_path = "/content/Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR/point_b_video.mp4"  # Replace with actual path

    predictor = SpeedCalculationPredictor(cfg, video_b_path)
    predictor.video = video_a_path
    predictor()

if __name__ == "__main__":
    # Set up EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Execute the system
    calculate_average_speed()
