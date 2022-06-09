import torch
import os

#Some adjustments
MAX_THROTTLE = 0.2 # m/s^2
TARGET_SPEED = 1 # m/s
CAMERA_FOV = 1.2 # 2.4 rad
CONES_LIM = 4
CONE_MIN = 2
CONES_RANGE_CUTOFF = 7 # meters
TIME_LIM_SEC = 300

# path settings
OUTPUT_VIDEO = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator/python/examples/yolo_camera/output.avi")
PATH_WITH_IMAGES = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator/python/examples/yolo_camera/cone_imgs/")

# model settings
PT_FILE = '/home/ivan/yolov5/runs/train/exp12/weights/best.pt'
MODEL = torch.hub.load('/home/ivan/yolov5', 'custom', path=PT_FILE, source='local')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FINAL_MODEL = MODEL.to(DEVICE)


