import torch
import numpy as np
import os

# Some adjustments
MAX_THROTTLE = 0.02 # m/s^2
TARGET_SPEED = 5 # m/s
CAMERA_FOV = 1.2 # 2.4 rad
CONES_LIM = 4
CONE_MIN = 2
CONES_RANGE_CUTOFF = 7 # meters
TIME_LIM_SEC = 60
CONE_MIN = 4
SPEED_STEP = 0.1
SAME_CONE_BOUND = 1.0
CONE_SEEN_CNT = 10
CAMCALIB_TRIG = 10

# Path settings
OUTPUT_VIDEO = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator/python/examples/yolo_camera/output.avi")
PATH_WITH_IMAGES = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator/python/examples/yolo_camera/cone_imgs")

# model settings
# WEIGHT_FILE = '/home/ivan/yolov5_new/runs/nano_all_imgs/weights/best.engine'
# WEIGHT_FILE = '/home/ivan/yolov5_new/runs/nano_all_imgs/weights/best_int8.engine'
WEIGHT_FILE = '/home/ivan/yolov5_new/runs/nano_all_imgs/weights/best.pt'
MODEL = torch.hub.load('/home/ivan/yolov5_new', 'custom', path=WEIGHT_FILE, source='local')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FINAL_MODEL = MODEL.to(DEVICE)

# Intrinsic camera parameters
fx=fy=392.5
cx=392.5
cy=196.
INTRINSICS = np.linalg.inv(np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float64))

#Distortion
r2=0
r4=0
t1=0
t2=0
r6=0
