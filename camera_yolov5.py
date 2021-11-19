# Result : https://github.com/SerSurguchev/Formula-Student-Driverless-Simulator-Examples/blob/main/result.mp4
# Import libraries
import sys
import os
import numpy 
import math
import matplotlib.pyplot as plt
import cv2
import time
from PIL import Image
import torch

#Some adjustments
max_throttle = 0.2 # m/s^2
target_speed = 1 # m/s
max_steering = 0.2
k_steering = 0.1

camera_fov = 1.2 # 2.4 rad

cones_lim=4
cone_min=2
cones_range_cutoff = 7 # meters
cones_size_cutoff_px = 20

time_lim_sec = 400.0

#Specify path
fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)

import fsds

image_folder = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator/python/examples/yolo_camera")
output_video = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator/python/examples/yolo_camera/video.avi")

def pointgroup_to_cone(group):
    average_x = 0
    average_y = 0
    for point in group:
        average_x += point['x']
        average_y += point['y']
    average_x = average_x / len(group)
    average_y = average_y / len(group)
    return {'x': average_x, 'y': average_y}

def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(abs(x1-x2), 2) + math.pow(abs(y1-y2), 2))

def cones_coords(df,im):
    global cones_lim

    cones = []
    
    df = df.drop(['confidence', 'class', 'name'], axis = 1)         
    sort_df = df.sort_values(by = ['ymax'], ascending = False) 

    idx=0
    for index, row  in sort_df.iterrows():
        dic = {}
        
        dic['x'] = (row['xmin'] + row['xmax'])/2
        dic['y'] =  row['ymax']

        if idx<cones_lim:
            cv2.circle(im, (int(dic['x']), int(dic['y'])) , 10, (0), 2)
            cones.append(dic)
        else:
            cv2.circle(im, (int(dic['x']), int(dic['y'])) , 10, (255), 2)

        idx+=1
        
    return cones,im

def cn2point(cn):
    return (int(cn['x']),int(cn['y']))

def calculate_steering(cones,im, w,h):
    if len(cones) > 1:
        cv2.line(im, cn2point(cones[-1]), cn2point(cones[-2]), (0), 2)
        x_span=abs(cones[-1]['x'] - cones[-2]['x'])
        tx=(cones[-1]['x'] + cones[-2]['x'])/2
        ty=(cones[-1]['y'] + cones[-2]['y'])/2
        cv2.line(im, (int(tx),int(ty)), (int(w/2),int(h)), (255), 2)
        return x_span,camera_fov*(tx-w/2)/(h-ty),im
    else:
        cv2.line(im, cn2point(cones[0]), (int(w/2),int(h)), (255), 2)
        return 0, camera_fov*(cones[0]['x']-w/2)/(h-cones[0]['y']),im

def calculate_throttle():
    global target_speed, max_throttle
    gps = client.getGpsData()
    # Calculate the velocity in the vehicle's frame
    velocity = math.sqrt(math.pow(gps.gnss.velocity.x_val, 2) + math.pow(gps.gnss.velocity.y_val, 2))

    # the lower the velocity, the more throttle, up to max_throttle
    return max_throttle * max(1 - velocity / target_speed, 0)

def fromOneDtoTwoD(arr, size):
    return [arr[i:i + size] for i in range(0, len(arr), size)]

#====================================MAIN====================================


#Connect with simulator
client = fsds.FSDSClient(ip="192.168.137.1")
client.confirmConnection()
client.enableApiControl(True)

t2=t1=time.time()
idx=0

# Model
pt_file = '.pt weight file path'
model = torch.hub.load('/home/user/yolov5', 'custom', path=pt_file, source='local')  # local repo

while (t2 - t1) < time_lim_sec:
    t2 = time.time()

    [image] = client.simGetImages([fsds.ImageRequest(camera_name = 'cam1', image_type = fsds.ImageType.Scene, pixels_as_float = False, compress = False)], vehicle_name = 'FSCar')        
#        print(len(image.image_data_uint8))
    Twod = fromOneDtoTwoD(image.image_data_uint8, 3)
#        print(len(Twod))

    im=numpy.zeros((image.width,image.height),dtype=numpy.uint8)
    
#        print(len(Twod)/image.width)
    x=0
    y=0
    for p in Twod:
        im[y][x]=(p[0]+p[1]+p[2])/3
        x+=1
        if x>=image.height:
            x=0
            y+=1

    results = model(Image.fromarray(im), size=785)

    df = results.pandas().xyxy[0]
    cones,im=cones_coords(df,im)
    
    if not 'video' in globals():
        video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (image.width,image.height),0)

    if len(cones) != 0:
        car_controls = fsds.CarControls()
        span_max,s_,i_ = calculate_steering(cones,im.copy(),image.width,image.height)
        while True:
            cones.pop()
            if len(cones)<=cone_min:
                break            
            span,s,i=calculate_steering(cones,im.copy(),image.width,image.height)
            if abs(span)>abs(span_max):
                s_=s
                i_=i
                span_max=span

        car_controls.steering = s_
        im=i_
        
        car_controls.throttle = calculate_throttle()
        car_controls.brake = 0
        client.setCarControls(car_controls)

    video.write(im)
    idx+=1

video.release()

car_controls = fsds.CarControls()
car_controls.throttle = 0.0
car_controls.brake = 0.1
client.setCarControls(car_controls)

print(idx/(t2 - t1))
