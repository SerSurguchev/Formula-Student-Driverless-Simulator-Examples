# Import libraries
import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import time
from PIL import Image
import torch
import config
from img_processing import (create_bitwise, 
                            brightness_change, 
                            cone_classification,
                            contour,
                            find_circle)

#Specify path
fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)

cones_lim = config.CONES_LIM
target_speed, max_throttle = config.TARGET_SPEED, config.MAX_THROTTLE

import fsds

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
            cones.append(dic)

        idx+=1
        
    return cones,im

def cn2point(cn):
    return (int(cn['x']),int(cn['y']))

def calculate_steering(cones,im, w,h):

    if len(cones) > 1:
        x_span=abs(cones[-1]['x'] - cones[-2]['x'])
        tx=(cones[-1]['x'] + cones[-2]['x'])/2
        ty=(cones[-1]['y'] + cones[-2]['y'])/2
        return x_span, config.CAMERA_FOV * (tx-w/2)/(h-ty)
        
    else:
        return 0, config.CAMERA_FOV * (cones[0]['x']-w/2)/(h-cones[0]['y'])

def calculate_throttle():

    global target_speed, max_throttle
    gps = client.getGpsData()
    # Calculate the velocity in the vehicle's frame
    velocity = math.sqrt(math.pow(gps.gnss.velocity.x_val, 2) + math.pow(gps.gnss.velocity.y_val, 2))

    # the lower the velocity, the more throttle, up to max_throttle
    return max_throttle * max(1 - velocity / target_speed, 0)

    
def sldim(arr, size, handler):
    return [handler(arr[i:i + size]) for i in range(0, len(arr), size)]
       
#====================================MAIN====================================

#Connect with simulator
client = fsds.FSDSClient(ip="192.168.137.1")
client.confirmConnection()
client.enableApiControl(True)

t2=t1=time.time()
idx=0

model = config.FINAL_MODEL
print('Model is on CUDA check: ', next(model.parameters()).is_cuda)

image_index = 0
line_begin = line_end = (0, 0)

while (t2 - t1) < config.TIME_LIM_SEC:
    t2 = time.time()

    [image] = client.simGetImages([fsds.ImageRequest(camera_name = 'cam1', image_type = fsds.ImageType.Scene, pixels_as_float = False, compress = False)], vehicle_name = 'FSCar')
            
    im=np.array(sldim(sldim(image.image_data_uint8, 3, lambda p: int((p[0]+p[1]+p[2])/3)), image.width, lambda l: l), dtype=np.uint8)
    
    im_copy = im.copy()
    
    if image_index == 1:  
        contour_im = contour(im.copy())
        line_begin, line_end = find_circle(contour_im.copy())

    cv2.line(im_copy, line_begin, line_end, (0), 1)
    
    results = model(Image.fromarray(im), size=785)

    df = results.pandas().xyxy[0]
      
    # classify cones
    bboxes = df.values.tolist()
    
    txt_file = open(config.PATH_WITH_IMAGES + f'im_{image_index}.txt' , 'w')

    for ind, bbox in enumerate(bboxes):
    
        cone_class = -1
    

        # define bbox parameters
        xmin, ymin, xmax, ymax = list(map(int, bbox[:4]))
        center_x = round((xmin + xmax) / 2)
        center_y = round((ymin + ymax) / 2)
        
        w_bbox = xmax - xmin
        h_bbox = ymax - ymin
         
        x_norm = round((float(xmin) + float(w_bbox) / 2) / float(im.shape[1]), 6)
        y_norm = round((float(ymin) + float(h_bbox) / 2) / float(im.shape[0]), 6)
        h_norm = round(float(h_bbox) / float(im.shape[1]), 6)
        w_norm = round(float(w_bbox) / float(im.shape[0]), 6)
    
        if (ymax - ymin) < 15:
            continue
                     
        if line_begin[1] > center_y or image_index == 0:
            bitwise_cone = create_bitwise(im.copy(), xmin, ymin, xmax, ymax)
            cone_class = cone_classification(bitwise_cone.copy())
            txt_file.write(f"{cone_class} {x_norm} {y_norm} {w_norm} {h_norm}\n")
        
        if cone_class != -1:
            if cone_class == 0:
                cv2.putText(im_copy, 'blue', (center_x, ymin + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

            elif cone_class == 1:
                cv2.putText(im_copy, 'orange', (center_x, ymin + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                
            elif cone_class == 4:
                cv2.putText(im_copy, 'big', (center_x, ymin + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        else:
            cv2.rectangle(im_copy, (xmin, ymin), (xmax, ymax), (0), 2)

#  write video and car autonomous driving    
    
    cones, im = cones_coords(df,im.copy())
    
    if not 'video' in globals():
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(config.OUTPUT_VIDEO, fourcc, 10, (im_copy.shape[1], im_copy.shape[0]), 0)

    if len(cones) != 0:
        car_controls = fsds.CarControls()
        span_max,s_ = calculate_steering(cones,im.copy(),image.width,image.height)
        while True:
            cones.pop()
            if len(cones) <= config.CONE_MIN:
                break            
            span,s =calculate_steering(cones,im.copy(),image.width,image.height)
            if abs(span)>abs(span_max):
                s_=s
                span_max=span

        car_controls.steering = s_       
        
        car_controls.throttle = calculate_throttle()
        car_controls.brake = 0
        client.setCarControls(car_controls)

    video.write(im_copy)
    cv2.imwrite(config.PATH_WITH_IMAGES + 'im_' + str(image_index) + '.jpg', im)
    
    txt_file.close()
    
    print(f"File im_{image_index}.jpg was written in {config.PATH_WITH_IMAGES}im_{image_index}.jpg")
    print(f"File im_{image_index}.txt was written in {config.PATH_WITH_IMAGES}im_{image_index}.txt")
    image_index += 1
    idx += 1

video.release()

car_controls = fsds.CarControls()
car_controls.throttle = 0.0
car_controls.brake = 0.1
client.reset()
client.setCarControls(car_controls)

print(idx/(t2 - t1))
