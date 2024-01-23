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
import threading
import utm

# Adds the fsds package located the parent directory to the python path
fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)

import sba
import fsds
from fsds.types import Quaternionr
from fsds.types import Vector3r


def get_cone_dXYZq(u, v, Rq, iA=config.INTRINSICS):
    xy1_c = iA.dot(np.array([[u, v, 1]], dtype=np.float64).T)
    return Vector3r(xy1_c[0, 0], xy1_c[1, 0], xy1_c[2, 0]).to_Quaternionr().rotate(Rq)


def reproj_cone_Z(t, dXYZq, Z):
    s = -(t.z_val - Z) / dXYZq.z_val
    return Vector3r(t.x_val + s * dXYZq.x_val, t.y_val + s * dXYZq.y_val, s)


def get_A(Rq, t):
    A = np.zeros(6, dtype=np.float64)
    iRq = Rq.inverse()

    if iRq.w_val > 0:
        A[-6:-2] = iRq.to_numpy_array()
    else:
        A[-6:-2] = -iRq.to_numpy_array()

    A[-3:] = -t.rotate(iRq).to_numpy_array()[0:3]
    return tuple(A)


def cn2point(cn):
    return (int(cn['x']), int(cn['y']))


def calculate_throttle(velocity, steering,
                       max_throttle=config.MAX_THROTTLE,
                       speed_step=config.SPEED_STEP,
                       target_speed=config.TARGET_SPEED):
    global filter_speed
    filter_speed = min(filter_speed, max(1 - abs(steering), 0) * target_speed)
    filter_speed = filter_speed + min(target_speed - filter_speed, speed_step)
    # The lower the velocity, the more throttle, up to max_throttle
    return max_throttle * max(1 - velocity / filter_speed, 0)


def sldim(arr, size, handler):
    return [handler(arr[i:i + size]) for i in range(0, len(arr), size)]


def cn2p(cn):
    return (cn['u'], cn['v'])


def calculate_steering(cones, im, w, h):
    if len(cones) > 1:
        cv2.line(im, cn2p(cones[-1]), cn2p(cones[-2]), (0), 2)
        x_span = abs(cones[-1]['u'] - cones[-2]['u'])
        tx = (cones[-1]['u'] + cones[-2]['u']) / 2
        ty = (cones[-1]['v'] + cones[-2]['v']) / 2
        cv2.line(im, (int(tx), int(ty)), (int(w / 2), int(h)), (255), 2)
        return x_span, config.CAMERA_FOV * (tx - w / 2) / (h - ty), im
    else:
        cv2.line(im, cn2p(cones[0]), (int(w / 2), int(h)), (255), 2)
        return 0, config.CAMERA_FOV * (cones[0]['u'] - w / 2) / (h - cones[0]['v']), im


def visual_thread(im, results, model, cones_lim=config.CONES_LIM):
    r = model(Image.fromarray(im), size=len(im))
    df = r.pandas().xyxy[0]

    # TODO: cone size filter should be used!

    df = df.drop(['confidence', 'class', 'name'], axis=1)
    # df['ratio'] = (df.ymax - df.ymin) / (df.xmax - df.xmin)
    # sort_df = df.sort_values(by=['ymax'], ascending=False)

    df['square'] = ((df.xmax - df.xmin)*(df.ymax - df.ymin))
    sort_df = df.sort_values(by = ['square'], ascending = False)

    cones = []
    for index, row in sort_df.iterrows():

        # Ignore intersections, too close detections
        while index > 0:
            index -= 1
            if (sort_df.ymax[index] + sort_df.ymin[index] - row['ymax'] - row['ymin']) ** 2 + \
                    (sort_df.xmax[index] + sort_df.xmin[index] - row['xmax'] - row['xmin']) ** 2 < \
                    (sort_df.xmax[index] - sort_df.xmin[index] + row['xmax'] - row['xmin']) ** 2:
                break

        if index > 0:
            continue

        dic = {'u': int((row['xmin'] + row['xmax']) / 2), 'v': int(row['ymax'])}

        # Check for cone width and height reduce monothonic accordingly to screen position ???
        if len(cones) < cones_lim:
            cones.append(dic)
            cv2.circle(im, cn2p(dic), int((row['xmax'] - row['xmin']) / 2), (0), 2)
        else:
            cv2.circle(im, cn2p(dic), int((row['xmax'] - row['xmin']) / 2), (255), 2)

    results[:] = cones, im


def loop_closure(current_point, start_radius=3.0):
    global start_point, loop_fsm
    if not 'start_point' in globals():
        start_point = current_point
        loop_fsm = 3
        return False

    if loop_fsm == 3:
        if (start_point - current_point).get_xy_length() > start_radius:
            loop_fsm = 2
    if loop_fsm == 2:
        if (start_point - current_point).get_xy_length() < start_radius:
            loop_fsm = 1
    if loop_fsm == 1:
        if (start_point - current_point).get_xy_length() > start_radius:
            loop_fsm = 0
            return True
    if loop_fsm == 0:
        return True
    return False


def imgpnt_scale(val, cen, scl):
    return scl * (val - cen) + cen


def mappnt2imgpnt(p, sz):
    global vehicle_path

    px = sz * (p['x'] - vehicle_path.min_x) / (vehicle_path.max_x - vehicle_path.min_x)
    py = sz * (p['y'] - vehicle_path.min_y) / (vehicle_path.max_y - vehicle_path.min_y)

    if vehicle_path.max_x - vehicle_path.min_x > vehicle_path.max_y - vehicle_path.min_y:
        py = imgpnt_scale(py, sz / 2,
                          (vehicle_path.max_y - vehicle_path.min_y) / (vehicle_path.max_x - vehicle_path.min_x))
    else:
        px = imgpnt_scale(px, sz / 2,
                          (vehicle_path.max_x - vehicle_path.min_x) / (vehicle_path.max_y - vehicle_path.min_y))

    return (int(px), int(py))


def get_car_imgmap(sz, o, vto, cone_seen_cnt=config.CONE_SEEN_CNT):
    global vehicle_path, cm
    im = np.zeros((sz, sz), dtype=np.uint8)
    for c in cm.cones:
        if len(c.fuv) > cone_seen_cnt:
            cv2.circle(im, mappnt2imgpnt({'x': c.ep.x_val, 'y': c.ep.y_val}, sz), radius=0, color=(255), thickness=-1)

    for p in vehicle_path.points:
        p1 = mappnt2imgpnt(p, sz)
        if vto is not None:
            if 'p0' in locals():
                cv2.line(im, p0, p1, (255), 1)
        p0 = p1

    cv2.circle(im, p0, 5, (255), 2)
    cv2.line(im, p0, (list(p0)[0] + int(10 * o.x_val), list(p0)[1] + int(10 * o.y_val)), (255), 2)
    if vto is not None:
        cv2.line(im, p0, (list(p0)[0] + int(vto.x_val), list(p0)[1] + int(vto.y_val)), (255), 1)
    return im

def get_objpoints_and_imgpoints():
    global c_eps_ratio, cm

    objpoints = []
    imgpoints = []
    for c in cm.cones:
        for c_fuv in c.fuv:
            if c.eps < c_eps_ratio * cm.eps:
                A = cm.frames[c_fuv[0]]
                iRq = Vector3r(A[0], A[1], A[2]).to_Quaternionr()
                iRq.calc_wval()

                ep = c.ep.to_Quaternionr().rotate(iRq) + Quaternionr(A[-3], A[-2], A[-1], 0)
                objpoints.append([ep.x_val, ep.y_val, ep.z_val])
                imgpoints.append(c_fuv[1:3])
    return (np.array(objpoints, dtype=np.float32), np.array(imgpoints, dtype=np.float32))


def calibrate_camera(cx, cy, fx, fy):
    cr = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e2)
    flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS +
            cv2.CALIB_FIX_PRINCIPAL_POINT +
            cv2.CALIB_FIX_ASPECT_RATIO +
            cv2.CALIB_ZERO_TANGENT_DIST
    )
    objp, imgp = get_objpoints_and_imgpoints()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [imgp], (int(2 * cx), int(2 * cy)),
                                                       np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
                                                                dtype=np.float64), None,
                                                       flags=flags,
                                                       criteria=cr)

class Gps_nav:
    def __init__(self, p):
        self.ix, self.iy, *_ = utm.from_latlon(p.latitude, p.longitude)

    def get_nav(self, p):
        cx, cy, *_ = utm.from_latlon(p.latitude, p.longitude)
        return cx - self.ix, cy - self.iy


# vehicle path
class Vehicle_path:
    points = []
    max_x = 0
    max_y = 0
    min_x = 0
    min_y = 0
    border = 1

    def __init__(self, border):
        self.border = border

    def add_point(self, p):
        self.max_x = max(p['x'] + self.border, self.max_x)
        self.max_y = max(p['y'] + self.border, self.max_y)
        self.min_x = min(p['x'] - self.border, self.min_x)
        self.min_y = min(p['y'] - self.border, self.min_y)

        self.points.append(p.copy())

    def get_vector_to(self, i, x, y):
        p = self.points[i]
        return Vector3r(p['x'] - x, p['y'] - y, 0)


class Cone:
    def __init__(self, frame_idx, cam_u, cam_v, pZ0):
        # Each cone record contains:
        # U, V - 2D projection coordinates
        # sgm - standard deviation
        # det - number of detections
        self.fuv = [(frame_idx, cam_u, cam_v)]  # list of (frame_idx, cam_u, cam_v) list of all pixel values for specific cone index on all images
        self.lp = pZ0  # lp - Vector3r() last XYZ world cone point with Z=0 (used for 1 step)
        self.ep = Vector3r(pZ0.x_val, pZ0.y_val, 0)  # ep - Vector3r() estimated XYZ world cone point
        self.ncams = 1  # number of cameras used when the estimate(ep) was done
        self.eps = np.inf  # L2 norm BA error in pixels

    # Make sure there are only one detected  cone on every frame
    def assign_cone(self, frame_idx, cam_u, cam_v, pZ0):
        if frame_idx > self.fuv[-1][0]:
            self.fuv.append((frame_idx, cam_u, cam_v))
            self.lp = pZ0
            return True
        else:
            return False


# ------------------------------------------- Tracking ------------------------------------------- 
# Global map of cones
class Cone_map:
    def __init__(self, fcones, Rq, t):
        self.cones = []  # list of 'cone' objects

        # First frame
        for c in fcones:
            self.cones.append(Cone(0, c['u'], c['v'], reproj_cone_Z(t, get_cone_dXYZq(c['u'], c['v'], Rq), 0)))

        self.frames = [get_A(Rq, t)]  # List of all camera rotations + translations
        self.eps = np.inf  # Min L2 norm BA error in pixels

    def __str__(self):
        return f"str Cones = {self.cones}"

    def add_cones(self, fcones, Rq, t, last, same_cone_bound=config.SAME_CONE_BOUND):

        cones_link = []  # (last_index, distance, u, v, pZ0, dXYZq)
        for c in fcones:
            dXYZq = get_cone_dXYZq(c['u'], c['v'], Rq)
            pZ0 = reproj_cone_Z(t, dXYZq, 0)
            cones_link.append(min(enumerate([(self.cones[l].lp - pZ0).get_xy_length() for l in last]),
                                  key=lambda x: x[1]) + (c['u'], c['v'], pZ0, dXYZq)
                              )

        new_last = []
        # First step: try to assign new cones using the 'continuity' between new and last frames(with reprojection to Z=0)
        cones_link = sorted(cones_link, key=lambda x: x[1])  # Sort by distance

        for l in cones_link:
            new_last.append(last[l[0]])
            if l[1] > same_cone_bound or not self.cones[last[l[0]]].assign_cone(len(self.frames), l[2], l[3], l[4]):
                # Second step: try to assing new cones using the 'closeness' estimated XYZ world cone points in cone map
                cp = min(enumerate([(c.ep - reproj_cone_Z(t, l[5], c.ep.z_val)).get_xy_length() for c in self.cones]),
                         key=lambda x: x[1])
                new_last[-1] = cp[0]
                if cp[1] > same_cone_bound or not self.cones[cp[0]].assign_cone(len(self.frames), l[2], l[3], l[4]):
                    # Finally add definitely new cone to the cone map
                    new_last[-1] = len(self.cones)
                    self.cones.append(Cone(len(self.frames), l[2], l[3], l[4]))

        self.frames.append(get_A(Rq, t))
        return new_last
# ------------------------------------------- Tracking -------------------------------------------

# ------------------------------------------- Bundle Adjustment -------------------------------------------   
def Map_cones(cones, kinematics, R0q, cam_pos,
              cone_seen_cnt=config.CONE_SEEN_CNT):
    global cm, last, camcalib_trig
    Rq = kinematics.orientation * R0q
    t = kinematics.position.to_Quaternionr() + cam_pos.to_Quaternionr().rotate(kinematics.orientation)

    if 'cm' not in globals():
        # Contains cone map indicies(cones) that was added on the frame(add_cones call)
        current = list(range(len(cones)))
        cm = Cone_map(cones, Rq, t)
    else:
        current = cm.add_cones(cones, Rq, t, last)
        print('Add_cones: ', current)

    # Contains cones that was on last frame but not on current
    out = list(set(last) - set(current))

    if out:  # We should have some points to make BA on them
        for c in out:
            if len(cm.cones[c].fuv) > cone_seen_cnt:
                # Pinhole model, no distortion
                cameras = sba.Cameras(np.array([list(cm.frames[f[0]]) for f in cm.cones[c].fuv], dtype=np.float64))
                points = sba.Points(
                    np.array([[cm.cones[c].ep.x_val, cm.cones[c].ep.y_val, cm.cones[c].ep.z_val]], dtype=np.float64),
                    np.array([[[fuv[1], fuv[2]] for fuv in cm.cones[c].fuv]], dtype=np.float64),
                    np.ones((1, cameras.ncameras), dtype=np.byte))
                print('Points: ', points)
                options = sba.Options.fromInput(cameras, points)
                options.motstruct = sba.OPTS_STRUCT  # Optimize structure only
                options.verbose = False

                options.intrcalib = np.array([config.fx, config.cx, config.cy, 1, 0], dtype=np.float64)

                newcameras, newpoints, info = sba.SparseBundleAdjust(cameras, points, options)
                print("New points: ", newpoints)

                if info.epsF < cm.cones[c].eps:
                    cm.cones[c].ep = Vector3r(newpoints.B[0, 0], newpoints.B[0, 1], newpoints.B[0, 2])
                    cm.cones[c].eps = info.epsF
                    if cm.eps < info.epsF:
                        cm.eps = info.epsF

                if camcalib_trig > -1:
                    camcalib_trig = camcalib_trig - 1

                # Perform camera calibration algorithm only once                    
                if camcalib_trig == 0:
                    calibrate_camera(config.cx, config.cy, config.fx, config.fy)
    last = current
#  ------------------------------------------- Bundle Adjustment -------------------------------------------

# ====================================MAIN====================================
t2 = t1 = time.time()
idx = 0
cones_lim = config.CONES_LIM
target_speed, max_throttle = config.TARGET_SPEED, config.MAX_THROTTLE
# Camera intrinsic matrix
intrinsics = config.INTRINSICS
# Images we need to collect before camera calibration
camcalib_trig = config.CAMCALIB_TRIG

c_eps_ratio = 2
filter_speed = 0

model = config.FINAL_MODEL
print('Model is on CUDA check: ', next(model.parameters()).is_cuda)

# Connect with simulator
client = fsds.FSDSClient(ip="")
client.confirmConnection()
client.enableApiControl(True)

# Car navigation
gps = client.getGpsData()
gps_navigator = Gps_nav(gps.gnss.geo_point)
car_state = client.getCarState()

vehicle_path = Vehicle_path(5)

cam_pitch = 22.5 * np.pi / 180
R0q = Quaternionr(0., np.sin(np.pi / 4. + cam_pitch / 2.), 0., np.cos(np.pi / 4. + cam_pitch / 2.)) \
      * Quaternionr(0., 0., -np.sin(np.pi / 4.), np.cos(np.pi / 4.))

# Camera position from settings.json
cam_pos = Vector3r(-0.3, -0.16, 0.8)

last = []

while (t2 - t1) < config.TIME_LIM_SEC:
    t2 = time.time()

    imu = client.getImuData(imu_name='Imu', vehicle_name='FSCar')

    # Calculate the velocity in the vehicle's frame
    v = math.sqrt(math.pow(gps.gnss.velocity.x_val, 2) + math.pow(gps.gnss.velocity.y_val, 2))

    cs_prev = car_state
    # Get the car state
    car_state = client.getCarState()

    [image] = client.simGetImages([fsds.ImageRequest(camera_name='cam1', image_type=fsds.ImageType.Scene,
                                                     pixels_as_float=False, compress=False)],
                                  vehicle_name='FSCar')

    img_arr = np.array(
        sldim(sldim(image.image_data_uint8, 3, lambda p: int((p[0] + p[1] + p[2]) / 3)), image.width, lambda l: l),
        dtype=np.uint8)

    t3 = time.time()
    if 'vt' in globals():
        vt.join()
        cones, im = results[:]

    results = [None, None]

    vt = threading.Thread(target=visual_thread, args=(img_arr, results, model))
    vt.start()

    if 'cones' in globals() and len(cones) != 0:
        Map_cones(cones, car_state.kinematics_estimated, R0q=R0q, cam_pos=cam_pos)
        car_controls = fsds.CarControls()
        span_max, s_, i_ = calculate_steering(cones, im.copy(), image.width, image.height)

        while True:
            cones.pop()
            if len(cones) <= config.CONE_MIN:
                break
            span, s, i = calculate_steering(cones, im.copy(), image.width, image.height)
            if abs(span) > abs(span_max):
                s_ = s
                i_ = i
                span_max = span

        car_controls.steering = s_
        im = i_

        for l in last:
            fuv = cm.cones[l].fuv[-1]
            cv2.putText(im, str(l), (fuv[1], fuv[2]), cv2.FONT_HERSHEY_PLAIN, 1, (255))

        car_controls.throttle = calculate_throttle(v, s_)
        car_controls.brake = 0
        car_controls.handbrake = False
        car_controls.gear_immediate = True
        car_controls.is_manual_gear = True
        car_controls.manual_gear = 1

        client.setCarControls(car_controls)

    gps = client.getGpsData()
    x, y = gps_navigator.get_nav(gps.gnss.geo_point)
    vehicle_path.add_point({'x': x, 'y': y})

    if not 'video' in globals():
        video = cv2.VideoWriter(str(time.time()) + ".avi", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10,
                                (image.width + image.height, image.height), 0)
    else:
        video.write(
            np.concatenate((im, get_car_imgmap(image.height, Quaternionr(1.0, 0, 0, 0).rotate(imu.orientation), None)),
                           axis=1))
    idx += 1

    if loop_closure(Vector3r(x, y, 0)):
        break

video.release()
car_controls = fsds.CarControls()
car_controls.throttle = 0.0
car_controls.brake = 0.1
client.reset()
client.setCarControls(car_controls)
print(idx / (t2 - t1))
