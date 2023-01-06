import bpy
import math as m
import os
import sys
import numpy as np
from ruamel.yaml import YAML
from contextlib import contextmanager
import mathutils

def look_at(obj_camera, obj_focused):
    # use world coordinates to 
    direction = obj_focused.location - obj_camera.location
    # point the cameras '-Z' and use its 'Y' as up
    obj_camera.rotation_mode = "QUATERNION"
    obj_camera.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    # rotate camera if height is bigger than base
    angle = m.radians(-90) if max(obj_focused.dimensions.x, obj_focused.dimensions.y) < obj_focused.dimensions.z else 0
    obj_camera.rotation_quaternion @= mathutils.Quaternion((0, 0, 1), angle)


def create_K(fx, fy, cx, cy, s):
    K = np.eye(3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    K[0,1] = s
    return K


def extract_K(K):
    K /= K[2,2]
    fx, fy, cx, cy, s = K[0,0], K[1,1], K[0,2], K[1,2], K[0,1]
    return fx, fy, cx, cy, s


def extract_E(E):
    return E[0][0], E[0][1], E[0][2], \
           E[1][0], E[1][1], E[1][2], \
           E[2][0], E[2][1], E[2][2], \
           E[0][3], E[1][3], E[2][3]


def load_intrinsics(path):
    yaml = YAML()
    with open(path, "r") as f:
        params = yaml.load(f)
    K = create_K(params["cam_fx"], params["cam_fy"], params["cam_cx"], params["cam_cy"], params["cam_s"])
    f = params["cam_f"]
    return K, f


def set_cam_intrinsics(K, f, hw=None, cam=bpy.data.cameras["Camera"]):
    # Blender ignores camera skew, so no  K[0,1]  here
    return set_cam_intrinsics_params(K[0,0], K[1,1], K[0,2], K[1,2], f, hw, cam)


# mimics bpycv.set_cam_intrinsics()
def set_cam_intrinsics_params(fx, fy, cx, cy, f, hw, cam):
    if hw is None:
        # resolution is set to give (cx,cy) = (0,0)
        hw = (int(cy*2), int(cx*2))
    if cam is None:
        cam = bpy.data.cameras["Camera"]
    
    h, w = hw
    cam.lens = f
    cam.sensor_fit = 'AUTO'
    cam.sensor_width = f * w / fx
    cam.sensor_height = f * h / fy
    cam.shift_x = 0.5 - cx / w
    cam.shift_y = cy / h - 0.5
    bpy.context.scene.render.resolution_x = w
    bpy.context.scene.render.resolution_y = h
    return (h, w)


# get the minimum distance of the camera to frame the whole target from intrinsics
def minimum_cam_distance(K, f, hw, target):
    fx, fy, cx, cy, _ = extract_K(K)
    dx, dy, dz = target.dimensions
    # radius of the sphere circumscribing the bounding box
    obj_radius = m.sqrt(dx**2 + dy**2 + dz**2) / 2
    h, w = hw
    t = cy * f / fy
    b = (h - cy) * f / fy
    l = cx * f / fx
    r = (w - cx) * f / fx
    theta = m.atan(min(t, b, l, r) / f)
    return obj_radius / m.sin(theta)

# fetch the necessary inputs directly from the camera
def minimum_cam_distance_cam(cam, target):
    h, w = bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x
    f = cam.lens
    fx = f * w / cam.sensor_width
    fy = f * h / cam.sensor_height
    cx = (0.5 - cam.shift_x) * w
    cy = (cam.shift_y + 0.5) * h
    K = create_K(fx, fy, cx, cy, 0)
    return minimum_cam_distance(K, f, (h, w), target)


@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

