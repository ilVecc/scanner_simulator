import numpy as np
from ruamel.yaml import YAML

class Camera(object):

    def __init__(self, focal, sensor, offset=(0, 0), hw=None):
        super().__init__()
        self.focal = focal
        self.sensor_x, self.sensor_y = sensor
        self.offset_x, self.offset_y = offset
        self.compute_intrinsics(hw)
    
    def __init__(self, K, f, hw=None):
        super().__init__()
        self.focal = f
        self.fx, self.fy, self.cx, self.cy, _ = Camera.unpack_K(K)
        self.compute_camera(hw)

    
    def compute_intrinsics(self, hw):
        if not hw:
            return
        self.fx, self.fy, self.cx, self.cy = Camera.compute_intrinsics(self.focal, self.sensor_x, self.sensor_y, self.offset_x, self.offset_y, hw)


    def compute_camera(self, hw):
        if not hw:
            return
        self.sw, self.sh, self.ow, self.oh = Camera.compute_camera(self.f, self.fx, self.fy, self.cx, self.cy, hw)


    def pack_K(self):
        return Camera.pack_K(self.fx, self.fy, self.cx, self.cy)


    @staticmethod
    def compute_intrinsics(f, sw, sh, ow, oh, hw):
        if not hw:
            raise AttributeError("Cannot compute intrisic parameters without image resolution")
        h, w = hw
        fx = f * w / sw
        fy = f * h / sh
        cx = w / 2 + ow
        cy = h / 2 + oh
        return fx, fy, cx, cy


    @staticmethod
    def compute_camera(f, fx, fy, cx, cy, hw):
        if not hw:
            raise AttributeError("Cannot compute camera parameters without image resolution")
        h, w = hw
        sw = f * w / fx
        sh = f * h / fy
        ow = cx - w / 2
        oh = cy - h / 2
        return sw, sh, ow, oh


    @staticmethod
    def pack_K(fx, fy, cx, cy, s=0.0):
        K = np.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy
        K[0,1] = s
        return K


    @staticmethod
    def unpack_K(K):
        K /= K[2,2]
        fx, fy, cx, cy, s = K[0,0], K[1,1], K[0,2], K[1,2], K[0,1]
        return fx, fy, cx, cy, s


    @staticmethod
    def decompose(T):
        return T[0][0], T[0][1], T[0][2], \
            T[1][0], T[1][1], T[1][2], \
            T[2][0], T[2][1], T[2][2], \
            T[0][3], T[1][3], T[2][3]


def load_intrinsics(path):
    yaml = YAML()
    with open(path, "r") as f:
        params = yaml.load(f)
    K = Camera.pack_K(params["cam_fx"], params["cam_fy"], params["cam_cx"], params["cam_cy"], params["cam_sk"])
    f = params.setdefault("cam_f", None)
    return K, f


# get the minimum distance of the camera to frame the whole target from intrinsics
def minimum_cam_distance(K, f, hw, dims):
    fx, fy, cx, cy, _ = Camera.unpack_K(K)
    # radius of the sphere circumscribing the bounding box
    dx, dy, dz = dims[0], dims[1], dims[2]
    obj_radius = np.sqrt(dx**2 + dy**2 + dz**2) / 2
    h, w = hw
    Kx, Ky = fx / f, fy / f
    t, b = cy / Ky, (h - cy) / Ky
    l, r = cx / Kx, (w - cx) / Kx
    theta = np.arctan(min(t, b, l, r) / f)
    return obj_radius / np.sin(theta)

