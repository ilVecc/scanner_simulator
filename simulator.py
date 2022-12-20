#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import of useful packages

import cv2
import bpy
import bpycv
import boxx
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import mathutils
from pathlib import Path
from ruamel.yaml import YAML
from pyntcloud import PyntCloud

# https://docs.blender.org/manual/en/2.79/render/workflows/command_line.html
# blender --background --python SCRIPT.py

# https://docs.blender.org/manual/en/latest/advanced/scripting/addon_tutorial.html
# https://docs.blender.org/manual/en/3.2/addons/add_curve/extra_objects.html#
# __import__("addons_utils").enable("add_curve_extra_objects")
# bpy.ops.curve.spirals(spiral_type='ARCH', spiral_direction='COUNTER_CLOCKWISE', turns=1, steps=24, radius=1, dif_z=0, dif_radius=0, B_force=1, inner_radius=0.2, dif_inner_radius=0, cycles=1, curves_number=1, touch=False, shape='3D', curve_type='POLY', use_cyclic_u=False, endp_u=True, order_u=4, handleType='VECTOR', edit_mode=True, startlocation=(0, 0, 0), rotation_euler=(0, 0, 0))


def look_at(obj_camera, obj_focused):
    # use world coordinates to 
    direction = obj_focused.location - obj_camera.location
    # point the cameras '-Z' and use its 'Y' as up
    obj_camera.rotation_mode = "QUATERNION"
    obj_camera.rotation_quaternion = direction.to_track_quat('-Z', 'Y')

    
# mimics bpycv.set_cam_intrinsics()
def set_cam_intrinsics(cam, fx, fy, cx, cy, f, hw=(1920, 1080)):
    h, w = hw
    cam.lens = f
    cam.sensor_width = f * w / fx
    cam.sensor_height = f * h / fy
    cam.shift_x = (0.5 - cx / w)
    cam.shift_y = (0.5 - cy / h)
    bpy.context.scene.render.resolution_x = w
    bpy.context.scene.render.resolution_y = h

# resolution is set to give (cx,cy) = (0,0)
def set_cam_intrinsics_proper(cam, fx, fy, cx, cy, f):
    set_cam_intrinsics(cam, fx, fy, cx, cy, f, hw=(cy*2, cx*2))

# INPUTS ###########################################################################
CURRENT_PATH = Path(__file__).resolve().parent.parent  # double  .parent  because .py file is (apparently) inside the .blend file
FILE_MESH = CURRENT_PATH / "input" / "dante.stl"
FILE_PARAMS = CURRENT_PATH / "input" / "params.yml"
W, H = 5472, 3648
####################################################################################

SAVE_DIR = CURRENT_PATH / "output"
SAVE_DIR.mkdir(exist_ok=True)
SAVE_DIR_IMAGES = SAVE_DIR / "images"
SAVE_DIR_IMAGES.mkdir(exist_ok=True)
SAVE_DIR_DEPTHS = SAVE_DIR / "depth"
SAVE_DIR_DEPTHS.mkdir(exist_ok=True)
SAVE_DIR_ANNOTS = SAVE_DIR / "annotations"
SAVE_DIR_ANNOTS.mkdir(exist_ok=True)
SAVE_DIR_CLOUDS = SAVE_DIR / "point_clouds"
SAVE_DIR_CLOUDS.mkdir(exist_ok=True)

# handy variables
obj_name = str(FILE_MESH.stem)

# remove all MESH objects and de-select/-activate everything
[bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]
bpy.context.view_layer.objects.active = None
bpy.ops.object.select_all(action='DESELECT')
# set metric units and meter as base (for depth images)
bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.unit_settings.scale_length = 1.0

# load custom object at world origin (this also activates, and thus selects, the object)
bpy.ops.import_mesh.stl(filepath=str(FILE_MESH))
obj = bpy.data.objects[obj_name]
# set the object origin to its geometrical center, and place it at the world origin
# obj.select_set(True)  # selection is needed for the following command, but the object is already selected
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
bpy.ops.object.location_clear(clear_delta=False)
# give the object a unique id for annotation and 6D pose with bpycv
obj["inst_id"] = 1

# load intrinsics from file
yaml = YAML()
with open(FILE_PARAMS, "r") as f:
    params = yaml.load(f)

cam = bpy.data.cameras["Camera"]
set_cam_intrinsics(
    cam, 
    params["cam_fx"], params["cam_fy"], 
    params["cam_cx"], params["cam_cy"], 
    params["cam_f"], 
    (H, W)
)
cam.clip_start = 0.1
cam.clip_end = 100

# set light
light = bpy.data.objects["Light"]


# prepare dataframe
df_columns = [
    # info
    "mesh_name",
    "path_image",
    "path_depth",
    "path_annot",
    "path_cloud",
    # I matrix
    "K_fx", "K_fy", "K_cx", "K_cy", "K_s"
] + [
    # E matrix
    f"E_R{i}{j}" for i in range(1,4) for j in range(1,4)
] + [f"E_T{i}" for i in range(1,4)
] + [
    # obj2cam matrix
    f"P_R{i}{j}" for i in range(1,4) for j in range(1,4)
] + [f"P_T{i}" for i in range(1,4)
] + [
    # vertices in bb2obj space
    i 
    for l in [[f"BB_x{i}", f"BB_y{i}", f"BB_z{i}"] for i in range(8)] 
    for i in l
]



# orbit around the object
obj_cam = bpy.data.objects["Camera"]
cam_min_radius = math.sqrt(obj.dimensions.x ** 2 + obj.dimensions.y ** 2 + obj.dimensions.z ** 2)

info_poses = pd.DataFrame(columns=df_columns)
n = 5
for i in tqdm(range(n)):
    
    path_image = str(SAVE_DIR_IMAGES / f"img_{i}.png")
    path_depth = str(SAVE_DIR_DEPTHS / f"depth_{i}.png")
    path_annot = str(SAVE_DIR_ANNOTS / f"annotations_{i}.png")
    path_cloud = str(SAVE_DIR_CLOUDS / f"cloud_{i}.ply")
    
    # move camera around object
    t = i / (n - 1)
    angle = 2 * math.pi * t
    height = (1.5 * obj.dimensions.z) * (t - 0.5)  # wrt z=0 (the same as the kbject origin)
    radius = (1.5 * cam_min_radius) * (math.sin(math.pi * t) * 0.25 + 0.75)  # TODO add some randomness here
    obj_cam.location = (radius * math.cos(angle), radius * math.sin(angle), height)
    # move light with camera
    light.location = obj_cam.location + mathutils.Vector([0, 0, 0.1])
    
    # point towards the object
    look_at(obj_cam, obj)
    # rotate camera if height is bigger than base
    angle = math.radians(-90) if max(obj.dimensions.x, obj.dimensions.y) < obj.dimensions.z else 0
    obj_cam.rotation_quaternion @= mathutils.Quaternion((0, 0, 1), angle)
    
    ###########################################################
    # DEBUGGING PURPOSES
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    ###########################################################
    
    # render image, instance annotation and depth in one line code
    frame_data = bpycv.render_data()
    image = frame_data["image"]
    depth = frame_data["depth"]  # in meters
    annot = frame_data["inst"]
    pose = frame_data["ycb_6d_pose"]  # 6D pose in YCB format
    
    
    ### 
    # IMAGE, DEPTH and ANNOTATIONS
    ###
    cv2.imwrite(path_image, image)
    cv2.imwrite(path_depth, depth)
    cv2.imwrite(path_annot, annot)
    
    
    ### 
    # POINT CLOUD
    ###
    xv, yv = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    
    # filter clipped points (they are outside frustum, Blender sets their value to 0) 
    xv, yv, f = xv[depth > 0], yv[depth > 0], depth[depth > 0]

    # anti-project the points
    # more on: https://blender.stackexchange.com/questions/86398/camera-view-coordinates-and-z-depth-values
    # here we also directly write the U = diag(1,-1,-1) matrix for Blender to "standrd computer vision" reference system conversion
    x = +1*(xv - params["cam_cx"]) * f / params["cam_fx"]
    y = -1*(yv - params["cam_cy"]) * f / params["cam_fy"]
    z = -1*f
    points = np.vstack((x, y, z)).T
    colors = image[yv, xv, :]

    # random subsampling
    idxs = random.choices(np.arange(points.shape[0]), k=100_000)
    points, colors = points[idxs, :], colors[idxs, :]

    cloud = PyntCloud(
        pd.DataFrame(
            # same arguments that you are passing to visualize_pcl
            columns=["x", "y", "z", "red", "green", "blue"],
            data=np.hstack((points, colors))
        )
    )
    cloud.to_file(path_cloud)


    ###
    # 6D POSES
    ###
    #  ycb_6d_pose: dict  10
    #    ├── intrinsic_matrix: (3, 3)float32
    #    ├── world_to_cam: (4, 4)float32      # pose of world wrt camera (extrinsics E in standard computer vision convention)
    #    ├── cam_matrix_world: (4, 4)float64  # pose of camera wrt world (cam2world  inv(E) and necessary adjustments)
    #    ├── inst_ids: list  1
    #    │   └── 0: 1
    #    ├── areas: list  1
    #    │   └── 0: 3095214
    #    ├── visibles: list  1
    #    │   └── 0: True
    #    ├── poses: (3, 4, 1)float64
    #    ├── 6ds: list  1                     # poses of object wrt camera (obj2cam)
    #    │   └── 0: (3, 4)float64
    #    ├── bound_boxs: list  1              # box vertices wrt obj (bb2obj)
    #    │   └── 0: (8, 3)float64
    #    └── mesh_names: list  1
    #        └── 0: dante
    name, I, E, P, BB = pose["mesh_names"][0], pose["intrinsic_matrix"], pose["world_to_cam"], pose["6ds"][0], pose["bound_boxs"][0]
    info_poses = info_poses.append(
        pd.DataFrame(
            columns=df_columns,
            data=np.array([[
                name,            
                path_image,
                path_depth,
                path_annot,
                path_cloud,
                I[0][0], I[1][1], I[0][2], I[1][2], I[0][1],
                E[0][0], E[0][1], E[0][2], E[1][0], E[1][1], E[1][2], E[2][0], E[2][1], E[2][2],
                E[0][3], E[1][3], E[2][3],
                P[0][0], P[0][1], P[0][2], P[1][0], P[1][1], P[1][2], P[2][0], P[2][1], P[2][2],
                P[0][3], P[1][3], P[2][3],
                BB[0][0], BB[0][1], BB[0][2], 
                BB[1][0], BB[1][1], BB[1][2], 
                BB[2][0], BB[2][1], BB[2][2], 
                BB[3][0], BB[3][1], BB[3][2], 
                BB[4][0], BB[4][1], BB[4][2], 
                BB[5][0], BB[5][1], BB[5][2], 
                BB[6][0], BB[6][1], BB[6][2], 
                BB[7][0], BB[7][1], BB[7][2]
            ]])  # this needs to be a 2D array, thus the [[ ]]
        )
    )

info_poses.reset_index()
info_poses.to_csv(str(SAVE_DIR / "poses.csv"))


# quick and easy render
#bpy.context.scene.render.filepath = str((PNG_PATH / str(inst_id)).with_suffix(".png"))
#bpy.ops.render.render(write_still=True)
