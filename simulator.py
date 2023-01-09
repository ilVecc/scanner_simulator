#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import of useful packages

import cv2
import bpy
import bpycv
#import boxx
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import math as m
import mathutils
from pathlib import Path
from pyntcloud import PyntCloud

import sys
sys.path.append(str(Path(bpy.data.filepath).parent))

from utils import *

CURRENT_DIR = Path(__file__).resolve().parent  # this results in either the .blend file or the directory
if CURRENT_DIR.suffix == ".blend":
    CURRENT_DIR = CURRENT_DIR.parent  # double  .parent  because .py file is (apparently) inside the .blend file

# https://docs.blender.org/manual/en/2.79/render/workflows/command_line.html
# blender --background --python SCRIPT.py

# https://docs.blender.org/manual/en/latest/advanced/scripting/addon_tutorial.html
# https://docs.blender.org/manual/en/3.2/addons/add_curve/extra_objects.html#
# __import__("addons_utils").enable("add_curve_extra_objects")
# bpy.ops.curve.spirals(spiral_type='ARCH', spiral_direction='COUNTER_CLOCKWISE', turns=1, steps=24, radius=1, dif_z=0, dif_radius=0, B_force=1, inner_radius=0.2, dif_inner_radius=0, cycles=1, curves_number=1, touch=False, shape='3D', curve_type='POLY', use_cyclic_u=False, endp_u=True, order_u=4, handleType='VECTOR', edit_mode=True, startlocation=(0, 0, 0), rotation_euler=(0, 0, 0))

import argparse

parser = argparse.ArgumentParser(
    prog="ScanSim",
    description="Blender 3D scanner simulator for Point Cloud and RGB-D synthetic dataset creation",
)
parser.add_argument("file_mesh", metavar="MESH", type=str, help="STL file path of the object to be scanned")
parser.add_argument("file_params", metavar="PARAMS", type=str, help="YAML file path of the camera parameters")
parser.add_argument("save_dir", metavar="OUTPUT", type=str, help="Output path for the dataset")
parser.add_argument("n_imgs", metavar="N_IMAGES", default=50, type=int, help="Number of samples to acquire (defaults to 50)")
parser.add_argument("--subsamples", metavar="SUBSAMPLES", default=100_000, type=int, help="Number of random points to save after 3D scan")
parser.add_argument("--random", action="store_true", help="Render from random poses around the object instead of following the default scanning path")
args = parser.parse_args(sys.argv[4:])  # skip "blender.exe", "-b", "--python", "simulator.py"

# INPUTS ###########################################################################
FILE_MESH = Path(args.file_mesh).resolve()  # CURRENT_DIR / "input" / "dante.stl"
FILE_PARAMS = Path(args.file_params).resolve()  # CURRENT_DIR / "input" / "params.yml"
# W, H = 5472, 3648
####################################################################################

def _barrel_positioning(radius, height, t_angle, t_height):
    angle = 2 * m.pi * t_angle
    r = radius * (m.sin(m.pi * t_height) * 0.25 + 0.75)  # TODO add some randomness here
    x, y = r * m.cos(angle), r * m.sin(angle)
    z = height * (t_height - 0.5)  # wrt z=0 (the same as the object origin)
    return x, y, z


def traj_random_barrel(t, min_radius, obj_focused):
    t_angle = random.uniform(0, 1)
    t_height = random.uniform(0, 1)
    return _barrel_positioning(1.5 * min_radius, 1.5 * obj_focused.dimensions.z, t_angle, t_height)


def traj_spiral_barrel(t, min_radius, obj_focused):
    return _barrel_positioning(1.5 * min_radius, 1.5 * obj_focused.dimensions.z, t, t)


# handy variables ##################################################################
SAVE_DIR = Path(args.save_dir).resolve()  # CURRENT_DIR / "output"
SAVE_DIR.mkdir(exist_ok=True)
SAVE_DIR_IMAGES = SAVE_DIR / "images"
SAVE_DIR_IMAGES.mkdir(exist_ok=True)
SAVE_DIR_DEPTHS = SAVE_DIR / "depth"
SAVE_DIR_DEPTHS.mkdir(exist_ok=True)
SAVE_DIR_ANNOTS = SAVE_DIR / "annotations"
SAVE_DIR_ANNOTS.mkdir(exist_ok=True)
SAVE_DIR_CLOUDS = SAVE_DIR / "point_clouds"
SAVE_DIR_CLOUDS.mkdir(exist_ok=True)
####################################################################################

# set metric units and meter as base (for depth images)
bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.unit_settings.scale_length = 1.0

# remove all MESH objects and de-select/-activate everything
# Blender default file (userpref.blend) contains three objects: Camera, Cube, and Light
# this actively removes the previously used mesh when launching the script from Blender,
# and the default Cube when launching the script from CLI, leaving only Camera and Light
[bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]
bpy.context.view_layer.objects.active = None
# https://blender.stackexchange.com/questions/99664
for obj in bpy.context.scene.objects:
    obj.select_set(False)

# load custom object at world origin (this also activates, and thus selects, the object)
bpy.ops.import_mesh.stl(filepath=str(FILE_MESH))
obj_name = str(FILE_MESH.stem)
obj = bpy.data.objects[obj_name]
# set the object origin to its geometrical center, and place it at the world origin
# obj.select_set(True)  # selection is needed for the following command, but the object is already selected when just imported
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
bpy.ops.object.location_clear(clear_delta=False)
# give the object a unique id for annotation and 6D pose with bpycv
obj["inst_id"] = 1


###
# CAMERA AND LIGHT SETUP
###
cam = bpy.data.cameras["Camera"]
light = bpy.data.lights["Light"]
obj_cam = bpy.data.objects["Camera"]
obj_light = bpy.data.objects["Light"]
# load intrinsics from file
K, f = load_intrinsics(FILE_PARAMS)
# setup camera intrinsics
H, W = set_cam_intrinsics(K, f, cam=cam)
# setup camera frustum (for Z-buffer)
cam.clip_start = 0.1
cam.clip_end = 100
# setup world background
bpy.context.scene.world.use_nodes = False
bpy.context.scene.world.color = mathutils.Color((0, 0, 0))


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
] + [
    f"E_T{i}" for i in range(1,4)
] + [
    # obj2cam matrix
    f"P_R{i}{j}" for i in range(1,4) for j in range(1,4)
] + [
    f"P_T{i}" for i in range(1,4)
] + [
    # vertices in bb2obj space
    i 
    for l in [[f"BB_{w}{i}" for w in ["x", "y", "z"]] for i in range(8)] 
    for i in l
]


N_IMAGES = args.n_imgs
SUBSAMPLES = args.subsamples
CAMERA_POSITIONING = traj_random_barrel if args.random else traj_spiral_barrel

# orbit around the object
cam_min_radius = minimum_cam_distance_cam(cam, obj)
# setup lighting
light.shadow_soft_size = cam_min_radius / 2
light.energy = 2000

# U is the conversion matrix from Blender to "standrd computer vision" reference system
# more on: https://blender.stackexchange.com/questions/86398/
U = np.diag([1, -1, -1])

info_poses = pd.DataFrame(columns=df_columns)
subsamples = min(SUBSAMPLES, H * W)
for i in tqdm(range(N_IMAGES)):
    
    path_image = str(SAVE_DIR_IMAGES / f"img_{i}.png")
    path_depth = str(SAVE_DIR_DEPTHS / f"depth_{i}.png")
    path_annot = str(SAVE_DIR_ANNOTS / f"annotations_{i}.png")
    path_cloud = str(SAVE_DIR_CLOUDS / f"cloud_{i}.ply")
    
    # move camera and light around object
    t = i / max(1, (N_IMAGES - 1))
    obj_cam.location = CAMERA_POSITIONING(t, cam_min_radius, obj)
    obj_light.location = obj_cam.location + mathutils.Vector([0, 0, 0.1])
    
    # point towards the object
    look_at(obj_cam, obj)
    
    ###########################################################
    # DEBUGGING PURPOSES (update the viewport)
    # works only when Blender is in windowed mode (not in CLI)
    #bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    ###########################################################
    
    # render image, instance annotation and depth in one line code
    with stdout_redirected():
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
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    # filter clipped points (they are outside frustum, Blender sets their value to 0) 
    u, v, z = u[depth > 0], v[depth > 0], depth[depth > 0]
    # random subsampling of pixels
    idxs = random.choices(np.arange(u.shape[0]), k=subsamples)
    u, v, z = u[idxs], v[idxs], z[idxs]
    
    # anti-project the homogeneous points and select the colors
    p = np.vstack([u, v, np.ones(u.shape)])
    points = U @ np.linalg.inv(K) @ p * z
    colors = image[v, u, :]

    cloud = PyntCloud(
        pd.DataFrame(
            # same arguments that you are passing to visualize_pcl
            columns=["x", "y", "z", "red", "green", "blue"],
            data=np.hstack((points.T, colors))
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
    info_poses_this = pd.DataFrame(
        columns=df_columns,
        data=np.array([[
            name,            
            path_image, path_depth, path_annot, path_cloud,
            *extract_K(I), *extract_E(E), *extract_E(P),
            *BB[0], *BB[1], *BB[2], *BB[3], *BB[4], *BB[5], *BB[6], *BB[7]
        ]])  # this needs to be a 2D array, thus the [[ ]]
    )
    info_poses = pd.concat([info_poses, info_poses_this], ignore_index=True)

info_poses.reset_index()
info_poses.to_csv(str(SAVE_DIR / "poses.csv"))


# quick and easy render
#bpy.context.scene.render.filepath = str((PNG_PATH / str(inst_id)).with_suffix(".png"))
#bpy.ops.render.render(write_still=True)
