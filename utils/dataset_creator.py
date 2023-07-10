import cv2
import pandas as pd
import numpy as np
rnd = np.random.default_rng()
from pyntcloud import PyntCloud
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from .camera import Camera


def create_info():
    return pd.DataFrame(columns=[
        # info
        "mesh_name",
        "path_mesh",
        "path_image",
        "path_depth",
        "path_annot",
        "path_cloud"
    ] + [
        # camera model name
        "camera",
        # focal length
        "f"
    ] + [
        # I matrix
        "K_fx", "K_fy", "K_cx", "K_cy", "K_s"
    ] + [
        # E matrix
        f"E_R{i}{j}" for i in range(1, 4) for j in range(1, 4)
    ] + [
        f"E_T{i}" for i in range(1, 4)
    ] + [
        # obj2cam matrix
        f"P_R{i}{j}" for i in range(1, 4) for j in range(1, 4)
    ] + [
        f"P_T{i}" for i in range(1, 4)
    ] + [
        # vertices in bb2obj space
        i 
        for l in [[f"BB_{w}{i}" for w in ["x", "y", "z"]] for i in range(8)] 
        for i in l
    ] + [
        # depth_ambiguity
        "ambiguity"
    ] + [
        # projection
        "cx", "cy", "tz"
    ])


def update_info(info_poses: pd.DataFrame, paths: dict, camera_model_name:str, pose: dict, f: float, depth_ambiguity: bool):
    ###
    # 6D POSES
    ###
    #  ycb_6d_pose: dict  10
    #    ├── intrinsic_matrix: (3, 3)float32  # (I matrix)
    #    ├── world_to_cam: (4, 4)float32      # pose of Blender world wrt CV camera (world2cvcam, E matrix)
    #    ├── cam_matrix_world: (4, 4)float64  # pose of Blender camera wrt Blender world (blendcam2world, inv(E_B) where E_B is the extrinsics matrix of the Blender camera)
    #    ├── inst_ids: list  1
    #    │   └── 0: 1
    #    ├── areas: list  1
    #    │   └── 0: 3095214
    #    ├── visibles: list  1
    #    │   └── 0: True
    #    ├── poses: (3, 4, 1)float64
    #    ├── 6ds: list  1                     # poses of Blender object (in Blender world coordinates) wrt CV camera (obj2cvcam, P matrix -> since the object is exactly at the origin and with the same orientation P = E)
    #    │   └── 0: (3, 4)float64
    #    ├── bound_boxs: list  1              # box vertices wrt Blender object (bb2obj)
    #    │   └── 0: (8, 3)float64
    #    └── mesh_names: list  1
    #        └── 0: dante
    name = pose["mesh_names"][0]
    I = pose["intrinsic_matrix"]
    E = pose["world_to_cam"]  # give 0, get world
    P = pose["6ds"][0]  # give 0, get object
    BB = pose["bound_boxs"][0]
    # calculate the center of projection of the object
    fx, fy, px, py, s = Camera.unpack_K(I)
    tx, ty, tz = P[:3, 3]
    cx = fx * tx / tz + px
    cy = fy * ty / tz + py
    # store everything
    info_poses_this = pd.DataFrame(
        columns=info_poses.columns,
        data=np.array([[
            name,
            paths["mesh"], paths["image"], paths["depth"], paths["annot"], paths["cloud"],
            camera_model_name,
            f, fx, fy, px, py, s, *Camera.decompose(E), *Camera.decompose(P),
            depth_ambiguity,
            *BB[0], *BB[1], *BB[2], *BB[3], *BB[4], *BB[5], *BB[6], *BB[7],
            cx, cy, tz
        ]])  # this needs to be a 2D array, thus the [[ ]]
    )
    info_poses = pd.concat([info_poses, info_poses_this], ignore_index=True)
    return info_poses


def create_cloud(image, depth, K, samples):
    # U is the conversion matrix from Blender to "standrd computer vision" reference system
    # more on: https://blender.stackexchange.com/questions/86398/
    U = np.diag([1, -1, -1])
    # prepare the grid
    h, w, _ = image.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    # filter clipped points (they are outside frustum, Blender sets their value to 0) 
    u, v, z = u[depth > 0], v[depth > 0], depth[depth > 0]
    # random subsampling of pixels
    idxs = rnd.choices(np.arange(z.shape[0]), k=samples)
    u, v, z = u[idxs], v[idxs], z[idxs]
    
    # anti-project the homogeneous points and select the colors
    p = np.vstack([u, v, np.ones(u.shape)])
    points = U @ np.linalg.inv(K) @ p * z  # TODO maybe U is useless
    colors = image[v, u, :]

    return points, colors


def save_cloud(filepath, points, colors):
    cloud = PyntCloud(
        pd.DataFrame(
            # same arguments that you are passing to visualize_pcl
            columns=["x", "y", "z", "red", "green", "blue"],
            data=np.hstack((points, colors))
        )
    )
    cloud.to_file(filepath)


def add_background(image, mask, backgound, augment=True):
    bh, bw = backgound.shape[:2]
    fh, fw = image.shape[:2]
    if fh > bh:
        bh, bw = fh, int(fh / bh * bw)
        backgound = cv2.resize(backgound, (bw, bh))
    if fw > bw:
        bh, bw = int(fw / bw * bh), fw
        backgound = cv2.resize(backgound, (bw, bh))
    
    y = (bh - fh) * .5
    x = (bw - fw) * .5
    back = backgound[int(y):int(y+fh), int(x):int(x+fw), :]
    bh, bw = fh, fw

    if augment:
        a = np.radians(rnd.uniform(-6.0, 6.0))
        ca, sa = np.cos(a), np.sin(a)
        ox, oy = rnd.uniform(-10, 10), rnd.uniform(-10, 10)
        s = rnd.uniform(1.50, 1.75)
        M = np.array([[s * ca, sa, ox], [-sa, s * ca, oy]])
        back = cv2.warpAffine(back, M, (bw, bh))
        if rnd.choice(a=[False, True]):
            back = cv2.flip(back, flipCode=1)
    
    back[mask, :] = image[mask, :]

    return back


def get_model_info(filepath):

    plydata = PyntCloud.from_file(filepath)
    points = plydata.xyz

    min_coord = np.min(points, axis=0)
    box_size = np.max(points, axis=0) - min_coord

    hull = ConvexHull(points)  # find convex hull in O(N log N)
    hullpoints = points[hull.vertices, :]  # extract the points forming the hull
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')  # naive worst-distance best pair in O(H^2)
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)  # get the farthest apart points
    p1, p2 = hullpoints[bestpair[0]], hullpoints[bestpair[1]]
    dist = np.linalg.norm(p1-p2)

    data = {
        'diameter': dist, 
        'min_x': min_coord[0], 
        'min_y': min_coord[1], 
        'min_z': min_coord[2], 
        'size_x': box_size[0], 
        'size_y': box_size[1], 
        'size_z': box_size[2]
    }
    return data


from pathlib import Path
def create_simple_dataset(path_data: Path, path_output: Path, classes: list, split=0.8):

    if not path_data.exists():
        raise ValueError(f"Provided data path does not exist: {path_data}")

    path_output.mkdir(exist_ok=True, parents=True)

    # mix all classes if no classes provided
    mixed_dataset = False
    if classes is None or classes is []:
        classes = [path.stem for path in path_data.iterdir()]
        mixed_dataset = True

    for clazz in classes:
        # read the dataset info
        dataset_path = path_data / clazz
        dataset_info = dataset_path / f"{clazz}.csv"
        info = pd.read_csv(dataset_info)
        # add the training selection column
        info["training"] = False
        training_idxs = rnd.choice(info.shape[0], int(info.shape[0] * split), replace=False)
        info.loc[training_idxs, ("training")] = True

        
    



    
# LINEMOD
# - data/
#   - 01/
#     - mask/
#     - rgb/
#     - gt.yml
#     - info.yml
#     - test.txt
#     - train.txt
# - models
#   - models_info.yml
#   - obj_01.ply
