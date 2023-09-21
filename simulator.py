#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TAKE A LOOK AT
# https://github.com/google-research/kubric


# Import of useful packages
import bpy
import bpycv  # creates a default world (with the usual cube)
import mathutils
#import boxx
import cv2
import numpy as np
rnd = np.random.default_rng()
import argparse

from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(str(Path(bpy.data.filepath).parent))

from utils.misc import *
from utils.blender import *
from utils.camera import *
from utils.dataset_creator import *
from utils.trajectory import *

CURRENT_DIR = Path(__file__).resolve().parent  # this results in either the .blend file or the directory
if CURRENT_DIR.suffix == ".blend":
    CURRENT_DIR = CURRENT_DIR.parent  # double  .parent  because .py file is (apparently) inside the .blend file

# https://docs.blender.org/api/current/info_gotcha.html
# https://docs.blender.org/manual/en/2.79/render/workflows/command_line.html

# https://docs.blender.org/manual/en/latest/advanced/scripting/addon_tutorial.html
# https://docs.blender.org/manual/en/3.2/addons/add_curve/extra_objects.html#
# __import__("addons_utils").enable("add_curve_extra_objects")
# bpy.ops.curve.spirals(spiral_type='ARCH', spiral_direction='COUNTER_CLOCKWISE', turns=1, steps=24, radius=1, dif_z=0, dif_radius=0, B_force=1, inner_radius=0.2, dif_inner_radius=0, cycles=1, curves_number=1, touch=False, shape='3D', curve_type='POLY', use_cyclic_u=False, endp_u=True, order_u=4, handleType='VECTOR', edit_mode=True, startlocation=(0, 0, 0), rotation_euler=(0, 0, 0))


def parse_args(args):
    parser = argparse.ArgumentParser(
        prog="ScanSim",
        description="Blender 3D scanner simulator for Point Cloud and RGB-D synthetic dataset creation",
    )
    parser.add_argument("outdir", metavar="OUTDIR", 
                        type=str, 
                        help="Output path for the datasets (grouped by mesh by default)")
    parser.add_argument("examples", metavar="EXAMPLES", 
                        type=int, default=1000, 
                        help="Number of samples to acquire for each mesh (defaults to 1000)")
    parser.add_argument("-M", "--meshes", metavar="MESHES", 
                        type=str, nargs='+', required=True,
                        help="STL/GLB/OBJ filepaths of the objects to be scanned (individually)")
    parser.add_argument("-P", "--params", metavar="PARAMS", 
                        type=str, nargs='+', required=True,
                        help="YAML filepaths of the camera parameters (randomly chosen during scanning)")
    parser.add_argument("-n", "--mesh_names", metavar="MESH_NAMES", 
                        type=str, nargs='+', default=None,
                        help="Force a specific name for each mesh")
    parser.add_argument("-m", "--meters", 
                        action="store_true",
                        help="Use meter unit instead of millimiter (because the meshes are given in meter unit)")
    parser.add_argument("-s", "--samples", metavar="SAMPLES", 
                        type=int, default=1500, 
                        help="Number of random points to save after 3D scan")
    parser.add_argument("-t", "--trajectory", metavar="TRAJECTORY", 
                        type=str, choices=["random_barrel", "random_sphere", "spiral_barrel"], default="spiral_barrel", 
                        help="Select the scanning trajectory")
    parser.add_argument("-b", "--backgrounds", metavar="BACKGROUNDS", 
                        type=str, nargs='*', default=None, 
                        help="Path for the optional backgound images")
    parser.add_argument("-d", "--depth_ambiguity", metavar="AMBIGUITY", 
                        type=float, default=0.0, 
                        help="Probability of repeating the previous pose with a different f/z setup")
    parser.add_argument("-j", "--jiggle_camera", metavar="JIGGLE", 
                        type=float, default=0.0, 
                        help="Magnitude of the xyz-translation noise of the camera from the trajectory")
    parser.add_argument("-z", "--zoom", metavar="ZOOM", 
                        type=float, default=1.0, 
                        help="Amplification of the automatically calculated minimum cam distance")
    parser.add_argument("-hw", "--resolution", metavar="HW", 
                        type=int, nargs=2, default=None, 
                        help="Resolution of the image")
        
    parser_dataset = parser.add_subparsers(dest='dataset_type', help='Arguments for specific dataset type output')
    parser_dataset.required = False
    
    parser_simple = parser_dataset.add_parser('simple')
    parser_simple.add_argument('--classes', metavar="SIMPLE_CLASSES",
                               type=str, nargs="?", const="opt_classes", default="opt_total", 
                               help='Split size of the dataset')
    parser_simple.add_argument('--split', metavar="SIMPLE_SPLIT",
                               type=float, default=0.8, 
                               help='Split size of the dataset')

    args = parser.parse_args(args)
    return args


def get_camera_positioner(traj):
    if traj == "random_barrel":
        camera_positioning = traj_random_barrel
    elif traj == "spiral_barrel":
        camera_positioning = traj_spiral_barrel
    elif traj == "random_sphere":
        camera_positioning = traj_random_sphere
    else:
        raise RuntimeError("No such trajectory exists")
    return camera_positioning


def get_random_camera_setup(filepaths):
    camera_model_name = rnd.choice(list(filepaths.keys()))
    K, f, G = load_params(filepaths[camera_model_name])
    return K, f, G, camera_model_name


def main(args):

    args = parse_args(args)

    # ARGS ############################################################################
    MESHES = [Path(mesh).resolve() for mesh in args.meshes]
    if len(MESHES) == 1 and MESHES[0].is_dir():
        MESHES = list(MESHES[0].glob("*.[glb obj stl]*"))
    for mesh in MESHES:
        if mesh.suffix not in [".glb", ".obj", ".stl"]:
            raise argparse.ArgumentError(f"{mesh.suffix} is currently not supported")
    if args.mesh_names is not None:
        if len(args.mesh_names) != len(MESHES):
            raise argparse.ArgumentError("The provided names do not match the provided meshes")
        NAMES = args.mesh_names
    else:
        NAMES = [path.stem for path in MESHES]
    PARAMS = [Path(params).resolve() for params in args.params]
    if len(PARAMS) == 1 and PARAMS[0].is_dir():
        PARAMS = list(PARAMS[0].glob("*.[yml yaml]*"))
    for params in PARAMS:
        if not params.match("*.[yml yaml]*"):
            raise argparse.ArgumentError("Parameters filenames must be CAMERANAME.yml")
    PARAMS = {path.stem.split("_", 1)[1]: path for path in PARAMS}
    if len(PARAMS) == 0:
        raise argparse.ArgumentError("No parameter files found!")
    HW = tuple([abs(v) for v in args.resolution]) if args.resolution is not None else None
    
    OUTDIR = Path(args.outdir).resolve()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    BACKGROUNDS = []
    if args.backgrounds:
        for bg_path in args.backgrounds:
            bg_path = Path(bg_path).resolve()
            BACKGROUNDS += [bg_path] if bg_path.is_file() else list(bg_path.iterdir())
    
    EXAMPLES = args.examples
    ANNOT_ID = 1
    CLOUD_SAMPLES = args.samples
    DEPTH_AMBIGUITY = np.clip(args.depth_ambiguity, 0.0, 1.0)
    DEPTH_ZOOM = np.max([0.0, args.zoom])
    ####################################################################################

    # set metric units and meter as base (for depth images)
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.0 if args.meters else 0.001
    # bpy.context.scene.unit_settings.length_unit = 'METERS' if args.meters else 'MILLIMETERS'

    # Blender default file (userpref.blend) contains three objects: Camera, Cube, and Light
    # here we remove the default Cube, leaving only Camera and Light
    [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]
    # de-select/-activate everything
    deselect_everything()


    # CAMERA AND LIGHT SETUP
    cam = bpy.data.objects["Camera"]
    light = bpy.data.objects["Light"]
    # setup world background
    bpy.context.scene.world.use_nodes = False
    bpy.context.scene.world.color = mathutils.Color((0, 0, 0))
    # trajectory
    CAMERA_POSITIONING = get_camera_positioner(args.trajectory)
    cam.data.clip_start = 0.1
    # light
    light_lumens = 25.0

    for i, mesh in enumerate(MESHES):
        
        # TODO this feels very wrong, and should be removed to 
        with suppress_stdout():

            ##################################
            #              SETUP             #
            ##################################

            name = NAMES[i]
            OUTDIR_OBJ = OUTDIR / name
            OUTDIR_OBJ.mkdir(exist_ok=True)

            # TODO with suppress_stdout():
            # load custom object at world origin (this also activates, and thus selects, the object)
            obj = import_objects(mesh, name)
            obj_dims = walk_dimensions(obj)
            # selection is needed for the following command, but the object is already selected when just imported
            bpy.ops.object.location_clear(clear_delta=True)
            # give the object a unique id for annotation and 6D pose with bpycv
            for o in list(walk_children(obj)) + [obj]: 
                o["inst_id"] = ANNOT_ID

            ##################################
            #              SCAN              #
            ##################################

            # absolute paths
            paths = {
                "info":  str(OUTDIR_OBJ / f"{name}.csv"),
                "blend": str(OUTDIR_OBJ / f"{name}.blend"),
                "mesh":  str(OUTDIR_OBJ / f"{name}{mesh.suffix}")
            }

            # prepare dataframe
            info_poses = create_info()
            done_ambiguity = True

            # copy the mesh object
            deselect_everything()
            obj.select_set(True)
            bpy.ops.export_scene.obj(filepath=str(paths["mesh"]), axis_forward='Y', axis_up='Z', use_selection=True, use_materials=True)

            # create output directories
            IMAGES_DIR = Path("images")
            (OUTDIR_OBJ / IMAGES_DIR).mkdir(exist_ok=True)
            DEPTHS_DIR = Path("depths")
            (OUTDIR_OBJ / DEPTHS_DIR).mkdir(exist_ok=True)
            ANNOTS_DIR = Path("annots")
            (OUTDIR_OBJ / ANNOTS_DIR).mkdir(exist_ok=True)
            CLOUDS_DIR = Path("clouds")
            (OUTDIR_OBJ / CLOUDS_DIR).mkdir(exist_ok=True)

            # tqdm prints on stderr, so it won't be suppressed
            for i in tqdm(range(EXAMPLES), desc=name):
                            
                # load intrinsics from file
                K, f, G, camera_model_name = get_random_camera_setup(PARAMS)
                H, W = set_cam_intrinsics(K, f, HW, cam=cam)
                subsamples = min(CLOUD_SAMPLES, H * W)

                # orbit around the object
                origin_distance = np.linalg.norm(G[:, 3])
                visibility_distance = minimum_cam_distance_cam(cam, obj_dims)
                cam_min_radius = 1.25 * max(origin_distance, visibility_distance) / DEPTH_ZOOM
                # setup camera frustum (for Z-buffer)
                cam.data.clip_end = 5.0 *  DEPTH_ZOOM * cam_min_radius  # * ((max(obj_dims) * SCALE) / args.jiggle_camera)
                # setup lighting
                light.data.shadow_soft_size = 2 * cam_min_radius
                light.data.energy = light_lumens * 4*np.pi * cam_min_radius**2

                # relative paths
                rel_paths = {
                    "image": str(IMAGES_DIR / f"{i}.png"),
                    "depth": str(DEPTHS_DIR / f"{i}.png"),
                    "annot": str(ANNOTS_DIR / f"{i}.png"),
                    "cloud": str(CLOUDS_DIR / f"{i}.ply") if subsamples > 0 else ""
                }
                paths.update(rel_paths)

                # apply depth ambiguity if requested
                if done_ambiguity or rnd.uniform(0, 1) > DEPTH_AMBIGUITY:
                    # move camera and light around object and point towards object
                    t = i / max(1, (EXAMPLES - 1))
                    CAMERA_POSITIONING(t, cam_min_radius, obj, cam, args.jiggle_camera)
                    light.location = cam.location
                    
                    # choose a background image
                    if BACKGROUNDS:
                        bg_img = rnd.choice(BACKGROUNDS)
                    done_ambiguity = False
                else:
                    # TODO check sw vs fx
                    z_direction = cam.location.normalized()
                    z_distance_from_obj = cam.location.length
                    depth_ratio = f / z_distance_from_obj
                    new_z_distance_from_obj = z_distance_from_obj * rnd.uniform(1.0, 1.25)
                    new_f = new_z_distance_from_obj * depth_ratio
                    # set iso-perspective setup
                    cam.data.lens = new_f
                    cam.location = z_direction * new_z_distance_from_obj
                    done_ambiguity = True

                ###########################################################
                # DEBUGGING PURPOSES (update the viewport)
                # works only when Blender is in windowed mode (not in CLI)
                #bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                ###########################################################
                
                # # quick and easy render
                # bpy.context.scene.render.filepath = paths["image"]
                # bpy.ops.render.render(write_still=True)
                
                # TODO with suppress_stdout():
                # render image, instance annotation and depth in one line code
                frame_data = bpycv.render_data()

                image = frame_data["image"]
                depth = frame_data["depth"]  # in meters
                annot = frame_data["inst"]
                pose = frame_data["ycb_6d_pose"]  # 6D pose in YCB format
                
                ### 
                # IMAGE, DEPTH and ANNOTATIONS
                ###

                # add a random pre-loaded image
                if BACKGROUNDS:
                    image = add_background(image, annot == ANNOT_ID, bg_img)

                cv2.imwrite(str(OUTDIR_OBJ / rel_paths["image"]), image[:,:,::-1])  # BGR to RGB
                cv2.imwrite(str(OUTDIR_OBJ / rel_paths["depth"]), depth)
                cv2.imwrite(str(OUTDIR_OBJ / rel_paths["annot"]), annot)
                
                ### 
                # POINT CLOUD
                ###
                if subsamples > 0:
                    points, colors = create_cloud(image, depth, K, subsamples)
                    save_cloud(str(OUTDIR_OBJ / rel_paths["cloud"]), points.T, colors)

                # update dataset info
                try:
                    info_poses = update_info(info_poses, paths, camera_model_name, pose, cam.data.lens, (H, W), done_ambiguity)
                except Exception:
                    print("\nObject not present in the render, check the Blender scene file!", file=sys.stderr)
                    break

                # VERY INEFFICIENT
                # reload everything due to bug https://github.com/DIYer22/bpycv/issues/27
                # TODO with suppress_stdout():
                bpy.data.objects.remove(obj, do_unlink=True)
                obj = import_objects(mesh, name)
                bpy.ops.object.location_clear(clear_delta=True)
                for o in list(walk_children(obj)) + [obj]: 
                    o["inst_id"] = ANNOT_ID

            info_poses.reset_index(drop=True)
            info_poses.to_csv(paths["info"])

            # save blend file for debug purposes
            bpy.ops.wm.save_as_mainfile(filepath=paths["blend"])

            # finally remove the object
            bpy.data.objects.remove(obj, do_unlink=True)


    # REFORMAT AS SINGLE DATASET
    if args.dataset_type == "simple":
        raise NotImplementedError("SIMPLE dataset format is not yet implemented")
        if args.classes == "opt_total":
            # mix all NAMES
            classes = []
        elif args.classes == "opt_classes":
            # use all NAMES
            classes = NAMES
        elif isinstance(args.classes, list):
            # check classes within NAMES
            unknown_classes = set(args.classes) - set(NAMES)
            if len(unknown_classes) > 0:
                raise argparse.ArgumentError(f"Classes {unknown_classes} are not known")
            classes = args.classes
        elif isinstance(args.classes, str):
            # check class within NAMES
            if args.classes not in NAMES:
                raise argparse.ArgumentError(f"Class {args.classes} is not known")
            classes = [args.classes]
        else:
            raise argparse.ArgumentError(f"Provided classes are not supported: {args.classes}")
        
        create_simple_dataset(OUTDIR, OUTDIR / f"dataset_{args.dataset_type.upper()}", classes, args.split)
    else:
        print("No dataset type specified, leaving as is")


if __name__ == "__main__":
    args = sys.argv[5:]  # skip the "blender.exe -b --python simulator.py --" arguments (required due to Blender argument bypass)
    main(args)
