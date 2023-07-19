import bpy
import mathutils
import numpy as np
from pathlib import Path

from .camera import Camera, minimum_cam_distance
from .trajectory import standard_traj, _init_random_traj, _init_spiral_traj, _sphere_positioning, _barrel_positioning


def deselect_everything():
    # https://blender.stackexchange.com/questions/99664
    bpy.context.view_layer.objects.active = None
    [obj.select_set(False) for obj in bpy.context.scene.objects]


def _import_object(filepath: Path):
    obj_name = str(filepath.stem)
    if filepath.suffix == ".stl":
        bpy.ops.import_mesh.stl(filepath=str(filepath), axis_forward="Y", axis_up="Z")
    elif filepath.suffix == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(filepath))
    elif filepath.suffix == ".obj":
        bpy.ops.import_scene.obj(filepath=str(filepath), axis_forward="Y", axis_up="Z")
    elif filepath.suffix in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=str(filepath))
        bpy.data.objects["root"].name = obj_name
    else:
        raise RuntimeError(f"File format {filepath.suffix} is not supported!")
    # imported object has same name as the file
    return obj_name


def import_objects(path: Path, name: str = None):
    if path.is_dir(): # TODO useless branch of IF
        par = bpy.data.objects.new("aggregator", None)
        for filepath in path.iterdir():
            obj_name = _import_object(filepath)
            obj = bpy.data.objects[obj_name]
            obj.parent = par
        bpy.scene.objects.link(par)
    else:
        obj_name = _import_object(path)
        par = [bpy.data.objects[o.name] for o in bpy.data.objects if o.name.startswith(obj_name)][0]
        par.name = obj_name
    if name is not None:
        par.name = name
    return par


def look_at(obj_camera, obj_focused, roll=0):
    # use world coordinates to
    direction = obj_focused.location - obj_camera.location
    # point the cameras '-Z' and use its 'Y' as up
    obj_camera.rotation_mode = "QUATERNION"
    obj_camera.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    # # rotate camera if height is bigger than base
    # angle = m.radians(-90) if max(obj_focused.dimensions.x, obj_focused.dimensions.y) < obj_focused.dimensions.z else 0
    if roll is None:
        roll = np.random.uniform(-1, 1) * (2*np.pi)
    obj_camera.rotation_quaternion @= mathutils.Quaternion(mathutils.Vector((0, 0, 1)), roll)


def set_cam_intrinsics(K, f, hw=None, cam=bpy.data.cameras["Camera"]):
    # Blender ignores camera skew, so no  K[0,1]  here
    return set_cam_intrinsics_params(K[0,0], K[1,1], K[0,2], K[1,2], f, hw, cam)


# mimics bpycv.set_cam_intrinsics()
def set_cam_intrinsics_params(fx, fy, cx, cy, f, hw, cam, scale=1):
    if hw is None:
        # resolution is set to give (cx,cy) = (0,0)
        # this implies (shift_x, shift_y) = (0, 0)
        hw = (int(cy*2), int(cx*2))
    if cam is None:
        cam = bpy.data.cameras["Camera"]

    h, w = hw
    if f is not None:
        sw = f / fx * w
        sh = f / fy * h
    else:
        # https://blender.stackexchange.com/a/40835
        # TODO include aspect ratio in  f  calculation
        if h > w:
            sw = 1  # assumption
            sh = fx * h / (fy * w)
            Ky = h / sh
            f = fy / Ky
        elif h < w:
            sw = fy * w / (fx * h)
            sh = 1  # assumption
            Kx = w / sw
            f = fx / Kx
        else:
            sw = 1  # assumption
            sh = 1  # assumption
            Kx = w / sw
            f = fx / Kx
    cam.lens = f
    cam.sensor_fit = 'AUTO'
    cam.sensor_width = sw
    cam.sensor_height = sh
    cam.shift_x = +(0.5 - cx / w)
    cam.shift_y = -(0.5 - cy / h)
    bpy.context.scene.render.resolution_x = int(w / scale)
    bpy.context.scene.render.resolution_y = int(h / scale)
    bpy.context.scene.render.resolution_percentage = scale * 100
    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = 1.0
    return h, w


def walk_children(ob, level=0, max_level=50, type='MESH'):
    if ob.type == type:
        yield ob
    if level < max_level:
        for child in ob.children:
            yield from walk_children(child, level=level + 1)


def walk_dimensions(obj):
    # multiply 3d coord list by matrix
    def np_matmul_coords(coords, matrix, space=None):
        M = (space @ matrix @ space.inverted() if space else matrix).transposed()
        ones = np.ones((coords.shape[0], 1))
        coords4d = np.hstack((coords, ones))
        return (coords4d @ M)[:, :-1]

    # get the global coordinates of all object bounding box corners
    coords = np.vstack([np_matmul_coords(np.array(o.bound_box), o.matrix_world.copy()) for o in walk_children(obj) if o.type == 'MESH'])
    bfl = coords.min(axis=0)  # bottom front left
    tbr = coords.max(axis=0)  # top back right
    dims = tbr - bfl
    return dims[0], dims[1], dims[2]


# fetch the necessary inputs directly from the camera
def minimum_cam_distance_cam(cam, dims):
    h, w = bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x
    f = cam.lens
    fx = f * w / cam.sensor_width
    fy = f * h / cam.sensor_height
    cx = (0.5 - cam.shift_x) * w
    cy = (cam.shift_y + 0.5) * h
    K = Camera.pack_K(fx, fy, cx, cy, 0)
    return minimum_cam_distance(K, f, (h, w), dims)


def traj_random_barrel(t, min_radius, obj_focused, obj_camera, translation_noise=0):
    pos, noise = standard_traj((_init_random_traj, _barrel_positioning), t, min_radius, obj_focused.dimensions.z, translation_noise)
    obj_camera.location = pos
    look_at(obj_camera, obj_focused, roll=None)
    obj_camera.location += mathutils.Vector(noise)


def traj_spiral_barrel(t, min_radius, obj_focused, obj_camera, translation_noise=0):
    pos, noise = standard_traj((_init_spiral_traj, _barrel_positioning), t, min_radius, obj_focused.dimensions.z, translation_noise)
    obj_camera.location = pos
    look_at(obj_camera, obj_focused, roll=None)
    obj_camera.location += mathutils.Vector(noise)


def traj_random_sphere(t, min_radius, obj_focused, obj_camera, translation_noise=0):
    pos, noise = standard_traj((_init_random_traj, _sphere_positioning), t, min_radius, None, translation_noise)
    obj_camera.location = pos
    look_at(obj_camera, obj_focused, roll=None)
    obj_camera.location += mathutils.Vector(noise)
