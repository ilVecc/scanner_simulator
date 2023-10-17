import bpy
import bpycv
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


def look_at(cam, obj, roll=0):
    K = bpycv.camera_utils.get_cam_intrinsic(cam)
    cx, cy, f = K[0][2], K[1][2], cam.data.lens
    h, w = bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y

    # point the cameras '-Z' and use its 'Y' as up
    direction = obj.location - cam.location
    track_quat = direction.to_track_quat('-Z', 'Y').normalized()    

    # track_quat rotates camera's -Z towards the object's location
    # align_quat rotates the image center optical ray to the camera's principal point optical ray 
    # roll_quat rotates the camera along Z

    # TODO check CameraRotationOffset.blend for insights on how to automatize this
    #      these values have been obtained by trial-and-error
    align_quat = mathutils.Euler(np.deg2rad([-0.9, -6.8, 0]), "ZYX").to_quaternion()
    # # find the rotation that brings the object origin to the image center
    # Qinv = track_quat.to_matrix().transposed() @ K.inverted()
    # v_center = Qinv @ mathutils.Vector((w/2, h/2, 1))
    # v_origin = Qinv @ mathutils.Vector((cx, cy, 1))
    # cvcam_to_bcam_quat = mathutils.Quaternion((1, 0, 0), np.pi)

    # rotate camera if height is bigger than base
    # angle = m.radians(-90) if max(obj_focused.dimensions.x, obj_focused.dimensions.y) < obj_focused.dimensions.z else 0
    
    # roll camera about it's center axis
    roll = np.random.uniform(-1, 1) * (2*np.pi) if roll is None else roll
    roll_quat = mathutils.Quaternion((0, 0, 1), roll)

    # set the rotation
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = track_quat @ roll_quat @ align_quat


def set_cam_intrinsics(K, f, hw=None, cam=bpy.data.objects["Camera"]):
    # Blender ignores camera skew, so no  K[0,1]  here, but we still need to check it
    sk = np.arctan(-K[0,0] / K[0,1])
    if np.allclose(sk, np.pi/2, atol=0.5):
        print("Image will appear flipped vertically")
    elif np.allclose(sk, -np.pi/2, atol=0.5):
        print("Image will appear correct vertically")
    else:
        raise Exception(f"Blender doesn't work with non-orthogonal skew ({np.rad2deg(sk)})")

    return set_cam_intrinsics_params(K[0,0], K[1,1], K[0,2], K[1,2], f, hw, cam.data)
    # bpycv.camera_utils.set_cam_intrinsic(cam, K, hw)
    # return bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x


# mimics bpycv.set_cam_intrinsics()
def set_cam_intrinsics_params(fx, fy, cx, cy, f, hw, cam=bpy.data.cameras["Camera"]):
    # https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/camera/CameraUtility.py#L226
    if hw is None:
        # resolution is set to give (cx,cy) = (w/2,h/2), which implies (shift_x,shift_y) = (0,0)
        hw = (int(cy*2), int(cx*2))
    h, w = hw

    # some preliminary stuff
    pixel_aspect_ratio = fx / fy
    image_aspect_ratio = w / h
    horizontal_sensor_fit = (image_aspect_ratio >= pixel_aspect_ratio)  # width > height ?
    
    # work out f if needed
    if f is None:
        f = fy / h if horizontal_sensor_fit else fx / w
    cam.lens = f

    # these are values of a sensor with one dimension set to 1 and the other calculated accordingly
    # sw = f / fx * w
    # sh = f / fy * h
    # these unfortunately don't work in blender, which calculates things differently, so
    if horizontal_sensor_fit:
        cam.sensor_fit = "HORIZONTAL"
        cam.sensor_width = w / h
    else:
        cam.sensor_fit = "VERTICAL"
        cam.sensor_height = h / w
    
    # here some magic to work out the shift
    view_fac_in_px = w if horizontal_sensor_fit else h * pixel_aspect_ratio
    cam.shift_x = - (cx - (w - 1) / 2) / view_fac_in_px
    cam.shift_y = + (cy - (h - 1) / 2) / view_fac_in_px * pixel_aspect_ratio

    bpy.context.scene.render.resolution_x = w
    bpy.context.scene.render.resolution_y = h
    bpy.context.scene.render.resolution_percentage = 100
    pixel_aspect_x, pixel_aspect_y = (1, pixel_aspect_ratio) if fx >= fy else (1/pixel_aspect_ratio, 1)
    bpy.context.scene.render.pixel_aspect_x = pixel_aspect_x
    bpy.context.scene.render.pixel_aspect_y = pixel_aspect_y

    return (h, w)


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
    f = cam.data.lens
    K = bpycv.camera_utils.get_cam_intrinsic(cam)
    K = np.array(K)
    return minimum_cam_distance(K, f, (h, w), dims)


def traj_random_barrel(t, min_radius, obj_focused, obj_camera, translation_noise=0):
    pos, noise = standard_traj((_init_random_traj, _barrel_positioning), t, min_radius, obj_focused.dimensions.z, translation_noise)
    obj_camera.location = pos
    look_at(obj_camera, obj_focused, roll=None)
    # obj_camera.location += mathutils.Vector(noise)


def traj_spiral_barrel(t, min_radius, obj_focused, obj_camera, translation_noise=0):
    pos, noise = standard_traj((_init_spiral_traj, _barrel_positioning), t, min_radius, obj_focused.dimensions.z, translation_noise)
    obj_camera.location = pos
    look_at(obj_camera, obj_focused, roll=None)
    # obj_camera.location += mathutils.Vector(noise)


def traj_random_sphere(t, min_radius, obj_focused, obj_camera, translation_noise=0):
    pos, noise = standard_traj((_init_random_traj, _sphere_positioning), t, min_radius, None, translation_noise)
    obj_camera.location = pos
    look_at(obj_camera, obj_focused, roll=None)
    # obj_camera.location += mathutils.Vector(noise)


