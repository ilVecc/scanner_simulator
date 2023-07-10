import numpy as np
rnd = np.random.default_rng()


def _barrel_positioning(radius, height, t_angle, t_height):
    angle = 2 * np.pi * t_angle
    r = radius * (np.sin(np.pi * t_height) * 0.25 + 0.75)
    x, y = r * np.cos(angle), r * np.sin(angle)
    z = height * (t_height - 0.5)  # wrt z=0 (the same as the object origin)
    return x, y, z


def _sphere_positioning(radius, height, t_angle, t_height):
    t_z = 2 * t_height - 1
    angle = 2 * np.pi * t_angle
    #r = 2 * radius * m.sin(m.acos(1 - 2 * t_height))  # look up on wikipedia that sin(acos(x)) = sqrt(1-x^2)
    r = 2 * radius * np.sqrt(1 - t_z ** 2)
    x, y, z = r * np.cos(angle), r * np.sin(angle), radius * t_z
    return x, y, z


def _init_spiral_traj(t, translation_noise):
    noise = translation_noise * np.clip(rnd.normal(scale=0.25, size=(3)), -1.0, 1.0)
    return t, t, noise


def _init_random_traj(t, translation_noise):
    t_angle = rnd.uniform(0, 1)
    t_height = rnd.uniform(0, 1)
    noise = translation_noise * np.clip(rnd.normal(scale=0.25, size=(3)), -1.0, 1.0)
    return t_angle, t_height, noise


def standard_traj(traj_funs, t, radius, height, translation_noise):
    traj_init, traj_prog = traj_funs
    t_angle, t_height, noise = traj_init(t, translation_noise)
    position = traj_prog(radius, height, t_angle, t_height)
    return position, noise.tolist()

