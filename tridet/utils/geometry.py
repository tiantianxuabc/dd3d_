# Copyright 2021 Toyota Research Institute.  All rights reserved.
from ast import Return
import logging

import math
import cv2
import numpy as np
import torch
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix

LOG = logging.getLogger(__name__)

PI = 3.14159265358979323846
EPS = 1e-7


def euler_to_rotation(euler_angle):
    """
    Parameters
    ----------
    euler_angle: np.array(yaw pitch roll) yaw-pitch-roll angles
    yaw z-axis rotation matix
    [[cos(a), -sin(a), 0]
     [sin(a),  cos(a), 0]
     [     0,       0, 1]]
    pitch x-axis rotation matrix
    [[ cos(ptich),    0,  sin(pitch)]
     [          0,    1,           0]
     [-sin(pitch),    0,  cos(pitch)]]
    roll y-axis rotation matrix
    [[     1,          0,           0]
     [     0   cos(roll),  -sin(roll)]
     [     0,  sin(roll),   cos(roll)]]
    rotation matrix
    [[coa(yaw)cos(ptich) cos(yaw)sin(pitch)sin(roll) - sin(yaw)cos(roll)  cos(yaw)sin(pitch)cos(roll) + sin(yaw)sin(roll)]
     [sin(yaw)cos(pitch) sin(yaw)sin(pitch)sin(roll) + cos(yaw)cos(roll)  sin(yaw)sin(pitch)cos(roll) - cos(yaw)sin(roll)]
     [       -sin(ptich)                    cos(pitch)sin(roll)                            cos(ptich)cos(roll)           ]]
    (1, 3), Batch of euler_angle

    Return
    (3*3)
    """
    yaw, pitch, roll  = euler_angle[0], euler_angle[1], euler_angle[2]
    R_Z = np.array([[math.cos(yaw), -math.sin(yaw), 0]
                    [math.sin(yaw),  math.cos(yaw), 0]
                    [            0,              0, 1]])
    R_X = np.array([[math.cos(pitch), -math.sin(pitch),  0]
                    [math.sin(pitch),  math.cos(pitch),  0]
                    [            0,              0,      1]])
    R_Y = np.array([[ 1,               0,               0 ]
                    [ 0,   math.cos(roll),  -math.sin(roll)]
                    [ 0,   math.sin(roll),   math.cos(roll)]])
    return np.multiply(np.multiply(R_Z, R_X), R_Y)



def allocentric_to_egocentric(quat, proj_ctr, inv_intrinsics):
    """
    Parameters
    ----------
    quat: Tensor
        (N, 4). Batch of (allocentric) quaternions.

    proj_ctr: Tensor
        (N, 2). Projected centers. xy coordninates.

    inv_intrinsics: [type]
        (N, 3, 3). Inverted intrinsics.
    """
    R_obj_to_local = quaternion_to_matrix(quat)
    
    # ray == z-axis in local orientaion
    ray = unproject_points2d(proj_ctr, inv_intrinsics)
    z = ray / ray.norm(dim=1, keepdim=True)
    

    # gram-schmit process: local_y = global_y - global_y \dot local_z
    y = z.new_tensor([[0., 1., 0.]]) - z[:, 1:2] * z
    y = y / y.norm(dim=1, keepdim=True)
    x = torch.cross(y, z, dim=1)
    

    # local -> global
    R_local_to_global = torch.stack([x, y, z], dim=-1)

    # obj -> global
    R_obj_to_global = torch.bmm(R_local_to_global, R_obj_to_local)

    egocentric_quat = matrix_to_quaternion(R_obj_to_global)

    # Make sure it's unit norm.
    quat_norm = egocentric_quat.norm(dim=1, keepdim=True)
    if not torch.allclose(quat_norm, torch.as_tensor(1.), atol=1e-3):
        LOG.warning(
            f"Some of the input quaternions are not unit norm: min={quat_norm.min()}, max={quat_norm.max()}; therefore normalizing."
        )
        egocentric_quat = egocentric_quat / quat_norm.clamp(min=EPS)

    return egocentric_quat


def homogenize_points(xy):
    """
    Parameters
    ----------
    xy: Tensor
        xy coordinates. shape=(N, ..., 2)
        E.g., (N, 2) or (N, K, 2) or (N, H, W, 2)

    Returns
    -------
    Tensor:
        1. is appended to the last dimension. shape=(N, ..., 3)
        E.g, (N, 3) or (N, K, 3) or (N, H, W, 3).
    """
    # NOTE: this seems to work for arbitrary number of dimensions of input
    pad = torch.nn.ConstantPad1d(padding=(0, 1), value=1.)
    return pad(xy)


def project_points3d(Xw, K):
    _, C = Xw.shape
    assert C == 3
    uv, _ = cv2.projectPoints(
        Xw, np.zeros((3, 1), dtype=np.float32), np.zeros(3, dtype=np.float32), K, np.zeros(5, dtype=np.float32)
    )
    return uv.reshape(-1, 2)


def unproject_points2d(points2d, inv_K, scale=1.0):
    """
    Parameters
    ----------
    points2d: Tensor
        xy coordinates. shape=(N, ..., 2)
        E.g., (N, 2) or (N, K, 2) or (N, H, W, 2)

    inv_K: Tensor
        Inverted intrinsics; shape=(N, 3, 3)

    scale: float, default: 1.0
        Scaling factor.

    Returns
    -------
    Tensor:
        Unprojected 3D point. shape=(N, ..., 3)
        E.g., (N, 3) or (N, K, 3) or (N, H, W, 3)
    """
    points2d = homogenize_points(points2d)
    siz = points2d.size()
    points2d = points2d.view(-1, 3).unsqueeze(-1)  # (N, 3, 1)
    unprojected = torch.matmul(inv_K, points2d)  # (N, 3, 3) x (N, 3, 1) -> (N, 3, 1)
    unprojected = unprojected.view(siz)

    return unprojected * scale
