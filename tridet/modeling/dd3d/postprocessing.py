# Copyright 2021 Toyota Research Institute.  All rights reserved.
from collections import OrderedDict, defaultdict
from pprint import pprint

import torch
from pytorch3d.transforms import transform3d as t3d
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix

from detectron2.structures import Instances

from tridet.layers import bev_nms
from tridet.structures.boxes3d import GenericBoxes3D
from tridet.structures.pose import Pose


def _indices_to_mask(indices, size):
    mask = indices.new_zeros(size, dtype=torch.bool)
    mask[indices] = True
    return mask


def sample_bev_nms(instances, poses, category_key='pred_classes', iou_threshold=0.3):
    boxes3d_global = []
    for _instances, pose in zip(instances, poses):
        # pose_SO
        box3d_vec = _instances.pred_boxes3d.vectorize()
        quat, tvec, wlh = box3d_vec[:, :4], box3d_vec[:, 4:7], box3d_vec[:, 7:10]
        R = quaternion_to_matrix(quat)
        rotation = t3d.Rotate(R=R.transpose(1, 2), device=quat.device)
        translation = t3d.Translate(tvec, device=quat.device)
        tfm_SO = rotation.compose(translation)

        # pose_WS
        quat, tvec = quat.new(pose.quat.elements), tvec.new(pose.tvec)
        R = quaternion_to_matrix(quat)
        rotation = t3d.Rotate(R=R.transpose(0, 1), device=quat.device)
        translation = t3d.Translate(tvec.unsqueeze(0), device=quat.device)
        tfm_WS = rotation.compose(translation)

        # boxes in global frame.
        tfm_WO = tfm_SO.compose(tfm_WS)
        pose_WO = tfm_WO.get_matrix().transpose(1, 2)
        rotation = pose_WO[:, :3, :3]
        quat = matrix_to_quaternion(rotation)
        tvec = pose_WO[:, :3, -1]

        boxes3d_global.append(torch.hstack([quat, tvec, wlh]))

    boxes3d_global = torch.vstack(boxes3d_global)
    boxes3d_global = GenericBoxes3D(boxes3d_global[:, :4], boxes3d_global[:, 4:7], boxes3d_global[:, 7:])

    _ids = torch.cat([x.get(category_key) for x in instances])
    scores = torch.cat([x.scores_3d for x in instances])
    keep = bev_nms(boxes3d_global, scores, iou_threshold, pose_cam_global=Pose(), class_idxs=_ids)
    return keep, boxes3d_global


def get_group_idxs(sample_tokens, num_images_per_sample, inverse=False):
    grouped_idxs = defaultdict(list)
    for idx, token in enumerate(sample_tokens):
        grouped_idxs[token].append(idx)
    group_sizes = {token: len(idxs) for token, idxs in grouped_idxs.items()}

    if not all([siz == num_images_per_sample for siz in group_sizes.values()]):
        pprint(group_sizes)
        raise ValueError("Group sizes does not match with 'num_images_per_sample'.")

    token_to_idxs = OrderedDict(grouped_idxs)
    if not inverse:
        return token_to_idxs
    else:
        idx_to_token = OrderedDict()
        for token, idxs in token_to_idxs.items():
            for idx in idxs:
                idx_to_token[idx] = token
        return idx_to_token
