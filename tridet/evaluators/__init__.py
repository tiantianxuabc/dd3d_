# Copyright 2021 Toyota Research Institute.  All rights reserved.
import inspect
import logging
import os

from detectron2.evaluation import COCOEvaluator, SemSegEvaluator

from tridet.evaluators.kitti_3d_evaluator import KITTI3DEvaluator
from tridet.utils.comm import is_distributed

LOG = logging.getLogger('tridet')

AVAILABLE_EVALUATORS = ["coco_evaluator", "kitti3d_evaluator"]


def get_evaluator(cfg, dataset_name, evaluator_name, output_dir):
    assert evaluator_name in AVAILABLE_EVALUATORS, f"Invalid evaluator name: {evaluator_name}."

    distributed = is_distributed()

    if evaluator_name == "coco_evaluator":
        tasks = []
        assert cfg.MODEL.BOX2D_ON
        tasks.append('bbox')
        return COCOEvaluator(dataset_name, tuple(tasks), distributed=distributed, output_dir=output_dir)
    elif evaluator_name == "kitti3d_evaluator":
        return KITTI3DEvaluator(
            dataset_name=dataset_name,
            iou_thresholds=cfg.EVALUATORS.KITTI3D.IOU_THRESHOLDS,
            only_prepare_submission=cfg.EVALUATORS.KITTI3D.ONLY_PREPARE_SUBMISSION,
            distributed=distributed,
            output_dir=output_dir,
        )
