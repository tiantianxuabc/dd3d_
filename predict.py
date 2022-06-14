#!/usr/bin/env python3
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os

import cv2

from collections import OrderedDict, defaultdict
import sys
from cv2 import putText
import hydra
import numpy
from requests import put
from sklearn import impute
import torch
import wandb
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torch.cuda import amp
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import detectron2.utils.comm as d2_comm

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluators, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, get_event_storage


import tridet.modeling  # pylint: disable=unused-import
import tridet.utils.comm as comm
from tridet.data import build_test_dataloader, build_train_dataloader
from tridet.data.dataset_mappers import get_dataset_mapper
from tridet.data.datasets import random_sample_dataset_dicts, register_datasets
from tridet.evaluators import get_evaluator
from tridet.modeling import build_tta_model
from tridet.utils.s3 import sync_output_dir_s3
from tridet.utils.setup import setup
from tridet.utils.train import get_inference_output_dir, print_test_results
from tridet.utils.visualization import mosaic, save_vis
from tridet.utils.wandb import flatten_dict, log_nested_dict
from tridet.visualizers import get_dataloader_visualizer, get_predictions_visualizer

from tridet.evaluators.rotate_iou import d3_box_overlap_kernel, rotate_iou_gpu_eval
from tridet.structures.boxes3d import GenericBoxes3D
from tridet.structures.pose import Pose

from contextlib import ExitStack, contextmanager
import torch.nn as nn

from detectron2.utils.visualizer import VisImage, _create_text_labels

from tridet.visualizers.box3d_visualizer import Box3DDataloaderVisualizer, Box3DPredictionVisualizer
from tridet.visualizers.d2_visualizer import D2DataloaderVisualizer, D2PredictionVisualizer
from tridet.visualizers.box3d_visualizer import draw_boxes3d_bev, draw_boxes3d_cam, bev_frustum_crop
from tridet.utils.visualization import change_color_brightness, draw_text, fill_color_polygon

import numpy as np

LOG = logging.getLogger('tridet')


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


@hydra.main(config_path="./configs/", config_name="defaults")
def main(cfg):
    setup(cfg)
    dataset_names = register_datasets(cfg)
    LOG.info("buld the model ...")
    model = build_model(cfg)
 

    checkpoint_file = '/home/phj/Data/dd3d-supplement/demo/model/model_final0.pth' 
    LOG.info("model loading from {}" .format(checkpoint_file))
    Checkpointer(model).load(checkpoint_file)
   
  
    dataset_name = [cfg.DATASETS.TEST.NAME][0]
    metadata = MetadataCatalog.get(dataset_name)
    class_names = [metadata.contiguous_id_to_name[class_id] for class_id in range(len(metadata.thing_classes))]
    
  
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())        

        file_name = "/home/phj/Data/dd3d-supplement/demo/images-self/6_1654579408545226000.bmp"
        # file_name = "/home/phj/Data/dd3d-supplement/demo/images/000041.png"

        file_img = cv2.imread(file_name)   
        file_img = file_img[:, :, ::-1]
        LOG.info("image shape {}".format(file_img.shape))
        img_tensor = torch.from_numpy(file_img.copy())
        img_tensor = img_tensor.permute(2, 0, 1)
        img_intrinsics = torch.Tensor([
                [ 1818.24, 0, 973.382],
                [0, 1817.57, 563.979],
                [0, 0, 1]])
        # img_intrinsics = torch.Tensor([
        #         [ 721.5377, 0, 609.5593],
        #         [0, 721.5377, 172.854],
        #         [0, 0, 1]])
        img_extrinsics = Pose( wxyz=np.float32([0.500, -0.500, 0.504, -0.496]), tvec = np.float32([0.30, -0.07, -0.06]))
        input = {}
        input["intrinsics"] = img_intrinsics
        input["extrinsics"] = img_extrinsics
        input["file_name"] = file_name
        input["image"] = img_tensor
        input["width"] = img_tensor.shape[2]
        input["height"] = img_tensor.shape[1]
        inputs = [input]

        img = cv2.imread(input["file_name"])
        img_copy = img[:, :, :].copy()           
                    
        outputs = model(inputs)            
        result = outputs[0]["instances"].get_fields()
            
        boxes_2d = result["pred_boxes"].tensor.cpu().detach().numpy()
        boxes_2d_scores = result["scores"].cpu().detach().numpy()
        boxes_2d_pred_classes = result["pred_classes"].cpu().detach().numpy()
        for idx, box in enumerate(boxes_2d):                
                score = boxes_2d_scores[idx]
                class_idx = boxes_2d_pred_classes[idx]
                tl = (int(box[0]), int(box[1]))
                br = (int(box[2]), int(box[3]))
                
                color = metadata.thing_colors[class_idx]
                cv2.rectangle(img, tl, br, color, 1)

                label = _create_text_labels([class_idx], [score] if score is not None else None, class_names)[0]
                # bottom-right corner
                text_pos = tuple([box[ 0], box[3]])
                horiz_align = "left"
                lighter_color = change_color_brightness(tuple([c / 255. for c in color]), brightness_factor=0.8)
                H, W = img.shape[:2]
                default_font_size = max(np.sqrt(H * W) // 90, 10)
                height_ratio = (box[ 3] - box[ 1]) / np.sqrt(H * W)
                font_size = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * default_font_size)
                V = VisImage(img=img)
                draw_text(V.ax, label, text_pos, font_size=font_size, color=lighter_color, horizontal_alignment=horiz_align)
                img = V.get_image()



        pred_boxes3d = result["pred_boxes3d"]


        pred_3d  = pred_boxes3d.vectorize().detach().cpu().numpy()   
       

        for idx, box in enumerate(boxes_2d):                
                score = boxes_2d_scores[idx]
                class_idx = boxes_2d_pred_classes[idx]
                tl = (int(box[0]), int(box[1]))
                br = (int(box[2]), int(box[3]))
                dist_z = pred_3d[idx][6]
                dist_z = "%.1f"%dist_z
                cv2.putText(img, str(dist_z), tl, 0, 1, (0, 130, 250), 1)

        d3_box = GenericBoxes3D.from_vectors(pred_3d)    
        img_3d = draw_boxes3d_cam(img_copy, pred_boxes3d, boxes_2d_pred_classes, metadata, input["intrinsics"], result["scores_3d"], render_labels=True)
        viz_image, bev_obj = draw_boxes3d_bev( d3_box,
            intrinsics=input['intrinsics'],
            extrinsics=input['extrinsics'],
            class_ids=boxes_2d_pred_classes,
            image_width=img_copy.shape[1],
            metadata=None,
            color=(0, 0, 255))
        viz_image = cv2.rotate(viz_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Crop the BEV image to show only frustum.
        viz_image = bev_frustum_crop(viz_image)
        cv2.imshow("image_object_bev", viz_image)
        cv2.imshow("image_object_3d", img_3d)
        cv2.imshow("image_object_2d", img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            sys.exit()



if __name__ == '__main__':
    LOG.info("start")
    main()  # pylint: disable=no-value-for-parameter
    LOG.info("DONE.")
