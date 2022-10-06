# dd3d-supplement

## fork from https://github.com/TRI-ML/dd3d.git

## DD3D: "Is Pseudo-Lidar needed for Monocular 3D Object detection?"

Official [PyTorch](https://pytorch.org/) implementation of _DD3D_: [**Is Pseudo-Lidar needed for Monocular 3D Object detection? (ICCV 2021)**](https://arxiv.org/abs/2108.06417),

### Datasets

By default, datasets are assumed to be downloaded in `/data/datasets/<dataset-name>` (can be a symbolic link). The dataset root is configurable by [`DATASET_ROOT`](https://github.com/TRI-ML/dd3d/blob/main/configs/defaults.yaml#L35).

#### KITTI

The KITTI 3D dataset used in our experiments can be downloaded from the [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
For convenience, we provide the standard splits used in [3DOP](https://xiaozhichen.github.io/papers/nips15chen.pdf) for training and evaluation:

The dataset must be organized as follows:

```
<DATASET_ROOT>
    └── KITTI3D
        ├── mv3d_kitti_splits
        │   ├── test.txt
        │   ├── train.txt
        │   ├── trainval.txt
        │   └── val.txt
        ├── testing
        │   ├── calib
        |   │   ├── 000000.txt
        |   │   ├── 000001.txt
        |   │   └── ...
        │   └── image_2
        │       ├── 000000.png
        │       ├── 000001.png
        │       └── ...
        └── training
            ├── calib
            │   ├── 000000.txt
            │   ├── 000001.txt
            │   └── ...
            ├── image_2
            │   ├── 000000.png
            │   ├── 000001.png
            │   └── ...
            └── label_2
                ├── 000000.txt
                ├── 000001.txt
                └── ..
```

## Install

```bash

pip install -r requirements.txt

```

### Pre-trained DD3D models

The DD3D models pre-trained on dense depth estimation using DDAD15M can be downloaded here:
| backbone | download |
| :---: | :---: |
| DLA34 | [model] (链接: https://pan.baidu.com/s/1vpncowhJjhivGNrqMLSTYg 提取码: 4f6q) |

#### (Optional) Eigen-clean subset of KITTI raw.

To train our Pseudo-Lidar detector, we curated a new subset of KITTI (raw) dataset and use it to fine-tune its depth network. This subset can be downloaded [here](https://tri-ml-public.s3.amazonaws.com/github/dd3d/eigen_clean.txt). Each row contains left and right image pairs. The KITTI raw dataset can be download [here](http://www.cvlibs.net/datasets/kitti/raw_data.php).

### Validating installation

To validate and visualize the dataloader (including [data augmentation](./configs/defaults/augmentation.yaml)), run the following:

```bash
python3 visualize_dataloader.py +experiments=dd3d_kitti_dla34 SOLVER.IMS_PER_BATCH=4
```

To validate the entire training loop (including [evaluation](./configs/evaluators) and [visualization](./configs/visualizers)), run the [overfit experiment](configs/experiments/dd3d_kitti_dla34_overfit.yaml) (trained on test set):

```bash
python3 train.py +experiments=dd3d_kitti_dla34
```

|                         experiment                          | backbone | train mem. (GB) | traiqn time (hr) |                                               train log                                                | Box AP (%) | BEV AP (%) |                                                  download                                                   |
| :---------------------------------------------------------: | :------: | :-------------: | :--------------: | :----------------------------------------------------------------------------------------------------: | :--------: | :--------: | :---------------------------------------------------------------------------------------------------------: |
| [config](configs/experiments/dd3d_kitti_dla34_overfit.yaml) |  DLA-34  |        6        |       0.25       | [log](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/dla34-kitti-overfit/logs/log.txt) |   84.54    |   88.83    | [model](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/dla34-kitti-overfit/model_final.pth) |

### Predict

```bash
python3 predict.py +experiments=dd3d_kitti_dla34
```

### Evaluation

One can run only evaluation using the pretrained models:

```bash
python3 train.py +experiments=dd3d_kitti_dla34 EVAL_ONLY=True MODEL.CKPT=<path-to-pretrained-model>
# use smaller batch size for single-gpu
python3 train.py +experiments=dd3d_kitti_dla34  EVAL_ONLY=True MODEL.CKPT=<path-to-pretrained-model> TEST.IMS_PER_BATCH=4
```

## Models

### KITTI

|                     experiment                      | backbone | train mem. (GB) | train time (hr) |                                                  train log                                                  | Box AP (%) | BEV AP (%) |                                                     download                                                     |
| :-------------------------------------------------: | :------: | :-------------: | :-------------: | :---------------------------------------------------------------------------------------------------------: | :--------: | :--------: | :--------------------------------------------------------------------------------------------------------------: |
| [config](configs/experiments/dd3d_kitti_dla34.yaml) |  DLA-34  |       256       |       4.5       | [log](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/26675chm-20210826_083148/logs/log.txt) |   16.92    |   24.77    | [model](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/26675chm-20210826_083148/model_final.pth) |
|  [config](configs/experiments/dd3d_kitti_v99.yaml)  |  V2-99   |       400       |       9.0       | [log](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/4elbgev2-20210825_201852/logs/log.txt) |   23.90    |   32.01    | [model](https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/4elbgev2-20210825_201852/model_final.pth) |

## License

The source code is released under the [MIT license](LICENSE.md). We note that some code in this repository is adapted from the following repositories:

- [detectron2](https://github.com/facebookresearch/detectron2)
- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)

## Reference

```
@inproceedings{park2021dd3d,
  author = {Dennis Park and Rares Ambrus and Vitor Guizilini and Jie Li and Adrien Gaidon},
  title = {Is Pseudo-Lidar needed for Monocular 3D Object detection?},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  primaryClass = {cs.CV},
  year = {2021},
}
```
