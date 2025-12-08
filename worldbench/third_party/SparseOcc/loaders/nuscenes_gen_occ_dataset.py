import os
import mmcv
import glob
import torch
import numpy as np
from tqdm import tqdm
from mmdet.datasets import DATASETS
from .nuscenes_occ_dataset import NuSceneOcc
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import DataLoader
from models.utils import sparse2dense
from .ray_metrics import main_rayiou, main_raypq
from .ego_pose_dataset import EgoPoseDataset
from configs.r50_nuimg_704x256_8f import occ_class_names as occ3d_class_names
from configs.r50_nuimg_704x256_8f_openocc import occ_class_names as openocc_class_names

@DATASETS.register_module()
class NuSceneGenOcc(NuSceneOcc):    
    def __init__(self, occ_gt_root, *args, **kwargs):
        super().__init__(occ_gt_root, *args, **kwargs)
        


    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos