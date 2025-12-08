from typing import Any, Dict, Tuple

import torch
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class BEVFusionSegmentation:
    """
    获取BEVFusion论文中使用的分割标注

    """
    def __init__(
        self,
        dataset_root,
        map_grid_conf,
        classes=('drivable_area', 'ped_crossing',
                 'walkway', 'stop_line', 'carpark_area',
                 'divider'),
    ):
        super().__init__()
        xbound = map_grid_conf['xbound']
        ybound = map_grid_conf['ybound']
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def show_seg(self, labels):

        mask_colors = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(len(self.classes) + 1)
        ]
        img = np.zeros((200, 400, 3))

        for index, mask_ in enumerate(labels):
            color_mask = mask_colors[index]
            mask_ = mask_.astype(bool)
            img[mask_] = color_mask

        return img

    def __call__(self, results):

        location, ego2global_translation, ego2global_rotation = \
            results['location'], results['ego2global_translation'], results['ego2global_rotation']
        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        # masks = masks.transpose(0, 2, 1)
        masks = masks.astype(np.bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        semantic_masks = labels  # 这里已经进行了one_hot编码了
        num_cls = semantic_masks.shape[0]
        indices = np.arange(1, num_cls + 1).reshape(-1, 1, 1)
        semantic_indices = np.sum(semantic_masks * indices, axis=0)
        semantic_indices = np.where(semantic_indices > 5, 6, semantic_indices)

        results.update({
            'semantic_map': torch.from_numpy(semantic_masks),
            'semantic_indices': torch.from_numpy(semantic_indices).long(),
        })
        return results