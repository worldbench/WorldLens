import torch
import cv2

import numpy as np

from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from mmdet.datasets.builder import PIPELINES


import warnings
warnings.filterwarnings('ignore')


@PIPELINES.register_module()
class LSS_Segmentation(object):
    """
    获取LSS论文中使用的分割标注

    """

    def __init__(self,
                 map_grid_conf=None
                 ):


        dx, bx, nx = self.gen_dx_bx(map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

    @staticmethod
    def gen_dx_bx(xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        return dx, bx, nx

    def __call__(self, results):
        egopose = results['egopose']
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        bin_img = np.zeros((self.nx[0], self.nx[1]))

        inst_ = results['inst']
        for inst in inst_:
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(bin_img, [pts], 1.0)


        results.update({
            'semantic_indices': torch.from_numpy(bin_img)
        })

        return results
