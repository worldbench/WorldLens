import os
import torch
import numpy as np
import easydict
from loguru import logger

class FVD:
    def __init__(self, method_name, 
                 need_preprocessing = False, 
                 method = 'videogpt', 
                 generated_data_path = "generated_results", 
                 pretrained_model_path = "pretrained_models/fvd/i3d_pretrained_400.pt",
                 data_info_path = "data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_track2_eval.pkl",
                 **kwargs):
        
        self.method_name = method_name
        self.need_preprocessing = need_preprocessing
        self.method = method
        self.generated_data_path = generated_data_path
        self.pretrained_model_path = pretrained_model_path
        self.data_info_path = data_info_path  

    def __call__(self):
        if self.need_preprocessing:
            from .utils.get_fvd_features_nusc import preprocess_gt
            from .utils.get_fvd_features_gen import preprocess_gen
            args = easydict.EasyDict({
                "model_name": "gt",
                "data_info": self.data_info_path,
                "method": self.method,
                "load_from": self.pretrained_model_path
            })
            preprocess_gt(args)
            logger.info("Preprocess ground truth FVD features done.")
            args.model_name = self.method_name
            preprocess_gen(args)
            logger.info("Preprocess generated FVD features done.")

        file1 = os.path.join(self.generated_data_path, 'gt', 'fvd', 'fvd_feats.npy')
        file2 = os.path.join(self.generated_data_path, args.model_name, 'fvd', 'fvd_feats.npy')
        d1 = np.load(file1)
        d2 = np.load(file2)

        from .videogpt.fvd import frechet_distance
        fvd = frechet_distance(torch.from_numpy(d1), torch.from_numpy(d2))
        logger.info(f"FVD: {fvd}")
        return {
            "FVD": fvd
        }