import os
import torch
import numpy as np
import easydict
from loguru import logger

class NOVEL_VIEW_FIDELITY_FVD:
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

    def get_custom_video_feature(self, args, preprocess_func):
        save_folder = os.path.join(self.generated_data_path, args.model_name, 'reconstruction_fvd')
        os.makedirs(save_folder, exist_ok=True)
        all_video_feature = {}
        reconstruction_video_folder = os.path.join(self.generated_data_path, args.model_name, 'reconstruction')
        dims = os.listdir(reconstruction_video_folder)
        for dim in dims:
            save_file_path = os.path.join(save_folder, f'{dim}.npy')
            if os.path.exists(save_file_path):
                video_feature = np.load(save_file_path)
                all_video_feature[dim] = video_feature
                continue

            video_folder=os.path.join(reconstruction_video_folder, dim)
            args.video_folder = video_folder
            video_feature = preprocess_func(args)
            all_video_feature[dim] = video_feature
            np.save(save_file_path, video_feature)
        return all_video_feature

    def __call__(self):
        if self.need_preprocessing:
            from worldbench.videogen.generation.perceptual_fidelity.fvd.utils.get_fvd_features_nusc import preprocess_gt
            from .utils.get_fvd_features_custom import preprocess_custom
            args = easydict.EasyDict({
                "model_name": "gt",
                "data_info": self.data_info_path,
                "method": self.method,
                "load_from": self.pretrained_model_path
            })
            # process gt
            args.model_name = 'gt'
            gt_video_feature = self.get_custom_video_feature(args, preprocess_custom)

            # process custom
            args.model_name = self.method_name
            method_video_feature = self.get_custom_video_feature(args, preprocess_custom)
            logger.info("Preprocess generated FVD features done.")

        from worldbench.videogen.generation.perceptual_fidelity.fvd.videogpt.fvd import frechet_distance
        result = {}
        for dim_name, video_feature in gt_video_feature.items():
            fvd = frechet_distance(torch.from_numpy(video_feature), torch.from_numpy(method_video_feature[dim_name]))
            result.update({f'FVD_{dim_name}': fvd})

        logger.info(result)
        return result