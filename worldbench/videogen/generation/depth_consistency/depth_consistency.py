'''
https://github.com/DepthAnything/Video-Depth-Anything
'''

import os
import torch
import numpy as np
import easydict
import imageio
import matplotlib.cm as cm
from loguru import logger   
from tqdm import tqdm
import random
import json
from .depth_mae.utils.dc_utils import read_video_frames, save_video

import sys
sys.path.append(".")
from worldbench.third_party.video_depth_anything.video_depth import VideoDepthAnything
from worldbench.utils import common
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

def select_scene(select_num = 100):
    with open('tools/ALTest/selected_scenes.json', 'r') as f:
        selected_scenes = json.load(f)
    return selected_scenes

class DEPTH_CONSISTENCY:
    def __init__(self, method_name, 
                 encoder = 'vits', 
                 need_preprocessing = False,
                 generated_data_path = "generated_results", 
                 pretrained_model_path = "pretrained_models/depth",
                 **kwargs):
        
        self.max_len = -1 # -1 means no limit
        self.target_fps = -1
        self.max_res = 400
        self.input_size = 400
        self.target_size = (450, 800)

        self.encoder = encoder
        self.method_name = method_name
        self.need_preprocessing = need_preprocessing
        self.generated_data_path = generated_data_path
        self.pretrained_model_path = pretrained_model_path
        self.selected_scenes = select_scene(100)
        self.build_engine()

    def build_engine(self):
        video_depth_anything = VideoDepthAnything(**model_configs[self.encoder])
        video_depth_anything.load_state_dict(torch.load(f'{self.pretrained_model_path}/metric_video_depth_anything_{self.encoder}.pth', map_location='cpu'), strict=True)
        video_depth_anything = video_depth_anything.to('cuda').eval()
        self.engine = video_depth_anything

    def inference_video(self, video_path):
        # just for bench
        frames, target_fps = read_video_frames(video_path, -1, self.target_fps, self.max_res)

        depths, fps = self.engine.infer_video_depth(frames, self.target_fps, input_size=self.input_size, target_size=self.target_size, device='cuda', fp32=True)
        return depths

    def save_depth(self, frames, grayscale=False):
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()

        # 避免0导致log问题
        eps = 1e-6
        frames_log = np.log(frames + eps)
        d_min, d_max = frames_log.min(), frames_log.max()

        depth_rel = []
        for i in range(frames.shape[0]):
            depth = frames_log[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            if grayscale:
                depth_vis = depth_norm
            else:
                depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)
            depth_rel.append(depth_vis)
        return np.stack(depth_rel, axis=0)

    def __call__(self):
        if self.need_preprocessing:
            # generate depth for each video
            # submission_video_path = os.path.join(self.generated_data_path, self.method_name, "video_submission")
            # video_list = []
            # submission_video_dirs = os.listdir(submission_video_path)
            # for scene in submission_video_dirs:
            #     if 'gen0' in scene:
            #         scene_path = os.path.join(submission_video_path, scene)
            #         video_list += common.find_videos_in_dir(scene_path)
            video_list = [os.path.join(f'generated_results/{self.method_name}/video_submission', scene.replace('.gif', '.mp4')) for scene in self.selected_scenes]
            logger.info(f"Total {len(video_list)} videos found for depth generation.")
            for video_path in tqdm(video_list):
                depth_save_path = video_path.replace("video_submission", "predicted_depth").replace(".mp4", ".gif")
                if os.path.exists(depth_save_path):
                    continue
                depths = self.inference_video(video_path)
                os.makedirs(os.path.dirname(depth_save_path), exist_ok=True)
                imageio.mimsave(depth_save_path, self.save_depth(depths), fps=6, loop=0)

            # save key frames for each video
            pass



if __name__ == "__main__":
    pass
    # depth = DEPTH(method_name="magicgrive", encoder="vits")
    # depth()