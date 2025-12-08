'''
The engine is refactored from the official repo of Video Depth Anything:
https://github.com/DepthAnything/Video-Depth-Anything
'''


import sys
import torch
import torch.nn as nn
from loguru import logger
from utils.dc_utils import read_video_frames, save_video
sys.path.append(".")
from worldbench.third_party.video_depth_anything.video_depth import VideoDepthAnything

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

class DepthEngine(nn.Module):
    def __init__(self, 
                 encoder = 'vits', 
                 **kwargs):
        super(DepthEngine, self).__init__()
        
        self.encoder = encoder
        self.max_len = -1 # -1 means no limit
        self.target_fps = 12
        self.max_res = 400
        self.input_size = 400
        self.target_size = (224, 400)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.build_engine()

    def build_engine(self):
        video_depth_anything = VideoDepthAnything(**model_configs[self.encoder], metric=True)
        video_depth_anything.load_state_dict(torch.load(f'pretrained_models/depth/metric_video_depth_anything_{self.encoder}.pth', map_location='cpu'), strict=True)
        video_depth_anything = video_depth_anything.to(self.device).eval()
        self.engine = video_depth_anything
        logger.info(f"Depth engine built with encoder: {self.encoder}")

    def inference_video(self, video_path):
        # just for bench
        frames, target_fps = read_video_frames(video_path, -1, self.target_fps, self.max_res)
        depths = self.forward(frames)
        save_video(depths, 'tools/ALTest/temp/temp_depth.mp4', fps=self.target_fps)

    def forward(self, frames):
        depths, fps = self.engine.infer_video_depth(frames, self.target_fps, input_size=self.input_size, target_size=self.target_size, device='cuda', fp32=True)
        return depths

if __name__ == "__main__":
    depth_engine = DepthEngine(encoder="vits")
    depth_engine.inference_video("generated_results/magicdrive/video_submission/0f39f34febc84a6689f08599105d1421_gen0/0f39f34febc84a6689f08599105d1421_CAM_FRONT.mp4")

    # check folder existence
    import os
    if not os.path.exists("tools/ALTest/temp"):
        os.makedirs("tools/ALTest/temp")