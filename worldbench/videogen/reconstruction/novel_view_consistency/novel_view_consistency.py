import os
import clip
from loguru import logger
import subprocess
from tqdm import tqdm
import json
import lpips
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ....utils import common, video_relative
from worldbench.videogen.generation.temporal_consistency.utils import clip_transform

def optical_flow_warp(img1, img2):
    """warp img2 to img1 using optical flow"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow = dis.calc(gray2, gray1, None)
    h, w = gray1.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(img2, map_x, map_y, cv2.INTER_LINEAR)
    return warped
    cv2.imwrite('warped.png', cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
class NOVEL_VIEW_CONSISTENCY:
    def __init__(self, method_name,
                 generated_data_path="generated_results",
                 repeat_times=1,
                 local_save_path=None,
                 repo_or_dir=None,
                 **kwargs):
        self.method_name = method_name
        self.generated_data_path = generated_data_path
        self.repeat_times = repeat_times
        self.local_save_path = local_save_path
        self.repo_or_dir = repo_or_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess, self.lpips_model = self.init_model()
    
    def init_model(self):
        # clip
        if self.local_save_path is not None:
            vit_b_path = self.local_save_path
            if not os.path.isfile(vit_b_path):
                wget_command = ['wget', 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', '-P', os.path.dirname(vit_b_path)]
                subprocess.run(wget_command, check=True)
        else:
            vit_b_path = 'ViT-B/32'
        
        clip_model, preprocess = clip.load(vit_b_path, device=self.device)
        logger.info(f"CLIP model loaded on {self.device}")
        # lpips
        lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()

        return clip_model, preprocess, lpips_model
    
    def to_tensor(self, img):
        t = torch.from_numpy(img).float().permute(2,0,1)[None]
        return t.to(self.device)

    def calculate_single_time(self, video_list):
        sim = 0.0
        cnt = 0
        lpips_score = 0.0
        video_results = []

        image_transform = clip_transform(224)
        for video_path in tqdm(video_list, disable=common.get_rank() > 0):
            video_sim = 0.0
            video_lpips = 0.0
            images = video_relative.load_video(video_path, return_tensor=False)
            for i in tqdm(range(0, images.shape[0]-1)):
                img1 = images[i]
                img2 = images[i+1]
                warped = optical_flow_warp(img1, img2)
                t1 = self.to_tensor(img1)
                t2 = self.to_tensor(warped)
                t1 = image_transform(t1)
                t2 = image_transform(t2)

                # LPIPS
                with torch.no_grad():
                    lp = self.lpips_model(t1*2-1, t2*2-1).mean().item()
                    video_lpips += lp

                # CLIP
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(t1)
                    warped_features = self.clip_model.encode_image(t2)
                    image_features = F.normalize(image_features, dim=-1, p=2)
                    warped_features = F.normalize(warped_features, dim=-1, p=2)
                    frame_sim = max(0.0, F.cosine_similarity(image_features, warped_features).item())
                    video_sim += frame_sim
                    cnt += 1

            sim_per_images = video_sim / (len(images) - 1)
            lpips_per_images = video_lpips / (len(images) - 1)
            sim += video_sim
            lpips_score += video_lpips
            video_results.append({'video_path': video_path, 'video_sim': sim_per_images, 'video_lpips': lpips_per_images})
        sim_per_frame = sim / cnt
        lpips_per_frame = lpips_score / cnt
        return sim_per_frame, lpips_per_frame, video_results

    def __call__(self):

        reconstruction_video_folder = os.path.join(self.generated_data_path, self.method_name, 'reconstruction')
        dims = os.listdir(reconstruction_video_folder)
        rel_save_path = os.path.join(self.generated_data_path, self.method_name, 'novel_view_consistency')
        os.makedirs(rel_save_path, exist_ok=True)
        for dim in dims:
            video_folder_path = os.path.join(reconstruction_video_folder, dim)
            video_list = common.find_videos_in_dir(video_folder_path, extension=".mp4")
            sim_per_frame, lpips_per_frame, video_results = self.calculate_single_time(video_list)
            logger.info(f"Method: {self.method_name}, Dim: {dim}, Subject Consistency (per frame) by DINO: {sim_per_frame:.4f}, LPIPS: {lpips_per_frame:.4f}")
            result = {
                'method_name': self.method_name,
                'dim': dim,
                'subject_consistency_per_frame': sim_per_frame,
                'lpips_per_frame': lpips_per_frame,
                'video_results': video_results
            }
            save_path = os.path.join(rel_save_path, f'{dim}.json')
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=4)
            logger.info(f"Results saved to {save_path}")