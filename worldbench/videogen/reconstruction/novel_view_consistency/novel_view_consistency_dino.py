import os
import clip
from loguru import logger
import subprocess
from tqdm import tqdm
import json

import torch
from torch.nn import functional as F
from ....utils import common, video_relative
from worldbench.videogen.generation.subject_consistency.utils import dino_transform

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
        self.dino_model = self.init_model()
    
    def init_model(self):
        if self.repo_or_dir is None:
            self.repo_or_dir = 'worldbench/third_party/dino'
            if not os.path.exists(self.repo_or_dir):
                subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/dino', self.repo_or_dir], check=True)

        if self.local_save_path is not None:
            dino_vit_path = self.local_save_path
            if not os.path.isfile(dino_vit_path):
                wget_command = ['wget', 'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth', '-P', os.path.dirname(dino_vit_path)]
                subprocess.run(wget_command, check=True)

        else:
            dino_vit_path = 'ViT-B/32'

        load_dict = {
                'repo_or_dir': self.repo_or_dir,
                'path': dino_vit_path, 
                'model': 'dino_vitb16',
                'source': 'local',
                }
        dino_model = torch.hub.load(**load_dict).to(self.device)
        logger.info(f"DINO model loaded on {self.device}")
        return dino_model
    
    def calculate_single_time(self, video_list):
        sim = 0.0
        cnt = 0
        video_results = []

        image_transform = dino_transform(224)
        for video_path in tqdm(video_list, disable=common.get_rank() > 0):
            video_sim = 0.0
            images = video_relative.load_video(video_path)
            images = image_transform(images)
            for i in range(len(images)):
                with torch.no_grad():
                    image = images[i].unsqueeze(0)
                    image = image.to(self.device)
                    image_features = self.dino_model(image)
                    image_features = F.normalize(image_features, dim=-1, p=2)
                    if i == 0:
                        first_image_features = image_features
                    else:
                        sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                        sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                        cur_sim = (sim_pre + sim_fir) / 2
                        video_sim += cur_sim
                        cnt += 1
                former_image_features = image_features
            sim_per_images = video_sim / (len(images) - 1)
            sim += video_sim
            video_results.append({'video_path': video_path, 'video_results': sim_per_images})
        sim_per_frame = sim / cnt
        return sim_per_frame, video_results

    def __call__(self):

        reconstruction_video_folder = os.path.join(self.generated_data_path, self.method_name, 'reconstruction')
        dims = os.listdir(reconstruction_video_folder)
        rel_save_path = os.path.join(self.generated_data_path, self.method_name, 'novel_view_consistency')
        os.makedirs(rel_save_path, exist_ok=True)
        for dim in dims:
            video_folder_path = os.path.join(reconstruction_video_folder, dim)
            video_list = common.find_videos_in_dir(video_folder_path, extension=".mp4")
            sim_per_frame, video_results = self.calculate_single_time(video_list)
            logger.info(f"Method: {self.method_name}, Dim: {dim}, Subject Consistency (per frame) by DINO: {sim_per_frame:.4f}")
            result = {
                'method_name': self.method_name,
                'dim': dim,
                'subject_consistency_per_frame': sim_per_frame,
                'video_results': video_results
            }
            save_path = os.path.join(rel_save_path, f'{dim}.json')
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=4)
            logger.info(f"Results saved to {save_path}")