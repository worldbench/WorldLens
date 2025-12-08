import os
import clip
from loguru import logger
import subprocess
from tqdm import tqdm
import json
import numpy as np
import torch
from torch.nn import functional as F
from ....utils import common, video_relative
from .utils import clip_transform

class TEMPORAL_CONSISTENCY:
    def __init__(self, method_name,
                 generated_data_path="generated_results",
                 repeat_times=1,
                 local_save_path=None,
                 **kwargs):
        self.method_name = method_name
        self.generated_data_path = generated_data_path
        self.repeat_times = repeat_times
        self.local_save_path = local_save_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = self.init_model()
    
    def init_model(self):
        if self.local_save_path is not None:
            vit_b_path = self.local_save_path
            if not os.path.isfile(vit_b_path):
                wget_command = ['wget', 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', '-P', os.path.dirname(vit_b_path)]
                subprocess.run(wget_command, check=True)
        else:
            vit_b_path = 'ViT-B/32'
        
        clip_model, preprocess = clip.load(vit_b_path, device=self.device)
        logger.info(f"CLIP model loaded on {self.device}")
        return clip_model, preprocess
    
    def get_video_image_feature(self, video_path, image_transform):
        images = video_relative.load_video(video_path)
        images = image_transform(images)

        images = images.to(self.device)
        image_features = self.clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1, p=2)
        return image_features

    def calculate_acm(self, image_features):
        video_sim = 0.0
        cnt_per_video = 0
        for i in range(len(image_features)):
            image_feature = image_features[i].unsqueeze(0)
            if i == 0:
                first_image_feature = image_feature
            else:
                sim_pre = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
                sim_fir = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
                cur_sim = (sim_pre + sim_fir) / 2
                
                video_sim += cur_sim
                cnt_per_video += 1
            former_image_feature = image_feature
        sim_per_image = video_sim / (len(image_features) - 1)

        return video_sim, sim_per_image, cnt_per_video

    def calculate_tji(self, image_features):
        v = (image_features[1:] - image_features[:-1]).norm(dim=1)                 # [T-1]
        a = (image_features[2:] - 2*image_features[1:-1] + image_features[:-2]).norm(dim=1)  # [T-2]
        tji = (a / (0.5*(v[1:] + v[:-1]) + 1e-8)).mean()
        tji_score = torch.exp(-0.5 * tji)
        return tji, tji_score

    def rel_score(self, x_gen, x_gt, w=4.0):
        x_gen = torch.tensor(x_gen)
        x_gt = torch.tensor(x_gt)
        return torch.exp(-w * torch.abs(torch.log((x_gen+1e-8)/(x_gt+1e-8))))

    def motion_align(self, F, G, beta=3.0):
        vF = F[1:] - F[:-1]; vG = G[1:] - G[:-1]
        vF_n = vF / (vF.norm(dim=1, keepdim=True) + 1e-8)
        vG_n = vG / (vG.norm(dim=1, keepdim=True) + 1e-8)
        MDA = (vF_n * vG_n).sum(1).mean().clamp(-1,1)
        sF = vF.norm(dim=1); sG = vG.norm(dim=1)
        MRS = torch.exp(-beta * torch.mean(torch.abs(torch.log((sF+1e-8)/(sG+1e-8)))))
        return MDA, MRS

    def calculate_single_video(self, video_path, image_transform):
        # gen
        image_features = self.get_video_image_feature(video_path, image_transform)  # [T, C]
        # acm: Adjacent Cosine Mean
        video_sim, sim_per_image, cnt_per_video = self.calculate_acm(image_features)
        # tji: Temporal Jitter Index
        tji, tji_score = self.calculate_tji(image_features)
        # gt
        gt_video_path = video_path.replace(f'{self.method_name}', 'gt')
        gt_image_features = self.get_video_image_feature(gt_video_path, image_transform)  # [T, C]
        _, gt_sim_per_image, _ = self.calculate_acm(gt_image_features)
        gt_tji, _ = self.calculate_tji(image_features)
        S_acm = self.rel_score(sim_per_image, gt_sim_per_image)
        S_tji = self.rel_score(tji, gt_tji)
        
        # motion alignment
        mda, mrs = self.motion_align(image_features, gt_image_features, beta=0.5)
        ts = sim_per_image * (S_acm*S_tji).sqrt() * mrs.sqrt()
        # check nan
        if torch.isnan(ts):
            ts = torch.tensor(0.0)

        single_rel_dict = {
                'video_path': video_path, 
                'video_results': sim_per_image,
                'video_sim': video_sim,
                'cnt_per_video': cnt_per_video,
                "TJI": tji.item(),
                "TJI_score": tji_score.item(),
                "ts": ts.item()}
        return single_rel_dict

    def calculate_single_time(self, video_list):
        sim = 0.0
        tji_all_score = 0.0
        ts = 0.0
        cnt = 0
        video_results = []
        image_transform = clip_transform(224)
        for video_path in tqdm(video_list, disable=common.get_rank() > 0):
            single_rel_dict = self.calculate_single_video(video_path, image_transform)
            video_results.append(single_rel_dict)

            tji_all_score += single_rel_dict['TJI_score']
            sim += single_rel_dict['video_sim']
            cnt += single_rel_dict['cnt_per_video']
            ts += single_rel_dict['ts']

        sim_per_frame = sim / cnt
        tji_per_frame = tji_all_score / len(video_list)
        ts_per_frame = ts / len(video_list)

        return sim_per_frame, tji_per_frame, ts_per_frame, video_results

    def __call__(self):
        generated_video_folder_path = f'{self.generated_data_path}/{self.method_name}/video_submission'
        scene_path_list = os.listdir(generated_video_folder_path)
        rel_save_path = os.path.join(self.generated_data_path, self.method_name, 'temporal_consistency')
        os.makedirs(rel_save_path, exist_ok=True)
        for i in range(self.repeat_times):
            video_list = []
            sub_scene_path_list = [os.path.join(generated_video_folder_path, video_path) for video_path in scene_path_list if f'_gen{i}' in video_path]
            for scene_path in sub_scene_path_list:
                video_list += common.find_videos_in_dir(scene_path, extension=".mp4")

            sim_per_frame, tji_per_frame, ts_per_frame, video_results = self.calculate_single_time(video_list)
            logger.info(f"Method: {self.method_name}, Repeat: {i}, Temporal Consistency (per frame) by CLIP: {sim_per_frame:.4f}, TJI score: {tji_per_frame:.4f}, TS: {ts_per_frame:.4f}")
            result = {
                'method_name': self.method_name,
                'repeat': i,
                'temporal_consistency_per_frame': sim_per_frame,
                'tji_per_frame': tji_per_frame,
                'ts_per_frame': ts_per_frame,
                'video_results': video_results
            }
            save_path = os.path.join(rel_save_path, f'repeat_{i}.json')
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=4)
            logger.info(f"Results saved to {save_path}")
