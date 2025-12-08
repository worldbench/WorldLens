'''
Imagee Quality Metrics for Novel View Fidelity Evaluation
'''

import os
import json
import subprocess
from pyiqa.archs.musiq_arch import MUSIQ
from loguru import logger
from worldbench.utils import common
from worldbench.utils import video_relative
from tqdm import tqdm
import torch
from torchvision import transforms

def transform(images, preprocess_mode='shorter'):
    if preprocess_mode.startswith('shorter'):
        _, _, h, w = images.size()
        if min(h,w) > 512:
            scale = 512./min(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ), antialias=False)(images)
            if preprocess_mode == 'shorter_centercrop':
                images = transforms.CenterCrop(512)(images)

    elif preprocess_mode == 'longer':
        _, _, h, w = images.size()
        if max(h,w) > 512:
            scale = 512./max(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ), antialias=False)(images)

    elif preprocess_mode == 'None':
        return images / 255.

    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.

class NOVEL_VIEW_FIDELITY_IQ:
    def __init__(self, method_name, 
                 need_preprocessing = False, 
                 generated_data_path = "generated_results", 
                 pretrained_model_path = "pretrained_models/musiq/musiq_spaq_ckpt-358bb6af.pth",
                 **kwargs):
        
        self.method_name = method_name
        self.need_preprocessing = need_preprocessing
        self.generated_data_path = generated_data_path
        self.pretrained_model_path = pretrained_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.init_model()

    def init_model(self):
        if not os.path.isfile(self.pretrained_model_path):
            logger.info(f"Downloading MUSIQ model to {self.pretrained_model_path}")
            os.makedirs(os.path.dirname(self.pretrained_model_path), exist_ok=True)
            wget_command = ['wget', 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth', '-P', os.path.dirname(self.pretrained_model_path)]
            subprocess.run(wget_command, check=True)

        model = MUSIQ(pretrained_model_path=self.pretrained_model_path)
        model.to(self.device)
        model.training = False
        return model

    def calculate_iq_metrics(self, video_list, **kwargs):
        if 'imaging_quality_preprocessing_mode' not in kwargs:
            preprocess_mode = 'longer'
        else:
            preprocess_mode = kwargs['imaging_quality_preprocessing_mode']
        video_results = []
        for video_path in tqdm(video_list):
            images = video_relative.load_video(video_path)
            images = transform(images, preprocess_mode)
            acc_score_video = 0.
            for i in range(len(images)):
                frame = images[i].unsqueeze(0).to(self.device)
                score = self.model(frame)
                acc_score_video += float(score)
            video_results.append({'video_path': video_path, 'video_results': acc_score_video/len(images)})
        average_score = sum([o['video_results'] for o in video_results]) / len(video_results)
        average_score = average_score / 100.
        return average_score, video_results

    def __call__(self, *args, **kwds):
        reconstruction_video_folder = os.path.join(self.generated_data_path, self.method_name, 'reconstruction')
        dims = os.listdir(reconstruction_video_folder)
        result = {}
        for dim in dims:
            result_save_path = os.path.join(self.generated_data_path, self.method_name, 'novel_view_consistency', f'{dim}_image_quality.json')
            os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
            video_list = common.find_videos_in_dir(os.path.join(reconstruction_video_folder, dim), extension=".mp4")
            average_score, video_results = self.calculate_iq_metrics(video_list, **kwds)
            result[dim] = {
                'average_score': average_score,
                'video_results': video_results
            }
            logger.info(f"Method: {self.method_name}, Dim: {dim}, Novel View Fidelity by MUSIQ: {average_score:.4f}")
            json.dump(result[dim], open(result_save_path, 'w'), indent=4)