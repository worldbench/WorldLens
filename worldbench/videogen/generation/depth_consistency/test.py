import os
import torch
import numpy as np
import cv2
from pathlib import Path
from depth_anything_v2.dpt import DepthAnythingV2
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image



DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

dino_model_name = "facebook/dinov2-large"
processor = AutoImageProcessor.from_pretrained(dino_model_name)
dino_model = AutoModel.from_pretrained(
    dino_model_name, 
)


frame_save_folder = "/data_ext/world_benchmark/"

_model = None

def _get_model():
    global _model
    if _model is None:
        model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load("/home/shishir/yixuan/DepthAnything/Depth-Anything-V2/depth_anything_v2_vitl.pth", map_location='cpu'))
        model = model.to(DEVICE).eval()
        _model = model
    return _model
    

def generate_depth_frames(frame, save_dir):
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    model = _get_model()
    
    depth = model.infer_image(frame)
    
    d = depth.copy()
    d[~np.isfinite(d)] = 0
    depth_u8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)  
    
    cv2.imwrite(save_dir, depth_vis)
    

def metrics(method_name):
        video_folder = os.path.join("/data_ext/world_benchmark/", method_name, "video_submission")
        frame_save_dir = os.path.join(frame_save_folder, method_name, "depth_frames")
        if not os.path.exists(frame_save_dir):
            os.makedirs(frame_save_dir, exist_ok=True)
        for sub_folder in os.listdir(video_folder):
            for video in os.listdir(os.path.join(video_folder, sub_folder)):
                video_path = os.path.join(video_folder, sub_folder, video)
                depth_folder = Path(video).stem.split("_",1)[1]
                cap = cv2.VideoCapture(video_path)
                src_fps = cap.get(cv2.CAP_PROP_FPS)
                src_fps = src_fps if src_fps and src_fps > 0 else None
                frame_interval = (1.0/src_fps) if src_fps else None
                target_interval = 1.0/float(12.0)  # Extract frames at 12 Hz
                next_ts = 0.0
                idx = 0
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_interval is not None:
                        current_ts = idx *frame_interval
                    else:
                        current_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    if current_ts+1e-6>=next_ts:
                        save_dir = os.path.join(frame_save_dir, sub_folder, depth_folder, f"frame{frame_idx:04d}.png")
                        generate_depth_frames(frame, save_dir)
                        frame_idx += 1
                        next_ts += target_interval
                cap.release()
        
        depth_feature_dir = "/data_ext/world_benchmark/gt/depth_features"
        if not os.path.exists(depth_feature_dir):
            os.makedirs(depth_feature_dir, exist_ok=True)
        
        
        for depth_folder in os.listdir(frame_save_dir):
            for depth_sub_folder in os.listdir(os.path.join(frame_save_dir, depth_folder)):
                for depth_frame in os.listdir(os.path.join(frame_save_dir, depth_folder, depth_sub_folder)):
                    depth_frame_path = os.path.join(frame_save_dir, depth_folder, depth_sub_folder, depth_frame)
                    depth_image = load_image(depth_frame_path)
                    input = processor(images=depth_image, return_tensors="pt").to(dino_model.device)
                    with torch.inference_mode():
                        outputs = dino_model(**input)
                    pooled_output = outputs.pooler_output
                    
                    depth_freature_subdir = os.path.join(depth_feature_dir, depth_folder, depth_sub_folder)
                    if not os.path.exists(depth_freature_subdir):
                        os.makedirs(depth_freature_subdir, exist_ok=True)
                    depth_save_path = os.path.join(depth_freature_subdir, depth_frame.replace(".png", ".pt"))
                    torch.save(pooled_output, depth_save_path)
        
        L2_distances = []
        
        for freature_folder in os.listdir(depth_feature_dir):
            for feature_sub_folder in os.listdir(os.path.join(depth_feature_dir, freature_folder)):
                feature_list = []
                for feature_file in sorted(os.listdir(os.path.join(depth_feature_dir, freature_folder, feature_sub_folder))):
                    feature_path = os.path.join(depth_feature_dir, freature_folder, feature_sub_folder, feature_file)
                    feature = torch.load(feature_path)
                    feature_list.append(feature)
                for i in range(1, len(feature_list)):
                    dist = torch.norm(feature_list[i]-feature_list[i-1], p=2, dim=-1).item()
                    L2_distances.append(dist)
        
        avg_L2_distance = np.mean(L2_distances)
        return avg_L2_distance
                    
                    
                            
                

if __name__ == "__main__":
    method_name = "gt"
    distance = metrics(method_name)
    print(f"Average L2 distance for {method_name}: {distance}")
                    
                        
                
        
        
        