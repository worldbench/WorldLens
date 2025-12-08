from PIL import Image
from typing import Tuple
from loguru import logger
import mmcv
import torch
import torchvision
from torch.utils.data import Dataset

class NuscClipDataset(Dataset):
    def __init__(self, num_frames = 16):

        self.num_frames = num_frames
        self.load_interval = 1
        self.ann_file = 'data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train.pkl'
        self.dataset_root = 'data/nuscenes'
        self.target_size = (400, 224)
        self.img_collate_param = {}
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor()
            ]
        )
        self.data_infos = self.load_clips()

    def load_clips(self):
        data = mmcv.load(self.ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.clip_infos = self.build_clips(data_infos, data['scene_tokens'])
        logger.info(f"Totally {len(self.clip_infos)} clips of {self.num_frames} frames are loaded for training")
        return data_infos

    def build_clips(self, data_infos, scene_tokens):

        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}

        all_clips = []
        for sid, scene in enumerate(scene_tokens):
            first_frames = range(0,len(scene)-16,16)
            for start in first_frames:
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + 16]]
                all_clips.append(clip)

        return all_clips


    def __len__(self):
        return len(self.clip_infos)

    def __getitem__(self, idx):
        return self.prepare_train_data(idx)

    def prepare_train_data(self, index):
        """This is called by `__getitem__`
        """
        frames = self.get_data_info(index)
        for frame in frames:
            frame['image_paths'] = [s.replace("../data/nuscenes",self.dataset_root) for s in frame['image_paths']]
        ret_dicts = self.load_frames(frames)
        return ret_dicts

    def get_data_info(self, index):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[index]
        frames = self.load_clip(clip)
        return frames

    def load_clip(self, clip):
        frames = []
        for frame in clip:
            frame_info = self.extract_data_info(frame)
            frames.append(frame_info)
        return frames
    
    def extract_data_info(self, index):
        info = self.data_infos[index]
        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info["location"],
        )
        data["image_paths"] = []
        for _, camera_info in info["cams"].items():
            data["image_paths"].append(camera_info["data_path"])        
        
        return data
    
    def load_frames(self, frames):
        if None in frames:
            return None
        examples = []
        for frame in frames:
            example = self.pipeline(frame)
            examples.append(example)
        ret_dicts = collate_fn_single_clip(examples, **self.img_collate_param)
        return ret_dicts

    def img_transform(self, img):
        img = img.resize(self.target_size)
        return img
        
    def pipeline(self, results):
        filename = results["image_paths"]
        images = []

        for name in filename:
            img = Image.open(name)
            new_img = self.img_transform(img)
            images.append(self.compose(new_img).permute(1,2,0))  # H, W, 3

        results["img"] = torch.stack(images)

        return results

def collate_fn_single_clip(
    examples: Tuple[dict, ...]):
    pixel_values = torch.stack([example["img"].data for example in examples])
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()
    ret_dict = {
        "pixel_values": pixel_values.numpy()*255}
    
    return ret_dict

if __name__ == "__main__":
    dataset = NuscClipDataset()
    print(len(dataset))
    data = dataset[0]
    print(data['pixel_values'].shape)
    # torch.Size([16, 6, 3, 224, 400])