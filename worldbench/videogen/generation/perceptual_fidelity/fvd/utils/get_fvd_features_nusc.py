import os
import argparse
import mmcv
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as TF

# import sys
# sys.path.append(f"{os.path.dirname(__file__)}/../")

class ImgPathDataset(torch.utils.data.Dataset):
    def __init__(self, listoflist, transform) -> None:
        self.data_list = listoflist
        self.transform = transform

    def __getitem__(self, idx):
        li = self.data_list[idx]
        imgs = []
        for img in li:
            img = Image.open(img).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        return imgs

    def __len__(self):
        return len(self.data_list)


def top_center_crop(img, target_size):
    fH, fW = target_size  # see mmdet3d, this is inversed
    newW, newH = img.size
    crop_h = newH - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    img = img.crop(crop)
    return img


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4)

    return x


# get fid score
VIEW_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]
IN_SIZE = (900, 1600)
TARGET_SIZE = (224, 400)
resize_ratio = 0.25


def get_clips(data_info):
    data = mmcv.load(data_info)
    scene_tokens = data['scene_tokens']
    token_info_dict = {}
    for info in data['infos']:
        token_info_dict[info['token']] = info
    clip_list = []
    for view in VIEW_ORDER:
        for tokens in scene_tokens:
            clip = []
            for token in tokens:
                ori_cam_img_path = token_info_dict[token]['cams'][view]['data_path']
                cam_img_path = ori_cam_img_path.replace('../data', './data')
                clip.append(cam_img_path)
            clip_list.append(clip)
    return clip_list


def preprocess_gt(args):
    out_dir = os.path.join('generated_results', args.model_name, 'fvd')
    os.makedirs(out_dir, exist_ok=True)
    save_file_path = os.path.join(out_dir, "fvd_feats.npy")
    if os.path.exists(save_file_path):
        return

    # original data
    _size = (int(IN_SIZE[0] * resize_ratio), int(IN_SIZE[1] * resize_ratio))
    transforms = TF.Compose([
        TF.Resize(_size, interpolation=TF.InterpolationMode.BICUBIC),
        lambda x: top_center_crop(x, target_size=TARGET_SIZE),
        TF.ToTensor(),
    ])
    clip_list = get_clips(args.data_info)
    dataset = ImgPathDataset(clip_list, transforms)

    dims = 400
    batch_size = 50
    num_workers = 4
    device = "cuda"
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    if args.method == 'styleganv':
        try:
            from wb_metric.generation_metric.fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
        except ImportError:
            print("Please install wb_metric with `pip install wb_metric`")
    elif args.method == 'videogpt':
        from ..videogpt.fvd import load_i3d_pretrained
        from ..videogpt.fvd import get_fvd_logits as get_fvd_feats
        from ..videogpt.fvd import frechet_distance

    i3d = load_i3d_pretrained(device=device, load_from=args.load_from)

    pred_arr = np.empty((len(dataset), dims))
    start_idx = 0
    first_batch = True
    for batch in tqdm(dataloader, ncols=80):
        if first_batch:
            print(f"size of batch: {batch.shape}")
            first_batch = False

        with torch.no_grad():
            batch = trans(batch)
            feats1 = get_fvd_feats(batch, i3d=i3d, device=device)

        pred = feats1.cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    np.save(save_file_path, pred_arr)

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--model_name", type=str, default="magicdrive")
    arg.add_argument("--data_info", type=str, default="data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_track2_eval.pkl")
    arg.add_argument("--method", type=str, default="videogpt")
    arg.add_argument("--load_from", type=str, default="pretrained_models/fvd/i3d_pretrained_400.pt")
    args = arg.parse_args()

    preprocess_gt(args)
