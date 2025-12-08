import argparse
import torch
from engine import DepthEngine, save_video
from datasets.nusc_clip_dataset import NuscClipDataset
from tqdm import tqdm
from loguru import logger

def main(encoder='vits', num_frames=16):
    # build depth engine
    depth_engine = DepthEngine(encoder=encoder)
    # build dataset
    nusc_clip_dataset = NuscClipDataset(num_frames=16)

    # infer depth for each video clip
    for idx in tqdm(range(len(nusc_clip_dataset))):
        video_info_dict = nusc_clip_dataset[idx]
        pixel_values = video_info_dict['pixel_values']
        with torch.no_grad():
            for cam_id in range(pixel_values.shape[1]):
                depth_maps = depth_engine(pixel_values[:,cam_id])
                # put a breakpoint here to save the depth maps
                breakpoint()
                
                # Just for test
                # save_video(depth_maps, f'tools/ALTest/temp/temp_depth_cam{cam_id}.mp4', fps=12)


                # TODO: Confirm the training format and save the needed depth data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="vits", choices=['vits', 'vitl'], help="the encoder type")
    parser.add_argument("--num_frames", type=int, default=16, help="the number of frames in each video clip")
    args = parser.parse_args()
    main(encoder=args.encoder, num_frames=args.num_frames)