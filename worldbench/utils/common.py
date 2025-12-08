import os
import torch

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

# find all .mp4 files in a directory
def find_videos_in_dir(directory, extension=".mp4"):
    video_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                video_list.append(os.path.join(root, file))
    return video_list