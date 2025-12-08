import os
import json
import numpy as np
import logging
import subprocess
import torch
import random
import re
import imageio
from pathlib import Path
from PIL import Image, ImageSequence
import matplotlib.cm as cm
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

def load_video(video_path, data_transform=None, num_frames=None, return_tensor=True, width=None, height=None):
    """
    Load a video from a given path and apply optional data transformations.

    The function supports loading video in GIF (.gif), PNG (.png), and MP4 (.mp4) formats.
    Depending on the format, it processes and extracts frames accordingly.
    
    Parameters:
    - video_path (str): The file path to the video or image to be loaded.
    - data_transform (callable, optional): A function that applies transformations to the video data.
    
    Returns:
    - frames (torch.Tensor): A tensor containing the video frames with shape (T, C, H, W),
      where T is the number of frames, C is the number of channels, H is the height, and W is the width.
    
    Raises:
    - NotImplementedError: If the video format is not supported.
    
    The function first determines the format of the video file by its extension.
    For GIFs, it iterates over each frame and converts them to RGB.
    For PNGs, it reads the single frame, converts it to RGB.
    For MP4s, it reads the frames using the VideoReader class and converts them to NumPy arrays.
    If a data_transform is provided, it is applied to the buffer before converting it to a tensor.
    Finally, the tensor is permuted to match the expected (T, C, H, W) format.
    """
    if video_path.endswith('.gif'):
        frame_ls = []
        img = Image.open(video_path)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert('RGB')
            frame = np.array(frame).astype(np.uint8)
            frame_ls.append(frame)
        buffer = np.array(frame_ls).astype(np.uint8)
    elif video_path.endswith('.png'):
        frame = Image.open(video_path)
        frame = frame.convert('RGB')
        frame = np.array(frame).astype(np.uint8)
        frame_ls = [frame]
        buffer = np.array(frame_ls)
    elif video_path.endswith('.mp4'):
        import decord
        decord.bridge.set_bridge('native')
        if width:
            video_reader = VideoReader(video_path, width=width, height=height, num_threads=1)
        else:
            video_reader = VideoReader(video_path, num_threads=1)
        frame_indices = range(len(video_reader))
        if num_frames:
            frame_indices = get_frame_indices(
            num_frames, len(video_reader), sample="middle"
            )
        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        buffer = frames.asnumpy().astype(np.uint8)
    else:
        raise NotImplementedError
    
    frames = buffer
    if num_frames and not video_path.endswith('.mp4'):
        frame_indices = get_frame_indices(
        num_frames, len(frames), sample="middle"
        )
        frames = frames[frame_indices]
    
    if data_transform:
        frames = data_transform(frames)
    elif return_tensor:
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    return frames

def tensor_to_gif(tensor: torch.Tensor, save_path: str, fps: int = 8):
    """
    将一个形状为 [T, C, H, W] 的图像张量保存为 gif。
    
    参数:
        tensor: torch.Tensor, 形状 [T, C, H, W]，数值范围 0~1 或 0~255。
        save_path: 输出 gif 文件路径。
        fps: gif 帧率（每秒帧数）。
    """
    # 如果在 GPU 上，先搬到 CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 确保是 float 格式
    tensor = tensor.float()
    
    # 如果数值范围是 0~1，则放大到 0~255
    if tensor.max() <= 1.0:
        tensor = tensor * 255.0

    # 转为 uint8 格式
    tensor = tensor.clamp(0, 255).byte()

    # 调整维度 [T, C, H, W] -> [T, H, W, C]
    frames = tensor.permute(0, 2, 3, 1).numpy()

    # 保存为 gif
    imageio.mimsave(save_path, frames, fps=fps, loop=0)

def tensor_to_compressed_gif(tensor: torch.Tensor, save_path: str, fps: int = 8, resize_ratio=1.0):
    """
    将 [T, C, H, W] 张量保存为压缩 gif。
    参数:
        tensor: torch.Tensor, 值范围 [0,1] 或 [0,255]
        save_path: 输出路径
        fps: 帧率
        resize_ratio: 尺寸缩放比例 (0.5 表示缩小一半)
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.float()

    if tensor.max() <= 1.0:
        tensor = tensor * 255
    tensor = tensor.clamp(0, 255).byte()
    frames = tensor.permute(0, 2, 3, 1).numpy()

    pil_frames = []
    for f in frames:
        img = Image.fromarray(f)
        if resize_ratio < 1.0:
            new_size = (int(img.width * resize_ratio), int(img.height * resize_ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        # 转为调色板模式（256色）
        img = img.convert('P', palette=Image.ADAPTIVE)
        pil_frames.append(img)

    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=True,
        duration=int(1000 / fps),
        loop=0,
        disposal=2
    )

def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False):
    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            writer.append_data(depth_vis)
    else:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])

    writer.close()