import os
import mmcv
import argparse
import numpy as np
from copy import deepcopy
from PIL import Image
from moviepy.editor import VideoFileClip
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_video(token, video_root, gen_idx, key_frame_index, view_order, 
                  token_info_dict, out_size, save_dir):
    # Prepare paths and load video info
    test_infos = []
    video_dir = os.path.join(video_root, f"{token}_gen{gen_idx}")

    # make new info for this video frames
    this_kf_info = deepcopy(token_info_dict[token])
    this_kf_info['scene_token'] = this_kf_info['scene_token'] + f"_gen{gen_idx}"
    vid_infos = [this_kf_info]
    for _ in range(len(key_frame_index) - 1):
        this_kf_info = deepcopy(token_info_dict[this_kf_info['next']])
        this_kf_info['scene_token'] = this_kf_info['scene_token'] + f"_gen{gen_idx}"
        vid_infos.append(this_kf_info)

    # handle each view
    for view in view_order:
        cam_save_dir = os.path.join(save_dir, view)
        video_name = os.path.join(video_dir, f"{token}_{view}.mp4")
        if not os.path.exists(video_name):
            print(f"There is no video for {token}")
            return []

        clip = VideoFileClip(video_name)

        # iterate frames and save images
        frames = clip.iter_frames()
        kf_idx = 0
        for idx, frame in enumerate(frames):
            if idx not in key_frame_index:
                continue

            this_kf_info = vid_infos[kf_idx]
            # save image
            this_img = Image.fromarray(frame).resize(out_size)
            img_name = os.path.splitext(os.path.basename(this_kf_info['cams'][view]['data_path']))[0] + ".png"
            save_path = os.path.join(cam_save_dir, img_name)
            assert not os.path.exists(save_path)
            this_img.save(save_path)
            this_kf_info['cams'][view]['data_path'] = save_path
            kf_idx += 1
    else:
        test_infos += vid_infos

    return test_infos


def decode_video(args):

    view_order = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
    ]
    key_frame_index = [0, 6, 12]
    out_size = (1600, 900)

    data = mmcv.load(args.data_info)

    # we sort infos for sequential read
    token_info_dict = {}
    first_frame_tokens = []
    for info in data['infos']:
        token_info_dict[info['token']] = info
        if info['prev'] == "":
            first_frame_tokens.append(info['token'])
    print(f"loaded scenes: {len(first_frame_tokens)}")

    save_paths = []
    for gen_idx in range(args.generation_times):
        save_dir = os.path.join('generated_results', args.model_name, f"gen_samples_{gen_idx}")
        if os.path.exists(save_dir):
            return
        # make save dir
        for view in view_order:
            cam_save_dir = os.path.join(save_dir, view)
            os.makedirs(cam_save_dir, exist_ok=True)

        test_infos = []
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_token = {executor.submit(process_video, token, args.vid_root, gen_idx, key_frame_index, view_order, token_info_dict, out_size, save_dir): token for token in first_frame_tokens}

            for future in as_completed(future_to_token):
                token = future_to_token[future]
                try:
                    result = future.result()
                    if result:
                        test_infos.extend(result)
                except Exception as exc:
                    print(f"{token} generated an exception: {exc}")

        print(f"test infos for gen {gen_idx} length = {len(test_infos)}")
        save_path = os.path.join('generated_results', args.model_name, f"nuscenes_infos_temporal_val_3keyframes_gen{gen_idx}.pkl")
        mmcv.dump({"infos": test_infos, "metadata": data['metadata']}, save_path)
        save_paths.append(save_path)

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    # vid_root
    arg.add_argument("--vid_root", type=str, default="generated_results/magicdrive/video_submission")
    arg.add_argument("--model_name", type=str, default="magicdrive")
    arg.add_argument("--data_info", type=str, default="data/nuscenes_mmdet3d_2/nuscenes_infos_temporal_val_3keyframes.pkl")
    arg.add_argument("--generation_times", type=int, default=4)
    arg.add_argument("--max_workers", type=int, default=8)

    args = arg.parse_args()

    decode_video(args)
