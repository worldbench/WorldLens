import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import argparse
from pathlib import Path

def compute_box_corners(box):

    x, y, z, dx, dy, dz, yaw = box
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])

    x_corners = np.array([dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2])
    y_corners = np.array([dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2])
    z_corners = np.array([dz/2, dz/2, dz/2, dz/2, -dz/2, -dz/2, -dz/2, -dz/2])
    
    corners = np.stack([x_corners, y_corners, z_corners], axis=1)
    corners = np.dot(corners, R.T) + np.array([x, y, z])
    return corners

def project_box_to_image(box, cam_info):

    corners = compute_box_corners(box)
    sensor2lidar_R = cam_info['sensor2lidar_rotation']
    sensor2lidar_T = cam_info['sensor2lidar_translation']
    
    R_lidar2cam = sensor2lidar_R.T
    t_lidar2cam = - R_lidar2cam @ sensor2lidar_T    
    corners_cam = (R_lidar2cam @ corners.T).T + t_lidar2cam

    K = cam_info['cam_intrinsic']
    corners_proj = (K @ corners_cam.T).T
    pixels = corners_proj[:, :2] / (corners_proj[:, 2:3] + 1e-6)
    
    if np.any(corners_proj[:,2:3]<0):
        return -1,-1,-1,-1

    xmin = int(np.min(pixels[:, 0]))
    ymin = int(np.min(pixels[:, 1]))
    xmax = int(np.max(pixels[:, 0]))
    ymax = int(np.max(pixels[:, 1]))
    
    return xmin, ymin, xmax, ymax

def crop_and_save_object(track_frame_info, custom_frame_info, out_root):

    scene_token = track_frame_info['scene_token']
    timestamp = track_frame_info['timestamp']
    gt_boxes = track_frame_info['gt_boxes']
    gt_names = track_frame_info['gt_names']
    instance_inds = track_frame_info['instance_inds'] 
    cams = custom_frame_info['cams']

    
    for idx, box in enumerate(gt_boxes):
        instance_name = gt_names[idx]
        if instance_name not in ['car', 'truck', 'bus', 'pedestrian', 'construction_vehicle', 'traiycle']:
            continue

        instance_id = str(instance_inds[idx]).zfill(7) +'_'+ f'{gt_names[idx]}'
        obj_folder = os.path.join(out_root, instance_id)
        
        for cam_name, cam_info in cams.items():

            image_path = cam_info['data_path']
            img = cv2.imread(image_path)
            
            try:
                xmin, ymin, xmax, ymax = project_box_to_image(box, cam_info)
            except Exception as e:
                # print(f"Fail {image_path}: {e}")
                continue
            
            h, w = img.shape[:2]
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            if xmin >= xmax or ymin >= ymax:
                # print(f"Fail: {image_path}")
                continue
            
            cropped = img[ymin:ymax, xmin:xmax]
            
            # os.makedirs(scene_folder, exist_ok=True)
            os.makedirs(obj_folder, exist_ok=True)
            save_path = os.path.join(obj_folder, f"{timestamp}_{cam_name}.jpg")
            cv2.imwrite(save_path, cropped)

def process_gt_track_info(gt_track_info, custom_info, out_root):

    for time_stamp, custom_frame_info in enumerate(tqdm(custom_info)):
        track_frame_info = gt_track_info[custom_frame_info['token']]
        track_frame_info['timestamp'] = time_stamp
        crop_and_save_object(track_frame_info, custom_frame_info, out_root)

def load_annotations(ann_file):
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    new_data_infos = dict()
    for di in data_infos:
        new_data_infos[di['token']] = di
    return new_data_infos

def extract_fg_objects(custom_info_path):
    gt_track_info = load_annotations('data/nuscenes_track/ada_track_infos_val.pkl')
    custom_info = pickle.load(open(custom_info_path, 'rb'))
    custom_info = list(sorted(custom_info["infos"], key=lambda e: e["timestamp"]))

    custom_pkl_idx = str(Path(custom_info_path).stem).split('_')[-1].split('.')[0]
    output_root = Path(custom_info_path).parent / "nuscenes_cropped" / f"{custom_pkl_idx}"
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        process_gt_track_info(gt_track_info, custom_info, output_root)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process Nuscenes 2D Tracking Data')
    parser.add_argument('--custom_info_path', type=str, zdefault='custom_info.pkl', help='Path to custom info file')
    args = parser.parse_args()

    gt_track_info = load_annotations('/home/alan/AlanLiang/Projects/AlanLiang/WorldBench-Perception/data/nuscenes_track/ada_track_infos_val.pkl')
    custom_info = pickle.load(open(args.custom_info_path, 'rb'))['infos']

    custom_pkl_idx = str(Path(args.custom_info_path).stem).split('_')[-1].split('.')[0]
    output_root = Path(args.custom_info_path).parent / "nuscenes_cropped" / f"{custom_pkl_idx}"
    output_root.mkdir(parents=True, exist_ok=True)
    process_gt_track_info(gt_track_info, custom_info, output_root)
