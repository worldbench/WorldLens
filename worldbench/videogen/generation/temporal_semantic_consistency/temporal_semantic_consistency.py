'''
For the metric, use the <fast3r> environment.
'''
import os
import json
import torch
from tqdm import tqdm
import sys
import easydict
import numpy as np
from loguru import logger
from typing import Optional, Tuple, Dict, Union
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy.spatial import cKDTree
from skimage.morphology import erosion, square
from skimage.measure import label, regionprops
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .engine import build_engine

sys.path.append('.')
from worldbench.utils import common, video_relative

def build_palette_from_video(
    video_rgb: np.ndarray,   # [T,C,H,W], uint8 preferred
    max_colors: Optional[int] = None
) -> np.ndarray:

    assert video_rgb.ndim == 4, "video must be [T,C,H,W]"
    T, C, H, W = video_rgb.shape
    assert C == 3, "Expect RGB with C=3"
    flat = video_rgb.transpose(0,2,3,1).reshape(-1, 3)  # [T*H*W, 3]
    uniq = np.unique(flat, axis=0)
    if max_colors is not None and uniq.shape[0] > max_colors:
        raise ValueError(f"Unique colors={uniq.shape[0]} exceed max_colors={max_colors}. "
                        f"This video likely isn't a clean color map.")
    return uniq.astype(np.uint8)

def labels_from_video_colormap(
    video_rgb: np.ndarray,                  # [T,3,H,W], uint8 优先
    palette: Optional[Union[np.ndarray, Dict[Tuple[int,int,int], int]]] = None,
    approximate: bool = True,
    ignore_color: Optional[Tuple[int,int,int]] = None,
    max_colors_auto: Optional[int] = 1024
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    将色彩编码的视频还原为整数标签序列。
    返回: (masks[T,H,W] int, num_classes, palette_array[K,3])

    - palette 若为 ndarray[K,3]：顺序即类别 id（0..K-1）
      若为 dict{(R,G,B)->id}：以其 id 作为类别编号（会根据dict的id范围构建palette）
    - approximate=True 时使用 KDTree 最近颜色匹配（适合有压缩轻微偏色的 mp4/webm）
    - ignore_color: 若给定，例如 (0,0,0) 表示将该颜色映射为背景/忽略类 id=0（可按需调整）
    """
    assert video_rgb.ndim == 4 and video_rgb.shape[1] == 3
    if video_rgb.dtype != np.uint8:
        # 保障为uint8，避免颜色精度混乱
        video_rgb = np.clip(np.rint(video_rgb), 0, 255).astype(np.uint8)

    T, _, H, W = video_rgb.shape
    pixels = video_rgb.transpose(0,2,3,1).reshape(-1, 3).astype(np.int16)  # [N,3]

    if isinstance(palette, dict):
        # 将dict映射到连续id，并构建palette数组
        items = sorted(palette.items(), key=lambda kv: kv[1])  # 按id排序
        K = max(palette.values()) + 1
        pal_arr = np.zeros((K, 3), dtype=np.uint8)
        for (rgb, cid) in items:
            pal_arr[cid] = np.array(rgb, dtype=np.uint8)

        # 精确或近邻匹配
        if approximate:
            tree = cKDTree(pal_arr.astype(np.float32))
            _, nn = tree.query(pixels.astype(np.float32), k=1)
            labels = nn.astype(np.int32)
        else:
            # 精确查表
            lut = {tuple(map(int, pal_arr[i])): i for i in range(pal_arr.shape[0])}
            labels = np.array([lut.get(tuple(px), 0) for px in pixels], dtype=np.int32)

        num_classes = pal_arr.shape[0]

    elif isinstance(palette, np.ndarray):
        assert palette.ndim == 2 and palette.shape[1] == 3
        pal_arr = palette.astype(np.uint8)
        if approximate:
            tree = cKDTree(pal_arr.astype(np.float32))
            _, nn = tree.query(pixels.astype(np.float32), k=1)
            labels = nn.astype(np.int32)
        else:
            lut = {tuple(map(int, pal_arr[i])): i for i in range(pal_arr.shape[0])}
            labels = np.array([lut.get(tuple(px), 0) for px in pixels], dtype=np.int32)
        num_classes = pal_arr.shape[0]

    else:
        # 未提供palette：自动收集唯一颜色并建立id
        pal_arr = build_palette_from_video(video_rgb, max_colors=max_colors_auto)
        tree = cKDTree(pal_arr.astype(np.float32)) if approximate else None
        if approximate:
            _, nn = tree.query(pixels.astype(np.float32), k=1)
            labels = nn.astype(np.int32)
        else:
            lut = {tuple(map(int, pal_arr[i])): i for i in range(pal_arr.shape[0])}
            labels = np.array([lut.get(tuple(px), 0) for px in pixels], dtype=np.int32)
        num_classes = pal_arr.shape[0]

    # 可选：忽略色映射（例如把(0,0,0)设为背景类0）
    if ignore_color is not None:
        ignore_color = tuple(int(x) for x in ignore_color)
        # 找到该颜色对应的类id
        # 若palette中不存在该颜色，则新增为0类并把原0类整体上移会变复杂，这里只在存在时替换
        match_idx = np.where(np.all(pal_arr == np.array(ignore_color, dtype=np.uint8), axis=1))[0]
        if match_idx.size > 0:
            labels[ np.all(pixels == np.array(ignore_color, dtype=np.int16), axis=1) ] = int(match_idx[0])

    masks = labels.reshape(T, H, W)
    return masks, num_classes, pal_arr

def compute_LFR_interior(masks, num_classes, erosion_k=2):
    T, H, W = masks.shape
    if T < 2:
        return 1.0
    selem = square(max(1, erosion_k))
    interior = np.zeros((T, H, W), dtype=bool)
    for t in range(T):
        interior_t = np.zeros((H, W), dtype=bool)
        for c in range(num_classes):
            cls_mask = (masks[t] == c)
            if cls_mask.any():
                interior_t |= erosion(cls_mask, selem)
        interior[t] = interior_t

    flips = []
    for t in range(T-1):
        inner = interior[t] & interior[t+1]
        if not inner.any():
            continue
        flips.append(np.mean(masks[t][inner] != masks[t+1][inner]))
    if len(flips) == 0:
        return 1.0
    return 1.0 - float(np.mean(flips))


def _class_components(mask, cls_id):
    cls = (mask == cls_id).astype(np.uint8)
    if cls.sum() == 0:
        return None, []
    lab = label(cls, connectivity=1)
    props = regionprops(lab)
    return lab, props

def _overlap_counts(lab_a, props_a, lab_b, props_b):
    if len(props_a) == 0 or len(props_b) == 0:
        return np.zeros((len(props_a), len(props_b)), dtype=np.int64)
    max_b = max(p.label for p in props_b)
    code = lab_a.astype(np.int64) * (max_b + 1) + lab_b.astype(np.int64)
    code = code[(lab_a > 0) & (lab_b > 0)]
    if code.size == 0:
        return np.zeros((len(props_a), len(props_b)), dtype=np.int64)
    uniq, cnt = np.unique(code, return_counts=True)
    O = np.zeros((lab_a.max()+1, lab_b.max()+1), dtype=np.int64)
    O.flat[uniq] = cnt
    idx_a = [p.label for p in props_a]
    idx_b = [p.label for p in props_b]
    return O[np.ix_(idx_a, idx_b)]

def compute_SAC(masks, num_classes, min_iou=0.1):
    T, H, W = masks.shape
    if T < 2:
        return 1.0
    iou_scores = []
    for t in range(T-1):
        m0, m1 = masks[t], masks[t+1]
        for c in range(num_classes):
            lab0, props0 = _class_components(m0, c)
            lab1, props1 = _class_components(m1, c)
            if lab0 is None or lab1 is None:
                if lab0 is not None and lab1 is None:
                    # 未匹配惩罚（m0里多出来的）
                    for p in props0:
                        iou_scores.append((0.0, float(p.area)))
                continue
            areas0 = np.array([p.area for p in props0], dtype=np.float64)
            areas1 = np.array([p.area for p in props1], dtype=np.float64)
            O = _overlap_counts(lab0, props0, lab1, props1)
            A = areas0[:, None]
            B = areas1[None, :]
            union = A + B - O
            with np.errstate(divide='ignore', invalid='ignore'):
                iou = np.where(union > 0, O / union, 0.0)
            if iou.size == 0:
                continue
            cost = 1.0 - iou
            r, cidx = linear_sum_assignment(cost)
            for i, j in zip(r, cidx):
                if iou[i, j] >= min_iou:
                    iou_scores.append((iou[i, j], areas0[i]))
                else:
                    iou_scores.append((0.0, areas0[i]))
            if len(props0) > len(props1):
                unmatched_idx = set(range(len(props0))) - set(r)
                for ui in unmatched_idx:
                    iou_scores.append((0.0, areas0[ui]))
    if not iou_scores:
        return 1.0
    ious, weights = zip(*iou_scores)
    weights = np.asarray(weights, dtype=np.float64)
    return float((np.asarray(ious) * weights).sum() / (weights.sum() + 1e-8))


def compute_CDS(masks, num_classes, eps=1e-8):
    T, H, W = masks.shape
    if T < 2:
        return 1.0
    hists = []
    for t in range(T):
        hist, _ = np.histogram(masks[t], bins=np.arange(num_classes+1))
        p = hist.astype(np.float64)
        p = (p + eps) / (p.sum() + eps * num_classes)
        hists.append(p)
    jsds = []
    for t in range(T-1):
        jsd = jensenshannon(hists[t], hists[t+1])**2
        jsds.append(jsd)
    return float(1.0 - np.mean(jsds))

def tscs_score(
    masks: np.ndarray,       # [T,H,W], int
    num_classes: int,
    w=(0.4, 0.4, 0.2),
    erosion_k: int = 2,
    min_iou: float = 0.1
):
    masks = np.asarray(masks)
    assert masks.ndim == 3, "masks must be [T,H,W]"
    w1, w2, w3 = w
    S_LFR = compute_LFR_interior(masks, num_classes, erosion_k=erosion_k)
    S_SAC = compute_SAC(masks, num_classes, min_iou=min_iou)
    S_CDS = compute_CDS(masks, num_classes)
    TSCS = float(w1 * S_LFR + w2 * S_SAC + w3 * S_CDS)
    return {"TSCS": TSCS, "S_LFR": S_LFR, "S_SAC": S_SAC, "S_CDS": S_CDS,
            "weights": (w1, w2, w3),
            "params": {"erosion_k": erosion_k, "min_iou": min_iou, "num_classes": num_classes}}


class TEMPORAL_SEMANTIC_CONSISTENCY:
    def __init__(self, method_name,
                 generated_data_path="generated_results",
                 need_preprocessing=True,
                 repeat_times=1,
                 weight='/home/alan/AlanLiang/Projects/AlanLiang/WorldBench/Code/OpenSeeD/model_state_dict_swint_51.2ap.pt',
                 conf_file='/home/alan/AlanLiang/Projects/AlanLiang/WorldBench/Code/OpenSeeD/configs/openseed/openseed_swint_lang.yaml',
                 **kwargs):
        self.method_name = method_name
        self.generated_data_path = generated_data_path
        self.need_preprocessing = need_preprocessing
        self.repeat_times = repeat_times
        self.weight = weight
        self.conf_files = [conf_file]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.color_map = [[255, 170, 0], [0, 170, 0], [84, 170, 255], [255, 170, 127], [84, 255, 255], [236, 176, 31], [236, 176, 31], [170, 84, 255], [170, 255, 255], [0, 42, 0], [218, 218, 218], [218, 218, 218],
                         [255, 255, 0], [170, 170, 127], [84, 170, 127], [161, 19, 46], [0, 255, 127], [0, 212, 0], [0, 42, 0], [0, 113, 188]]
        self.palette = {
            i: self.color_map[i] for i in range(len(self.color_map))
        }
        self.engine, self.transform, self.metadata = self.init_engine()

    def init_engine(self):

        args = easydict.EasyDict({
            "conf_files": self.conf_files,
            "weight": self.weight,
        })
        engine, transform, metadata = build_engine(args)
        return engine, transform, metadata
    
    def segmap_to_rgb(self, seg_map, pano_seg_info):
        h, w = seg_map.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        semantic_label_map = np.zeros((h, w), dtype=np.uint8)
        for id, value in enumerate(pano_seg_info):
            rgb_img[seg_map == value['id']] = self.color_map[value['category_id']]
            semantic_label_map[seg_map == value['id']] = value['category_id']

        return rgb_img, semantic_label_map

    def __call__(self):
        if self.need_preprocessing:
            # find all .mp4
            dll_video_list = common.find_videos_in_dir(os.path.join(self.generated_data_path, self.method_name, 'video_submission'), extension=".mp4")
            video_list = []
            for i in range(self.repeat_times):
                sub_video_list = [video_path for video_path in dll_video_list if f"gen{i}" in video_path]
                video_list.extend(sub_video_list)
            for video_path in tqdm(video_list, desc=f"[TSC] Preprocessing videos for {self.method_name}"):
                file_save_path = video_path.replace('video_submission', 'pano_seg_results')
                os.makedirs(os.path.dirname(file_save_path), exist_ok=True)
                if os.path.exists(file_save_path):
                    continue
                images = video_relative.load_video(video_path).cuda()
                images = self.transform(images)
                pano_seg_list = []
                semantic_label_map_list = []
                for image in images:
                    with torch.no_grad():
                        batch_inputs = [{'image': image, 'height': image.shape[1], 'width': image.shape[2]}]
                        outputs = self.engine.forward(batch_inputs)
                        pano_seg = outputs[-1]['panoptic_seg'][0]
                        pano_seg_info = outputs[-1]['panoptic_seg'][1]
                        pano_seg_img, semantic_label_map = self.segmap_to_rgb(pano_seg.cpu().numpy(), pano_seg_info)  # (H, W, 3) np.uint8.
                        pano_seg_list.append(pano_seg_img)
                        semantic_label_map_list.append(semantic_label_map)

                # save video, could be ignored during evaluation
                video_relative.save_video(np.array(pano_seg_list), file_save_path, fps=8)  # (T, H, W, 3) np.uint8

        logger.info(f"Preprocessing done for {self.method_name}!")

        # compute TSC metric
        dll_video_list = common.find_videos_in_dir(os.path.join(self.generated_data_path, self.method_name, 'pano_seg_results'), extension=".mp4")
        video_list = []

        for i in range(self.repeat_times):
            sub_video_list = [video_path for video_path in dll_video_list if f"gen{i}" in video_path]
            video_list.extend(sub_video_list)

        self.compute_tsc_multithread(video_list, max_workers=4)

    def compute_tsc_multithread(self, video_list, max_workers=4):
        num_videos = len(video_list)
        workers = min(max_workers or (os.cpu_count() or 4), num_videos)
        results_by_idx = {}
        TSCS = 0.0
        S_LFR = 0.0
        S_SAC = 0.0
        S_CDS = 0.0
        failed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor, tqdm(total=num_videos, desc=self.method_name) as pbar:
            future_map = {
                executor.submit(self.calculate_single_video, vp): (i, vp)
                for i, vp in enumerate(video_list)
            }

            for fut in as_completed(future_map):
                i, video_path = future_map[fut]
                try:
                    single_score_dict = fut.result()
                    # 补充视频名
                    single_score_dict = dict(single_score_dict)  # 防御性拷贝
                    single_score_dict["video_name"] = os.path.basename(video_path)
                    results_by_idx[i] = single_score_dict

                    # 累加指标（容错：非数值则按0处理）
                    TSCS += float(single_score_dict.get('TSCS', 0.0) or 0.0)
                    S_LFR += float(single_score_dict.get('S_LFR', 0.0) or 0.0)
                    S_SAC += float(single_score_dict.get('S_SAC', 0.0) or 0.0)
                    S_CDS += float(single_score_dict.get('S_CDS', 0.0) or 0.0)
                except Exception as e:
                    failed += 1
                    logger.exception(f"[TSC] Failed on {video_path}: {e}")
                finally:
                    pbar.update(1)

        # 按输入顺序整理结果（过滤失败的）
        ordered_results = [results_by_idx[i] for i in range(num_videos) if i in results_by_idx]
        valid_n = len(ordered_results)
        if valid_n == 0:
            logger.error(f"{self.method_name}: all tasks failed.")
            avg = {"TSCS": 0, "S_LFR": 0, "S_SAC": 0, "S_CDS": 0}
        else:
            avg = {
                "TSCS": TSCS / valid_n,
                "S_LFR": S_LFR / valid_n,
                "S_SAC": S_SAC / valid_n,
                "S_CDS": S_CDS / valid_n,
            }

        final_rel = {"video_results": ordered_results, "average_results": avg}
        logger.info(f"TSC Metric for {self.method_name}: {final_rel['average_results']} (failed {failed}/{num_videos})")

        # 确保输出目录存在并写文件
        out_dir = os.path.join(self.generated_data_path, self.method_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'tscs_metric_results.json')
        json.dump(final_rel, open(out_path, 'w'), indent=4)

        return final_rel

    def calculate_single_video(self, video_path):
        # load video
        pano_seg_video = video_relative.load_video(video_path)
        T, C, H, W = pano_seg_video.shape
        pano_seg_video = pano_seg_video.cpu().numpy().astype(np.uint8)
        # map rgb to label

        masks, num_classes, pal = labels_from_video_colormap(
            pano_seg_video, palette=np.array(self.color_map), approximate=True,
            ignore_color=(255, 255, 0), max_colors_auto=(0.4,0.4,0.2)
        )
        scores = tscs_score(masks, num_classes=num_classes, w=(0.4, 0.4, 0.2),
                            erosion_k=2, min_iou=0.1)
        return scores


if __name__ == "__main__":
    method_name = "gt"
    tsc_metric = TEMPORAL_SEMANTIC_CONSISTENCY(method_name=method_name, checkpoint='pretrained_models/segmentation/sam2.1_hiera_large.pt', 
                                               model_cfg='configs/sam2.1/sam2.1_hiera_l.yaml')
    tsc_metric()