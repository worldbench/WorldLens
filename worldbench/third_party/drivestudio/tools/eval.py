from typing import List, Optional
from omegaconf import OmegaConf
import os
import numpy as np
import time
import json
import wandb
import logging
import argparse

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views
)
from utils.visualization import depth_visualizer, to8b
import imageio

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

@torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True
):
    trainer.set_eval()

    logger.info("Evaluating Pixels...")
    if dataset.test_image_set is not None and cfg.render.render_test:
        logger.info("Evaluating Test Set Pixels...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.test_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
        )

        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/test/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_test_{current_time}.json"
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/test_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/test_set_{step}_{args.render_video_postfix}.mp4"
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_test_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=2,
            verbose=True,
            save_images=False,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/test/" + k: wandb.Image(v)})
        # optionally dump raw depth for test set
        if getattr(args, "save_raw_depth", None) and getattr(args, "include_depth", False):
            base_out = args.save_raw_depth
            out_base = base_out if os.path.isabs(base_out) else os.path.join(cfg.log_dir, base_out)
            raw_dir = os.path.join(out_base, f"test_set_{step}")
            os.makedirs(raw_dir, exist_ok=True)
            if "depths" in render_results:
                for i, depth in enumerate(render_results["depths"]):
                    cam_name = render_results["cam_names"][i]
                    np.save(os.path.join(raw_dir, f"{i:05d}_{cam_name}.npy"), depth)
            logger.info(f"Saved raw test depth to {raw_dir}")
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()

    if cfg.render.render_full:
        logger.info("Evaluating Full Set...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.full_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
            save_image_pairs=True
        )

        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/full/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            full_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_full_{current_time}.json"
            with open(full_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {full_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_img_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=cfg.render.fps,
            verbose=True,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/full/" + k: wandb.Image(v)})
        # optionally dump raw depth for full set
        if getattr(args, "save_raw_depth", None) and getattr(args, "include_depth", False):
            base_out = args.save_raw_depth
            out_base = base_out if os.path.isabs(base_out) else os.path.join(cfg.log_dir, base_out)
            raw_dir = os.path.join(out_base, f"full_set_{step}")
            os.makedirs(raw_dir, exist_ok=True)
            if "depths" in render_results:
                for i, depth in enumerate(render_results["depths"]):
                    cam_name = render_results["cam_names"][i]
                    np.save(os.path.join(raw_dir, f"{i:05d}_{cam_name}.npy"), depth)
            logger.info(f"Saved raw full depth to {raw_dir}")
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()

    render_novel_cfg = cfg.render.get("render_novel", None)
    if render_novel_cfg is not None:
        logger.info("Rendering novel views...")
        render_traj = dataset.get_novel_render_traj(
            traj_types=['s_curve', 'front_center_interp', 'lateral_offset'], # render_novel_cfg.traj_types,
            target_frames=render_novel_cfg.get("frames", dataset.frame_num),
        )
        video_output_dir = f"{cfg.log_dir}/videos{post_fix}/novel_{step}"
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        # determine which cameras to render for novel views
        cam_selector = getattr(args, "novel_cam_ids", None)
        if cam_selector is None:
            cam_ids = [None]  # keep default behavior
        else:
            if isinstance(cam_selector, str) and cam_selector.strip().lower() == "all":
                cam_ids = dataset.pixel_source.camera_list
            else:
                try:
                    cam_ids = [int(x) for x in str(cam_selector).split(',') if x != '']
                except Exception:
                    cam_ids = [None]

        for traj_type, traj in render_traj.items():
            for cid in cam_ids:
                # Prepare rendering data for selected camera
                render_data = dataset.prepare_novel_view_render_data(traj, cam_id=cid)
                # Render and save video
                suffix = "" if cid is None else f"_cam{cid}"
                save_path = os.path.join(video_output_dir, f"{traj_type}{suffix}.mp4")
                render_novel_views(
                    trainer, render_data, save_path,
                    fps=render_novel_cfg.get("fps", cfg.render.fps)
                )
                logger.info(
                    f"Saved novel view video for trajectory type: {traj_type}"
                    + ("" if cid is None else f", cam {cid}")
                    + f" to {save_path}"
                )
                # Optionally dump raw depth for novel views
                if getattr(args, "save_raw_depth", None):
                    base_out = args.save_raw_depth
                    out_base = base_out if os.path.isabs(base_out) else os.path.join(cfg.log_dir, base_out)
                    raw_dir = os.path.join(out_base, f"novel_{step}", f"{traj_type}{suffix}")
                    os.makedirs(raw_dir, exist_ok=True)
                    # If also asked to include depth visualization, open a writer
                    writer = None
                    if getattr(args, "include_depth", False):
                        depth_mp4 = os.path.join(video_output_dir, f"{traj_type}{suffix}_depths.mp4")
                        writer = imageio.get_writer(depth_mp4, mode='I', fps=render_novel_cfg.get("fps", cfg.render.fps))
                    for i, frame_data in enumerate(render_data):
                        # move to device
                        for key, value in frame_data["cam_infos"].items():
                            frame_data["cam_infos"][key] = value.cuda(non_blocking=True)
                        for key, value in frame_data["image_infos"].items():
                            frame_data["image_infos"][key] = value.cuda(non_blocking=True)
                        outputs = trainer(
                            image_infos=frame_data["image_infos"],
                            camera_infos=frame_data["cam_infos"],
                            novel_view=True
                        )
                        depth = outputs["depth"].detach().cpu().numpy()
                        np.save(os.path.join(raw_dir, f"{i:05d}.npy"), depth)
                        if writer is not None:
                            opacity = outputs.get("opacity").detach().cpu().numpy() if "opacity" in outputs else None
                            colored = depth_visualizer(depth, opacity)
                            writer.append_data(to8b(colored))
                    if writer is not None:
                        writer.close()
                        logger.info(f"Saved novel depth video to {depth_mp4}")
                    logger.info(f"Saved raw novel depth to {raw_dir}")

def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    args.enable_wandb = False
    for folder in ["videos_eval", "metrics_eval"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device
    )

    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )

    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)

    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")

    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False
    # optionally include depth renders in output videos
    if getattr(args, "include_depth", False):
        if "depths" not in render_keys:
            render_keys.append("depths")

    do_evaluation(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        post_fix="_eval"
    )

    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str, required=True)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")
    parser.add_argument("--save_catted_videos", type=bool, default=False, help="visualize lidar on image")
    parser.add_argument("--novel_cam_ids", type=str, default=None,
                        help="Novel view cameras to render: 'all' or comma-separated ids (e.g., '0,1,2'). Default: dataset default (front cam).")
    parser.add_argument("--include_depth", action="store_true",
                        help="Include depth renders in evaluation videos (adds 'depths' to render keys).")
    parser.add_argument("--save_raw_depth", type=str, default=None,
                        help="Directory to save raw depth as .npy (meters). Relative paths are under log_dir.")

    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")

    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    main(args)
