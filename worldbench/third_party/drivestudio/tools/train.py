from omegaconf import OmegaConf
import numpy as np
import os
import time
import wandb
import random
import imageio
import logging
import argparse
import socket
import atexit

import torch
from tools.eval import do_evaluation
from utils.misc import import_str
from utils.backup import backup_project
from utils.logging import MetricLogger, setup_logging
from models.video_utils import render_images, save_videos
from datasets.driving_dataset import DrivingDataset

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def set_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup(args):
    # get config
    cfg = OmegaConf.load(args.config_file)

    # parse datasets
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")

    assert "dataset" in cfg or "data" in cfg, \
        "Please specify dataset in config or data in config"

    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        # merge data
        cfg = OmegaConf.merge(cfg, dataset_cfg)

    # merge cli
    cfg = OmegaConf.merge(cfg, args_from_cli)
    log_dir = os.path.join(args.output_root, args.project, args.run_name)

    # update config and create log dir
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for folder in ["images", "videos", "metrics", "configs_bk", "buffer_maps", "backup"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)

    # setup wandb
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
            wandb.init(
                project=args.project,
                entity=args.entity,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="fork"),
            )
            is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    # setup random seeds
    set_seeds(cfg.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # also save a backup copy
    saved_cfg_path_bk = os.path.join(log_dir, "configs_bk", f"config_{current_time}.yaml")
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")

    # Backup codes
    backup_project(
        os.path.join(log_dir, 'backup'), "./",
        ["configs", "datasets", "models", "utils", "tools"],
        [".py", ".h", ".cpp", ".cuh", ".cu", ".sh", ".yaml"]
    )
    return cfg

def train(cfg, device):

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

    # NOTE: If resume, gaussians will be loaded from checkpoint
    #       If not, gaussians will be initialized from dataset
    if args.resume_from is not None:
        trainer.resume_from_checkpoint(
            ckpt_path=args.resume_from,
            load_only_model=True
        )
        logger.info(
            f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
        )
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(
            f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}"
        )

    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)

    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "Dynamic_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "Dynamic_depths",
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

    # setup optimizer
    trainer.initialize_optimizer()

    # setup metric logger
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    all_iters = np.arange(trainer.step, trainer.num_iters + 1)

    # DEBUG USE
    # do_evaluation(
    #     step=0,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    # )

    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        #----------------------------------------------------------------------------
        #----------------------------     Validate     ------------------------------
        if step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0:
            logger.info("Visualizing...")
            vis_timestep = np.linspace(
                0,
                dataset.num_img_timesteps,
                trainer.num_iters // cfg.logging.vis_freq + 1,
                endpoint=False,
                dtype=int,
            )[step // cfg.logging.vis_freq]
            with torch.no_grad():
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                    compute_metrics=True,
                    compute_error_map=cfg.render.vis_error,
                    vis_indices=[
                        vis_timestep * dataset.pixel_source.num_cams + i
                        for i in range(dataset.pixel_source.num_cams)
                    ],
                )
            if args.enable_wandb:
                wandb.log(
                    {
                        "image_metrics/psnr": render_results["psnr"],
                        "image_metrics/ssim": render_results["ssim"],
                        "image_metrics/occupied_psnr": render_results["occupied_psnr"],
                        "image_metrics/occupied_ssim": render_results["occupied_ssim"],
                    }
                )
            vis_frame_dict = save_videos(
                render_results,
                save_pth=os.path.join(
                    cfg.log_dir, "images", f"step_{step}.png"
                ),  # don't save the video
                layout=dataset.layout,
                num_timestamps=1,
                keys=render_keys,
                save_seperate_video=cfg.logging.save_seperate_video,
                num_cams=dataset.pixel_source.num_cams,
                fps=cfg.render.fps,
                verbose=False,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"image_rendering/" + k: wandb.Image(v)})
            del render_results
            torch.cuda.empty_cache()


        #----------------------------------------------------------------------------
        #----------------------------  training step  -------------------------------
        # prepare for training
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad() # zero grad

        # get data
        train_step_camera_downscale = trainer._get_downscale_factor()
        image_infos, cam_infos = dataset.train_image_set.next(train_step_camera_downscale)
        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.cuda(non_blocking=True)

        # forward & backward
        outputs = trainer(image_infos, cam_infos)
        trainer.update_visibility_filter()

        loss_dict = trainer.compute_losses(
            outputs=outputs,
            image_infos=image_infos,
            cam_infos=cam_infos,
        )
        # check nan or inf
        # for k, v in loss_dict.items():
        #     if torch.isnan(v).any():
        #         raise ValueError(f"NaN detected in loss {k} at step {step}")
        #     if torch.isinf(v).any():
        #         raise ValueError(f"Inf detected in loss {k} at step {step}")

        # TODO: 用生成数据训练的时候Background_sharp_shape_reg会出现nan, 需要check
        loss_dict_keys = list(loss_dict.keys())
        for k in loss_dict_keys:
            if torch.isnan(loss_dict[k]).any() or torch.isinf(loss_dict[k]).any():
                loss_dict.pop(k)
        trainer.backward(loss_dict)

        # after training step
        trainer.postprocess_per_train_step(step=step)

        #----------------------------------------------------------------------------
        #-------------------------------  logging  ----------------------------------
        with torch.no_grad():
            # cal stats
            metric_dict = trainer.compute_metrics(
                outputs=outputs,
                image_infos=image_infos,
            )
        metric_logger.update(**{"train_metrics/"+k: v.item() for k, v in metric_dict.items()})
        metric_logger.update(**{"train_stats/gaussian_num_" + k: v for k, v in trainer.get_gaussian_count().items()})
        metric_logger.update(**{"losses/"+k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{"train_stats/lr_" + group['name']: group['lr'] for group in trainer.optimizer.param_groups})
        if args.enable_wandb:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        #----------------------------------------------------------------------------
        #----------------------------     Saving     --------------------------------
        do_save = step > 0 and (
            (step % cfg.logging.saveckpt_freq == 0) or (step == trainer.num_iters)
        ) and (args.resume_from is None)
        if do_save:
            trainer.save_checkpoint(
                log_dir=cfg.log_dir,
                save_only_model=True,
                is_final=step == trainer.num_iters,
            )

        #----------------------------------------------------------------------------
        #------------------------    Cache Image Error    ---------------------------
        if (
            step > 0 and trainer.optim_general.cache_buffer_freq > 0
            and step % trainer.optim_general.cache_buffer_freq == 0
        ):
            logger.info("Caching image error...")
            trainer.set_eval()
            with torch.no_grad():
                dataset.pixel_source.update_downscale_factor(
                    1 / dataset.pixel_source.buffer_downscale
                )
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                )
                dataset.pixel_source.reset_downscale_factor()
                dataset.pixel_source.update_image_error_maps(render_results)

                # save error maps
                merged_error_video = dataset.pixel_source.get_image_error_video(
                    dataset.layout
                )
                imageio.mimsave(
                    os.path.join(
                        cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
                    ),
                    merged_error_video,
                    fps=cfg.render.fps,
                )
            logger.info("Done caching rgb error maps")


    logger.info("Training done!")

    do_evaluation(
        step=step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
    )

    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

    return step

def is_data_ready(scene_dir):
    return 'fine_dynamic_masks' in os.listdir(scene_dir) and \
           'humanpose' in os.listdir(scene_dir) and \
           len(os.listdir(os.path.join(scene_dir, 'fine_dynamic_masks', 'all'))) == len(os.listdir(os.path.join(scene_dir, 'images'))) and \
           len(os.listdir(os.path.join(scene_dir, 'fine_dynamic_masks', 'human'))) == len(os.listdir(os.path.join(scene_dir, 'images'))) and \
           len(os.listdir(os.path.join(scene_dir, 'fine_dynamic_masks', 'vehicle'))) == len(os.listdir(os.path.join(scene_dir, 'images'))) and \
           os.path.exists(os.path.join(scene_dir, 'humanpose', 'smpl.pkl'))

# ---------------------------------------------------------------
# Simple file-based lock for multi-process scene scheduling.
# - Lock files live under: <output_root>/<project>/_locks/<scene>.lock
# - Create uses O_CREAT|O_EXCL for atomic acquisition across processes.
# - Always release in finally; atexit guard removes any leftover locks.
# ---------------------------------------------------------------
_HELD_LOCKS = set()

def _release_all_locks_at_exit():
    for p in list(_HELD_LOCKS):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

atexit.register(_release_all_locks_at_exit)

def acquire_scene_lock(lock_root, scene_idx):
    """Try to acquire an exclusive lock for a scene.

    Returns lock_path (str) on success, or None if already locked or on failure.
    """
    try:
        os.makedirs(lock_root, exist_ok=True)
        lock_path = os.path.join(lock_root, f"{scene_idx:03d}.lock")
        # atomic create; fails if exists
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        # record some metadata for debugging
        with os.fdopen(fd, "w") as f:
            f.write(
                f"pid={os.getpid()} host={socket.gethostname()} time={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
        _HELD_LOCKS.add(lock_path)
        return lock_path
    except FileExistsError:
        return None
    except Exception as e:
        logger.warning(f"Failed to acquire lock for scene {scene_idx:03d}: {e}")
        return None

def release_scene_lock(lock_path):
    """Release a previously acquired lock (idempotent)."""
    if not lock_path:
        return
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception as e:
        logger.warning(f"Failed to release lock {lock_path}: {e}")
    finally:
        _HELD_LOCKS.discard(lock_path)

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts = dict([(x.split("=", 1)[0], x.split("=", 1)[1]) for x in args.opts])
    # all scenes
    if opts['data.scene_idx'] == '-1':

        # 轮询间隔（秒）
        POLL_INTERVAL = 5

        # 获取所有场景索引（假设目录名是数字，如 000、001、...）
        assert 'data.data_root' in opts
        data_root = opts['data.data_root']
        all_scene_idxs = sorted(
            int(d) for d in os.listdir(data_root)
            if d.isdigit()
        )
        if args.end_idx != -1:
            all_scene_idxs = all_scene_idxs[args.start_idx:args.end_idx+1]

        # 待处理集合与失败记录
        pending = set(all_scene_idxs)
        all_fails = {}

        # 锁目录：与输出同盘，避免写到只读数据盘
        lock_root = os.path.join(args.output_root, args.project, "_locks")

        while pending:
            locked_scene = None
            lock_path = None

            # 在所有待处理场景中，优先挑选已就绪并且成功抢到锁的那一个
            for scene_idx in sorted(pending):
                scene_dir = os.path.join(data_root, f"{scene_idx:03d}")
                if not is_data_ready(scene_dir):
                    continue

                # 已处理完成的直接剔除（避免无谓加锁）
                run_dir = os.path.join(args.output_root, args.project, f"{scene_idx:03d}")
                if os.path.exists(os.path.join(run_dir, 'checkpoint_final.pth')):
                    print(f"Scene {scene_idx:03d} already processed, skipping.")
                    pending.remove(scene_idx)
                    continue

                # 抢占式加锁：原子创建锁文件，失败则尝试下一个场景
                lock_path_try = acquire_scene_lock(lock_root, scene_idx)
                if lock_path_try is None:
                    continue

                # 加锁后再二次确认是否已完成（与其他进程竞争到末尾的竞态）
                if os.path.exists(os.path.join(run_dir, 'checkpoint_final.pth')):
                    # 已完成则释放锁并跳过
                    release_scene_lock(lock_path_try)
                    pending.remove(scene_idx)
                    continue

                locked_scene = scene_idx
                lock_path = lock_path_try
                break

            if locked_scene is None:
                # 没有可加锁的且 ready 的场景，等待后重试
                print(f"No ready & unlocked scenes. Waiting for {POLL_INTERVAL} seconds...")
                time.sleep(POLL_INTERVAL)
                continue

            # 处理这个获得锁的场景
            args.run_name = f"{locked_scene:03d}"
            # 更新 CLI opts 中的 scene_idx（去重后添加）
            for item in list(args.opts):
                if item.startswith("data.scene_idx="):
                    args.opts.remove(item)
            args.opts.append(f"data.scene_idx={locked_scene}")

            # try:
            cfg = setup(args)
            train(cfg, device)
            # except Exception as e:
            #     print(f"Fail to process scene {locked_scene:03d}, due to {e}")
            #     all_fails[locked_scene] = e
            # finally:
            #     # 失败或成功都释放锁，避免死锁
            #     release_scene_lock(lock_path)

            # 从待处理集合里移除
            pending.remove(locked_scene)

        print(all_fails)
    else:
        cfg = setup(args)
        return train(cfg, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument("--output_root", default="./work_dirs/", help="path to save checkpoints and logs", type=str)

    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")

    # wandb logging part
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--entity", default="ziyc", type=str, help="wandb entity name")
    parser.add_argument("--project", default="drivestudio", type=str, help="wandb project name, also used to enhance log_dir")
    parser.add_argument("--run_name", default="omnire", type=str, help="wandb run name, also used to enhance log_dir")

    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")

    # start & end
    parser.add_argument("--start_idx", type=int, default=0, help="start idx")
    parser.add_argument("--end_idx", type=int, default=-1, help="end idx")

    # novel view rendering
    parser.add_argument("--novel_cam_ids", type=str, default=None,
                        help="Novel view cameras for evaluation: 'all' or comma-separated ids (e.g., '0,1,2'). Default: dataset default (front cam).")

    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    final_step = main(args)
