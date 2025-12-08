import easydict
from loguru import logger
from ..bev_seg_od.engine import run, parse_args

class BEV_SEG_OD_FULL:
    def __init__(self, method_name,
                 need_preprocessing=False,
                 generated_data_path="generated_results",
                 data_info_path="data/nuscenes_mmdet3d_2/nuscenes_infos_temporal_val_3keyframes.pkl",
                 generation_times=4,
                 max_workers=8,
                 repeat_times=1,
                 **kwargs):
        
        self.method_name = method_name
        self.need_preprocessing = need_preprocessing
        self.generated_data_path = generated_data_path
        self.data_info_path = data_info_path
        self.generation_times = generation_times
        self.max_workers = max_workers
        self.repeat_times = repeat_times

    def __call__(self):
        if self.need_preprocessing:
            from .decode_video_full import decode_video
            args = easydict.EasyDict({
                "vid_root": self.generated_data_path + f"/{self.method_name}/video_submission",
                "model_name": self.method_name,
                "data_info": self.data_info_path,
                "generation_times": self.generation_times,
                "max_workers": self.max_workers,
            })
            decode_video(args)
            logger.info(f"Decoding video for {self.method_name} done.")
        return
        for i in range(self.repeat_times):
            info_path = f'generated_results/{self.method_name}/nuscenes_infos_temporal_val_fullframes_gen{i}.pkl'
            args = parse_args(
                ["--method_name", self.method_name,
                "--eval", "box",
                "--cfg-options", f"data.test.ann_file={info_path}", 
                # "data.workers_per_gpu=1"
                ]
            )
            run(args)
        pass