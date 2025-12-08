We use [BEVFomer-base-150x150](projects/configs/bevformer/bevformer_base_seg_det_150x150.py) for evaluation on both 3D object detection and BEV segmentation task. Download the [pre-trained model](https://drive.google.com/file/d/1UM9Ifduu-IMF7qAW9nUlwXiXPZ17RkqQ/view?usp=share_link).

We have adapt the original code to the python environment for `MagicDrive-t`. To run the evaluation, please install the [`mmsegmentation-0.20.0`](https://github.com/open-mmlab/mmsegmentation/releases/tag/v0.20.0) with the following commands:

```bash
pip install https://github.com/open-mmlab/mmsegmentation/archive/refs/tags/v0.20.0.zip
```

**Important!** Please verify the version for following packages:

 ```bash
torch==1.10.2
torchmetrics==0.9.0  # not work with > 0.10
mmcv-full==1.4.5
mmdet==2.20.0
mmdet3d from third_party/bevfusion
 ```

### Run BEVFormer

Please verify your local environment for BEVFormer with the command:

```bash
cd ${MagicDrive_Root}
bash perception/BEVFormer/tools/dist_test.sh \
	perception/BEVFormer/projects/configs/bevformer/bevformer_base_seg_det_150x150.py \
    ${PRETRAIN}/bevformer_base_seg_det_150.pth 8  # you can use any number of gpus
```

The results will be located in `magicdrive-t-log/evaluation/bevformer_base_seg_det_150x150/*`

- `segmentation_result.json` for BEV segmentation. We use `mIoU`
- `pts_bbox/metrics_summary.json` for 3D object detection, We use `mean_ap`
