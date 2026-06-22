# Reconstruction Metrics

Reconstruction metrics use the same WorldLens evaluator flow as the other
video-generation metrics: enable metrics in the YAML config, then run
`tools/evaluate.py`.

WorldLens owns metric calculation and aggregation. DriveStudio is only used to
prepare reconstruction assets before evaluation. The DriveStudio source needed
for this path is vendored in `worldbench/third_party/drivestudio`.

## Environment

Start from the normal WorldLens environment described in the project README.
Reconstruction metric computation runs in that environment.

To train/render 4DGS assets with the vendored DriveStudio backend, install only
the Reconstruction extras on top of the WorldLens environment:

```bash
cd /path/to/WorldLens
pip install -r worldbench/videogen/reconstruction/requirements-reconstruction-extra.txt
```

Do not install
`worldbench/third_party/drivestudio/requirements.txt` into
the WorldLens environment. That file pins a separate DriveStudio stack,
including older torch/torchvision/CUDA versions, and can conflict with the
WorldLens CUDA extensions.

The base WorldLens environment already includes several dependencies used by
this path, such as `gsplat`, `lpips`, `nuscenes-devkit`, `open3d`,
`pyquaternion`, and `torchmetrics`. The extra file adds the missing runtime
pieces without changing the WorldLens torch/CUDA version. If `pytorch3d` is not
already available in your environment, install a wheel that matches the
WorldLens PyTorch/CUDA build before running 4DGS training.

## Asset Contract

Reconstruction metrics expect this layout:

```text
generated_results/reconstruction/
  magicdrive/
    manifest.json
    ckpt/081/config.yaml
    ckpt/081/checkpoint_final.pth
    train/081/pred/0000_CAM_FRONT.jpg
    train/081/gt/0000_CAM_FRONT.jpg
    depth/081/pred/00000_CAM_FRONT.npy
    depth/081/gt/00000_CAM_FRONT.npy
    depth/081/masks/00000_CAM_FRONT.png
    novel/081/front_center_interp.mp4
    novel/081/s_curve.mp4
    novel/081/lateral_offset.mp4
    novel/081/lateral_offset_left.mp4
```

Legacy `train/<clip>/metrics.json` and `depth/<clip>/metrics.json` files are
still accepted as a fallback when train/depth assets are not available, but the
recommended path is asset-based metric computation inside WorldLens.

## Evaluate Assets

Use this after 4DGS assets have been prepared.

```yaml
dimensions:
  reconstruction:
    photometric_error:
      - name: photometric_error
        method_name: magicdrive
        reconstruction_root: generated_results/reconstruction
        clips: ["081"]

    geometric_discrepancy:
      - name: geometric_discrepancy
        method_name: magicdrive
        reconstruction_root: generated_results/reconstruction
        clips: ["081"]

    novel_view_fidelity:
      - name: novel_view_fidelity
        method_name: magicdrive
        reconstruction_root: generated_results/reconstruction
        clips: ["081"]

    novel_view_consistency:
      - name: novel_view_consistency
        method_name: magicdrive
        reconstruction_root: generated_results/reconstruction
        clips: ["081"]
        local_save_path: pretrained_models/clip/ViT-B-32.pt
```

Run through the normal WorldLens entrypoint:

```bash
cd /path/to/WorldLens
python tools/evaluate.py method_name=magicdrive
```

## Prepare Assets With DriveStudio

Use this when new generated videos need to be reconstructed with DriveStudio 4DGS
before WorldLens evaluates them.

```bash
cd /path/to/WorldLens
python worldbench/videogen/reconstruction/prepare_assets.py \
  --method-name magicdrive \
  --clips 081 \
  --reconstruction-root generated_results/reconstruction \
  --work-root work_dirs \
  --data-root data/nuscenes_trainval/raw \
  --target-dir data/nuscenes_trainval/processed
```

The exporter preprocesses generated videos, trains 4DGS, renders train/depth and
novel-view outputs, exports standard assets, and writes a manifest. WorldLens
then computes photometric, geometric, and novel-view metrics from those assets.

The default 4DGS training config is
`worldbench/third_party/drivestudio/configs/omnire.yaml`.
Override it with `--config-file`; relative config paths are resolved from the
vendored DriveStudio root:

```bash
python worldbench/videogen/reconstruction/prepare_assets.py \
  --method-name magicdrive \
  --clips 081 \
  --reconstruction-root generated_results/reconstruction \
  --work-root work_dirs \
  --data-root data/nuscenes_trainval/raw \
  --target-dir data/nuscenes_trainval/processed \
  --config-file configs/streetgs.yaml
```

Available vendored configs include `configs/omnire.yaml`,
`configs/streetgs.yaml`, `configs/pvg.yaml`, and `configs/deformablegs.yaml`.
The backend uses DriveStudio's `nuscenes_gen/6cams` dataset config for generated
video reconstruction and derives the training clip range from `--clips`.

For reference, the wrapper launches DriveStudio training with the same effective
arguments as:

```bash
cd worldbench/third_party/drivestudio
python tools/train.py \
  --config_file configs/omnire.yaml \
  --output_root /path/to/WorldLens/work_dirs \
  --project drivestudio-nus-magicdrive \
  --run_name -1 \
  --start_idx 81 \
  --end_idx 81 \
  dataset=nuscenes_gen/6cams \
  data.scene_idx=-1 \
  data.start_timestep=0 \
  data.end_timestep=-1 \
  data.data_root=/path/to/WorldLens/data/nuscenes_trainval/processed_magicdrive/advanced_12Hz_trainval
```

DriveStudio renders novel-view videos; WorldLens computes novel-view fidelity
and consistency from those videos.

The DriveStudio source code is vendored under `worldbench/third_party`, so users
do not need a separate DriveStudio checkout. By default the backend runs with
the current WorldLens Python. Use `--python-bin` only if you deliberately want
to run the backend in a separate Python environment.

## New Data Requirements

Generated videos should use the DriveStudio `nuscenes_gen` layout for the
selected `method_name`.

Before 4DGS training, each requested clip must have:

- processed images
- `fine_dynamic_masks/all`
- `fine_dynamic_masks/human`
- `fine_dynamic_masks/vehicle`
- `humanpose/smpl.pkl`

The backend validates these inputs before launching training. Missing assets
fail early with a direct error message.

`geometric_discrepancy` is not a pure RGB-video metric. It requires reference/GT
depth assets and Grounded-SAM2 evaluation masks in addition to generated videos.
The DriveStudio exporter copies GT depth from `--gt-method`, which defaults to
`gt`, and copies masks from `processed_<gt-method>`.

Generate the evaluation masks once from GT processed images and reuse them for
all methods. Run this in an environment where Grounded-SAM-2, GroundingDINO,
SAM2, and their checkpoints are available:

```bash
cd /path/to/WorldLens
python worldbench/videogen/reconstruction/generate_eval_masks.py \
  --grounded-sam-root /path/to/Grounded-SAM-2 \
  --processed-root data/nuscenes_trainval/processed_gt/advanced_12Hz_trainval \
  --clips 081 \
  --skip-existing
```

This writes
`data/nuscenes_trainval/processed_gt/advanced_12Hz_trainval/<clip>/sam_mask/<frame>_<camera_id>.png`.
The masks use the paper prompts `road surface` and `vehicles`. They are only for
geometric discrepancy; DriveStudio's `fine_dynamic_masks/{all,human,vehicle}`
remain training inputs and are not WorldLens metric masks.

If checkpoints already exist and only rendering/export is needed, skip the
preprocess and training phases:

```bash
cd /path/to/WorldLens
python worldbench/videogen/reconstruction/prepare_assets.py \
  --method-name magicdrive \
  --clips 081 \
  --reconstruction-root generated_results/reconstruction \
  --work-root work_dirs \
  --data-root data/nuscenes_trainval/raw \
  --target-dir data/nuscenes_trainval/processed \
  --skip-preprocess \
  --skip-train
```

If all renders and raw depths already exist and only the WorldLens asset layout
needs to be refreshed, add `--skip-eval` as well.

Use `--drivestudio-root /path/to/drivestudio` only to override the vendored
source during backend development.
