# Getting Started

## Reconstruction

Reconstruction metrics follow the same WorldLens YAML and evaluator flow as other
video-generation metrics. They read 4DGS assets that already exist under
`generated_results/reconstruction` and compute metrics in the normal WorldLens
environment.

See `worldbench/videogen/reconstruction/README.md` for the full Reconstruction
usage guide.

For new videos, first run the vendored DriveStudio Reconstruction backend to
prepare assets. The backend runs on top of the WorldLens environment with the
Reconstruction extras installed. It trains 4DGS, renders train/depth/novel
outputs, and writes the asset layout consumed by WorldLens.

Install the Reconstruction extras after setting up the base WorldLens
environment:

```bash
pip install -r worldbench/videogen/reconstruction/requirements-reconstruction-extra.txt
```

Example asset preparation:

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

Example WorldLens metric:

```yaml
reconstruction:
  photometric_error:
    - name: photometric_error
      method_name: ${method_name}
      reconstruction_root: generated_results/reconstruction
      clips: ["081"]
```

DriveStudio training expects generated-data preprocessing dependencies to be
available under the configured data roots, including fine dynamic masks and
`humanpose/smpl.pkl` for each clip. The backend validates those inputs before
launching training so missing preprocessing is reported before a long 4DGS run.
Generated videos should use the DriveStudio `nuscenes_gen` layout for the
selected `method_name`.

`geometric_discrepancy` also needs reference/GT depth assets and GT
Grounded-SAM2 evaluation masks. RGB videos alone are not sufficient for that
metric.
