import json
import math
from pathlib import Path

import numpy as np


def _clip_ids(clips):
    return [str(clip) for clip in clips]


def _read_legacy_metrics(path):
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return payload["metrics"]


def _load_rgb(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path, shape):
    from PIL import Image

    if not path.is_file():
        return None
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) > 0
    return mask


def _psnr(pred, gt):
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return -10.0 * math.log10(mse)


def _ssim(pred, gt):
    from skimage.metrics import structural_similarity

    return float(structural_similarity(pred, gt, data_range=1.0, channel_axis=-1))


def _ssim_masked(pred, gt, mask):
    from skimage.metrics import structural_similarity

    _, ssim_map = structural_similarity(pred, gt, data_range=1.0, channel_axis=-1, full=True)
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=-1)
    return float(ssim_map[mask].mean())


def _mean(values):
    finite = [float(value) for value in values if value is not None and not np.isnan(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


class AssetPhotometricMetric:
    def __init__(
        self,
        method_name,
        reconstruction_root,
        clips,
        compute_lpips=True,
        **kwargs,
    ):
        self.method_name = method_name
        self.reconstruction_root = Path(reconstruction_root)
        self.clips = _clip_ids(clips)
        self.compute_lpips = compute_lpips

    def _legacy_path(self, clip):
        return self.reconstruction_root / self.method_name / "train" / clip / "metrics.json"

    def _clip_dir(self, clip):
        return self.reconstruction_root / self.method_name / "train" / clip

    def _clip_assets_ready(self, clip):
        clip_dir = self._clip_dir(clip)
        return any((clip_dir / "pred").glob("*")) and any((clip_dir / "gt").glob("*"))

    def _assets_ready(self):
        for clip in self.clips:
            if not self._clip_assets_ready(clip):
                return False
        return True

    def _lpips_model(self):
        if not self.compute_lpips:
            return None, None, None
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            return LearnedPerceptualImagePatchSimilarity(normalize=True).to(device).eval(), device, "torchmetrics"
        except ImportError:
            import lpips

            return lpips.LPIPS(net="alex").to(device).eval(), device, "lpips"

    def _lpips_score(self, model, device, backend, pred, gt):
        if model is None:
            return None
        import torch

        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
        if backend == "lpips":
            pred_t = pred_t * 2.0 - 1.0
            gt_t = gt_t * 2.0 - 1.0
        with torch.no_grad():
            return float(model(pred_t, gt_t).mean().item())

    def _compute_clip(self, clip, lpips_model, lpips_device, lpips_backend):
        clip_dir = self._clip_dir(clip)
        pred_dir = clip_dir / "pred"
        gt_dir = clip_dir / "gt"
        pred_files = sorted(path for path in pred_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not pred_files:
            raise FileNotFoundError(f"No photometric prediction images found under {pred_dir}")

        psnr_values = []
        ssim_values = []
        lpips_values = []
        masked_psnr_values = []
        masked_ssim_values = []

        for pred_path in pred_files:
            gt_path = gt_dir / pred_path.name
            if not gt_path.is_file():
                raise FileNotFoundError(f"Missing GT image for {pred_path.name}: {gt_path}")
            pred = _load_rgb(pred_path)
            gt = _load_rgb(gt_path)
            psnr_values.append(_psnr(pred, gt))
            ssim_values.append(_ssim(pred, gt))
            lpips_value = self._lpips_score(lpips_model, lpips_device, lpips_backend, pred, gt)
            if lpips_value is not None:
                lpips_values.append(lpips_value)

            mask = _load_mask(clip_dir / "masks" / "all" / pred_path.with_suffix(".png").name, pred.shape)
            if mask is not None and mask.any():
                masked_psnr_values.append(_psnr(pred[mask], gt[mask]))
                masked_ssim_values.append(_ssim_masked(pred, gt, mask))

        metrics = {
            "image_metrics/full/psnr": _mean(psnr_values),
            "image_metrics/full/ssim": _mean(ssim_values),
        }
        if lpips_values:
            metrics["image_metrics/full/lpips"] = _mean(lpips_values)
        if masked_psnr_values:
            metrics["image_metrics/full/masked_psnr"] = _mean(masked_psnr_values)
        if masked_ssim_values:
            metrics["image_metrics/full/masked_ssim"] = _mean(masked_ssim_values)
        return metrics

    def __call__(self):
        lpips_model = lpips_device = lpips_backend = None
        result = {}
        for clip in self.clips:
            legacy = _read_legacy_metrics(self._legacy_path(clip))
            if legacy is not None and not self._clip_assets_ready(clip):
                result[clip] = legacy
                continue
            if lpips_model is None and self.compute_lpips:
                lpips_model, lpips_device, lpips_backend = self._lpips_model()
            result[clip] = self._compute_clip(clip, lpips_model, lpips_device, lpips_backend)
        return result


def _load_depth(path):
    depth = np.load(path)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth.astype(np.float64)


class AssetGeometricMetric:
    def __init__(
        self,
        method_name,
        reconstruction_root,
        clips,
        **kwargs,
    ):
        self.method_name = method_name
        self.reconstruction_root = Path(reconstruction_root)
        self.clips = _clip_ids(clips)

    def _legacy_path(self, clip):
        return self.reconstruction_root / self.method_name / "depth" / clip / "metrics.json"

    def _clip_dir(self, clip):
        return self.reconstruction_root / self.method_name / "depth" / clip

    def _clip_assets_ready(self, clip):
        clip_dir = self._clip_dir(clip)
        return any((clip_dir / "pred").glob("*.npy")) and any((clip_dir / "gt").glob("*.npy"))

    def _assets_ready(self):
        for clip in self.clips:
            if not self._clip_assets_ready(clip):
                return False
        return True

    def _compute_clip(self, clip):
        clip_dir = self._clip_dir(clip)
        pred_files = sorted((clip_dir / "pred").glob("*.npy"))
        if not pred_files:
            raise FileNotFoundError(f"No depth predictions found under {clip_dir / 'pred'}")

        sq_sum = 0.0
        log_sq_sum = 0.0
        abs_rel_sum = 0.0
        sq_rel_sum = 0.0
        delta1 = delta2 = delta3 = 0
        pixels = 0
        rel_pixels = 0
        eps = 1e-6

        for pred_path in pred_files:
            gt_path = clip_dir / "gt" / pred_path.name
            if not gt_path.is_file():
                raise FileNotFoundError(f"Missing GT depth for {pred_path.name}: {gt_path}")
            pred = _load_depth(pred_path)
            gt = _load_depth(gt_path)
            mask = _load_mask(clip_dir / "masks" / pred_path.with_suffix(".png").name, pred.shape)
            valid = np.ones_like(gt, dtype=bool)
            if mask is not None:
                valid &= mask
            if not valid.any():
                continue
            pred_values = pred[valid]
            gt_values = gt[valid]
            diff = pred_values - gt_values
            sq_sum += float((diff ** 2).sum())
            pixels += int(valid.sum())
            rel_mask = gt_values > eps
            if rel_mask.any():
                pred_rel = pred_values[rel_mask]
                gt_rel = gt_values[rel_mask]
                rel_diff = pred_rel - gt_rel
                log_sq_sum += float(((np.log(np.maximum(pred_rel, eps)) - np.log(np.maximum(gt_rel, eps))) ** 2).sum())
                abs_rel_sum += float((np.abs(rel_diff) / gt_rel).sum())
                sq_rel_sum += float(((rel_diff ** 2) / gt_rel).sum())
                ratio = np.maximum(np.maximum(pred_rel, eps) / gt_rel, gt_rel / np.maximum(pred_rel, eps))
                delta1 += int((ratio < 1.25).sum())
                delta2 += int((ratio < 1.25 ** 2).sum())
                delta3 += int((ratio < 1.25 ** 3).sum())
                rel_pixels += int(rel_mask.sum())

        if pixels == 0:
            raise ValueError(f"No valid depth pixels for method={self.method_name}, clip={clip}")
        if rel_pixels == 0:
            raise ValueError(f"No valid relative depth pixels for method={self.method_name}, clip={clip}")
        return {
            "global_rmse": math.sqrt(sq_sum / pixels),
            "global_log_rmse": math.sqrt(log_sq_sum / rel_pixels),
            "global_abs_rel": abs_rel_sum / rel_pixels,
            "global_sq_rel": sq_rel_sum / rel_pixels,
            "global_delta1": delta1 / rel_pixels,
            "global_delta2": delta2 / rel_pixels,
            "global_delta3": delta3 / rel_pixels,
        }

    def __call__(self):
        result = {}
        for clip in self.clips:
            legacy = _read_legacy_metrics(self._legacy_path(clip))
            if legacy is not None and not self._clip_assets_ready(clip):
                result[clip] = legacy
                continue
            result[clip] = self._compute_clip(clip)
        return result
