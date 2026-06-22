import os
import json

# novel view synthesis results
root = "logs/omnire_waymo/nvs"
split = "test"
# scene reconstruction results
# root = "logs/omnire_waymo/recon"
# split = "full"

scene_ids = os.listdir(root)
scene_ids.sort()
metrics = {}
for scene_id in scene_ids:
    try:
        metric = os.listdir(os.path.join(root, str(scene_id), "metrics"))
        metric = [m for m in metric if m.startswith(f"images_{split}")][0]
        with open(os.path.join(root, str(scene_id), "metrics", metric), "r") as f:
            data = json.load(f)
    except:
        print("error", scene_id)
        continue
    try:
        metrics[scene_id] = {
            "psnr": data[f"image_metrics/{split}/psnr"],
            "ssim": data[f"image_metrics/{split}/ssim"],
            "human_psnr": data[f"image_metrics/{split}/human_psnr"],
            "human_ssim": data[f"image_metrics/{split}/human_ssim"],
            "vehicle_psnr": data[f"image_metrics/{split}/vehicle_psnr"],
            "vehicle_ssim": data[f"image_metrics/{split}/vehicle_ssim"],
        }
    except:
        print(data)
        breakpoint()
for scene_id in scene_ids:
    if scene_id in metrics:
        print(
            scene_id,
            metrics[scene_id]["psnr"],
            metrics[scene_id]["ssim"],
            metrics[scene_id]["human_psnr"],
            metrics[scene_id]["human_ssim"],
            metrics[scene_id]["vehicle_psnr"],
            metrics[scene_id]["vehicle_ssim"],
        )
    else:
        print("error", scene_id)
        continue
# print avg
psnr = [metrics[scene_id]["psnr"] for scene_id in metrics.keys()]
ssim = [metrics[scene_id]["ssim"] for scene_id in metrics.keys()]
human_psnr = [metrics[scene_id]["human_psnr"] for scene_id in metrics.keys()]
human_ssim = [metrics[scene_id]["human_ssim"] for scene_id in metrics.keys()]
vehicle_psnr = [metrics[scene_id]["vehicle_psnr"] for scene_id in metrics.keys()]
vehicle_ssim = [metrics[scene_id]["vehicle_ssim"] for scene_id in metrics.keys()]
print("avg    Full Image  PSNR: ", sum(psnr) / len(psnr))
print("avg    Full Image  SSIM: ", sum(ssim) / len(ssim))
print("avg     Human-Only PSNR: ", sum(human_psnr) / len(human_psnr))
print("avg     Human-Only SSIM: ", sum(human_ssim) / len(human_ssim))
print("avg   Vehicle-Only PSNR: ", sum(vehicle_psnr) / len(vehicle_psnr))
print("avg   Vehicle-Only SSIM: ", sum(vehicle_ssim) / len(vehicle_ssim))
print("len", len(psnr))