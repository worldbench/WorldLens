import os
import torch
import torchvision.transforms as T
import json
from loguru import logger
from collections import defaultdict
from tqdm import tqdm
from .similarity_calculator import build_model,read_image,compute_identity_consistency,plot_similarity_histogram
from .object_classification import Checkpoint,Predictor,PedestrianClassifier,plot_confidence_histogram
from .car_classification_model import CarClassificationFundationModel

class OBJECT_COHERENCE:

    TARGET_DICTS = {
        "Pedestrian": ["pedestrian"],
        "Vehicle": ["car", "construction_vehicle", "bus", "traiycle", "truck"]
    }

    def __init__(self, method_name,
                 need_preprocessing=False,
                 generation_times=4,
                 repeat_times=1,
                 confidence_threshold = {'Vehicle':0.25, 'Pedestrian':0.5},
                 **kwargs):
        
        self.method_name = method_name
        self.need_preprocessing = need_preprocessing
        self.generation_times = generation_times
        self.repeat_times = repeat_times
        self.confidence_threshold = confidence_threshold
        self.select_img_path_save_path = 'generated_results/gt/subject_consistency/gt_selected_images.json'
        if self.method_name != 'gt':
            assert os.path.exists(self.select_img_path_save_path)
            with open(self.select_img_path_save_path, 'r') as f:
                self.select_img_path = json.load(f)
        else:
            self.select_img_path = defaultdict(list)

        self.val_transforms = T.Compose([
            T.Resize([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_single_target(self, data_root, save_root, target, target_categories):

        reid_ckpt_path = f"pretrained_models/reid/{target.lower()}_reid.pth"
        classify_ckpt_path = f"pretrained_models/classifier/{target.lower()}_classification.pth"
        reid_model = build_model(reid_ckpt_path, neck_feat='before').to(self.device).eval()
        if target == 'Pedestrian':
            classify_model = Checkpoint.load(classify_ckpt_path, model_class=PedestrianClassifier)
            classify_predictor = Predictor(model=classify_model, transform=self.val_transforms, device=self.device)
        else:
            classify_predictor = CarClassificationFundationModel()

        tracklet_dirs = os.listdir(data_root)
        target_dirs = [x for x in tracklet_dirs if "_".join(x.split("_")[1:]) in target_categories]
        target_dirs = sorted(target_dirs)
        avg_sim_list = []
        avg_sim_content = ""
        avg_confidence_list = []
        avg_conf_content = ""
        statistics = {}
        for tracklet in tracklet_dirs:
            category = tracklet.split("_")[1:]
            category = "_".join(category)
            if category not in statistics.keys():
                statistics[category] = 1
            else:
                statistics[category] += 1
        
        print(f"{statistics}")
        for dir in tqdm(target_dirs, f'Processing {target} Directories'):
            img_paths = os.listdir(os.path.join(data_root, dir))
            filtered_img_paths = []
            if len(img_paths) <= 1:
                continue

            confidence_list = []
            for img_path in img_paths:
                actual_img_path = os.path.join(data_root, dir, img_path)
                result = classify_predictor.predict(image_path=actual_img_path, as_dict=False)
                confidence = result.confidence

                if self.method_name == 'gt':
                    if confidence >= self.confidence_threshold[target]:
                        confidence_list.append(round(confidence, 4))
                        self.select_img_path[target].append(f'{dir}_{img_path}')
                        filtered_img_paths.append(img_path)

                else:
                    if f'{dir}_{img_path}' in self.select_img_path[target]:
                        confidence_list.append(round(confidence, 4))                
                        filtered_img_paths.append(img_path)

            if len(confidence_list) == 0:
                continue

            avg_conf = sum(confidence_list) / len(confidence_list)
            print(f"{dir} AVG CONF: {avg_conf}")
            avg_confidence_list.append(round(avg_conf, 4))
            avg_conf_content += f"{dir}: {avg_conf}\n"


            if len(filtered_img_paths)<=1:   # 当前DIR里的样本被滤除至小于等于一张
                print(f"{dir} AVG SIM: ALL Filtered")
                avg_sim_content += f"{dir}: ALL Filtered\n"
                continue

            feats_list = []
            for img_path in filtered_img_paths:
                actual_img_path = os.path.join(data_root, dir, img_path)
                img = read_image(actual_img_path, self.val_transforms).to('cuda').unsqueeze(0)
                feat = reid_model(img).cpu().detach()
                feats_list.append(feat)
            feats = torch.cat(feats_list, dim=0)
            avg_sim, consistency, sim_matrix = compute_identity_consistency(feats)
            print(f"{dir} AVG SIM: {avg_sim}, {consistency}")
            avg_sim_content += f"{dir}: {avg_sim}\n"
            avg_sim_list.append(round(avg_sim, 4))

        avg_sim_list = sorted(avg_sim_list)
        avg_confidence_list = sorted(avg_confidence_list)
        print(f"Total Number of {target}s for Similarity Calculation:{len(avg_sim_list)}")
        print(f"Average Similarities:{sum(avg_sim_list) / len(avg_sim_list)}")
        print(f"Total Number of {target}s for Confidence Calculation:{len(avg_confidence_list)}")
        print(f"Average Confidence:{sum(avg_confidence_list) / len(avg_confidence_list)}")

        avg_sim_content += f"Total Number of {target}s for Similarity Calculation: {len(avg_sim_list)}\n"
        avg_sim_content += f"Average Similarities: {sum(avg_sim_list) / len(avg_sim_list)}"
        avg_conf_content += f"Total Number of {target}s for Confidence Calculation: {len(avg_confidence_list)}\n"
        avg_conf_content += f"Average Confidence: {sum(avg_confidence_list) / len(avg_confidence_list)}"

        sim_txt_save_path = os.path.join(save_root, f"{target}_similarity.txt")
        sim_plot_save_path = os.path.join(save_root, f"{target}_similarity.png")
        conf_txt_save_path = os.path.join(save_root, f"{target}_confidence.txt")
        conf_plot_save_path = os.path.join(save_root, f"{target}_confidence.png")
        with open(sim_txt_save_path, 'w') as file:
            file.write(avg_sim_content)
        plot_similarity_histogram(avg_sim_list, bins=100, save_path=sim_plot_save_path, target=target)
        with open(conf_txt_save_path, 'w') as file:
            file.write(avg_conf_content)
        plot_confidence_histogram(avg_confidence_list, bins=100, save_path=conf_plot_save_path, target=target)


    def run_single_info(self, i):
        for target, target_categories in self.TARGET_DICTS.items():
            data_root = f'generated_results/{self.method_name}/nuscenes_cropped/gen{i}'
            save_root = f'generated_results/{self.method_name}/subject_consistency/gen{i}'
            os.makedirs(save_root, exist_ok=True)
            self.run_single_target(data_root, save_root, target, target_categories)

    def __call__(self):
        if self.need_preprocessing:
            from .extrac_nusc_2d_trac import extract_fg_objects
            for i in range(self.repeat_times):
                custom_info_path = f'generated_results/{self.method_name}/nuscenes_infos_temporal_val_3keyframes_gen{i}.pkl'
                extract_fg_objects(custom_info_path)
                logger.info(f"Extracting foreground objects for {self.method_name} gen{i} done.")

        for i in range(self.repeat_times):
            self.run_single_info(i)
            if i == 0 and self.method_name == 'gt':
                with open(self.select_img_path_save_path, 'w') as f:
                    json.dump(self.select_img_path, f, indent=4)
                logger.info(f"Selected image paths for GT saved to {self.select_img_path_save_path}.")
