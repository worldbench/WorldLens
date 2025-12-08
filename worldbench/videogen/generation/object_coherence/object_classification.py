import logging
from PIL import Image
import os
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def get_val_transform(image_size: list = [256, 128],normalize_mean: list = [0.5, 0.5, 0.5],normalize_std: list = [0.5, 0.5, 0.5]):
    """
    创建验证集的图像变换管道（无增强，仅基础预处理）

    参数:
        image_size: 图像缩放尺寸 [height, width]
        normalize_mean: 归一化均值
        normalize_std: 归一化标准差
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])


class Checkpoint:
    """Checkpoint 管理类，负责 checkpoint 的保存、加载和训练状态恢复"""
    def __init__(self, save_dir: Optional[str] = None):
        if save_dir is not None:
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
        self.logger = logging.getLogger("pbc.checkpoint")

    @staticmethod
    def load(checkpoint_path: str, model_class, base_model=None) -> torch.nn.Module:
        """
        仅加载模型权重（供推理用）

        Args:
            checkpoint_path: checkpoint 文件路径
            model_class: 模型类（如 PedestrianClassifier）
            base_model: 基础模型设置（可选参数）。若设置，则加载过程优先该参数

        Returns:
            加载好权重的模型实例
        """
        # 直接加载模型权重（兼容 full 和 model 模式保存的文件）
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # temp_checkpoint = torch.load(checkpoint_path.replace('final', 'classification'), map_location="cpu")
        # 从 model_args 重建模型并加载权重
        if base_model is None:
            model = model_class.from_args(checkpoint["model_args"])
            # model = model_class.from_args(temp_checkpoint["model_args"])
        else:
            model = model_class.from_args({"base_model": base_model})
        model.load_state_dict(checkpoint["model_state_dict"])
        # model.load_state_dict(checkpoint)

        print(f"Model weights loaded from {checkpoint_path}")
        return model


@dataclass(frozen=True)
class ClassifierResults:
    """模型推理结果数据类"""

    output: float

    @property
    def predicted_class(self):
        return 1 if self.output > 0.5 else 0

    @property
    def confidence(self):
        #return self.output if self.predicted_class == 1 else (1 - self.output)
        return self.output

    def to_dict(self) -> dict:
        """转换为字典（过滤值为None的字段）"""
        result = {}
        result["output"] = self.output.item()
        result["predicted_class"] = self.predicted_class
        result["confidence"] = self.confidence
        return result


class Predictor:
    def __init__(
        self, model: torch.nn.Module, transform: transforms.Compose, device="cpu"
    ):
        """
        行人分类预测器

        参数:
            model_path: 模型权重文件路径
            model_class: 模型类
            device: 运行设备
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        # 定义预处理转换
        self.transform = transform

    def predict(self, image_path, as_dict=False):
        """
        对单张图像进行预测

        参数:
            image_path: 图像路径

        返回:
            dict: 包含预测结果的字典
        """
        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 模型推理
        with torch.no_grad():
            output = self.model(image_tensor).squeeze().item()

        return (
            ClassifierResults(output).to_dict() if as_dict else ClassifierResults(output)
        )

    def predict_batch(self, image_paths, as_dict=False):
        """
        对一批图像进行预测

        参数:
            image_paths: 图像路径列表

        返回:
            list: 包含每个图像预测结果的列表
        """
        results = []
        for path in image_paths:
            results.append(self.predict(path, as_dict=as_dict))
        return results


class PedestrianClassifier(nn.Module):
    def __init__(self, base_model="resnet50", pretrained=False, freeze_layers=False):
        """
        行人二分类模型初始化

        参数:
            base_model: 基础ResNet模型类型，可选'resnet18', 'resnet34', 'resnet50'
            pretrained: 是否使用预训练权重
            freeze_layers: 是否冻结预训练模型的层
        """
        super(PedestrianClassifier, self).__init__()

        # 保存模型初始化参数（用于从checkpoint重建）
        self.model_args = {
            "base_model": base_model,
            "pretrained": pretrained,
            "freeze_layers": freeze_layers,
        }

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

        # 选择基础模型
        self._set_base_model(base_model, weights, freeze_layers)

        # 修改分类头以适应二分类任务
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # 可调整dropout比例
            nn.Linear(num_ftrs, 1),  # 二分类只需要一个输出
            nn.Sigmoid(),  # 使用Sigmoid激活函数输出概率
        )

    def _set_base_model(self, base_model, weights, freeze_layers):
        if base_model == "resnet18":
            self.model = resnet18(weights=weights)
        elif base_model == "resnet34":
            self.model = resnet34(weights=weights)
        elif base_model == "resnet50":
            self.model = resnet50(weights=weights)
        else:
            raise ValueError("Unsupported model types")

        # 冻结预训练模型的层
        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image):
        """模型前向传播"""
        return self.model(image)

    def get_trainable_params(self):
        """获取可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]

    @classmethod
    def from_args(cls, args):
        """从参数字典创建模型实例（用于checkpoint加载）"""
        return cls(**args)


def plot_confidence_histogram(confidence_data, bins=50, save_path='test.png',target='Pedestrian'):
    # 计算平均值
    mean_value = np.mean(confidence_data)
    #mean_value_rounded = round(mean_value, 4)  # 保留4位小数

    # 创建画布
    plt.figure(figsize=(10, 6))

    # 绘制柱状图
    n, bins, patches = plt.hist(confidence_data, bins=bins, edgecolor='black', alpha=0.7)

    # 绘制平均值竖线（虚线）
    plt.axvline(
        x=mean_value,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Average: {mean_value}'  # 图例标签
    )

    # 设置标题和轴标签
    plt.title(f'{target} Confidence Distribution', fontsize=16)
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Number', fontsize=14)

    # 设置刻度标签
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 为每个柱子添加数值标签
    for i in range(len(patches)):
        if n[i] > 0:  # 只在柱子高度大于0时添加标签
            plt.text(patches[i].get_x() + patches[i].get_width() / 2,
                     n[i] + max(n) * 0.01,
                     f'{int(n[i])}',
                     ha='center',
                     fontsize=5)

    # 添加图例（显示平均值）
    plt.legend(fontsize=12)
    # 显示图形
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")


def classification_confidence_calculation():
    target = "Pedestrian"  # choose from ['Pedestrian','Vehicle']
    ckpt_path = f"ckpt/{target.lower()}_classification.pth"
    data_root = 'nuscenes_gt/gen0'
    save_root = "eval_results/"

    pedestrian_categories = ["pedestrian"]
    vehicle_categories = ["car", "construction_vehicle", "bus", "traiycle", "truck"]
    assert target in ['Pedestrian', 'Vehicle']
    if target == 'Pedestrian':
        target_categories = pedestrian_categories
    else:
        target_categories = vehicle_categories
    os.makedirs(save_root, exist_ok=True)

    model = Checkpoint.load(ckpt_path, model_class=PedestrianClassifier)
    transform = get_val_transform()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = Predictor(model=model, transform=transform, device=device)

    tracklet_dirs = os.listdir(data_root)
    target_dirs = [x for x in tracklet_dirs if "_".join(x.split("_")[1:]) in target_categories]
    target_dirs = sorted(target_dirs)

    statistics = {}
    for tracklet in tracklet_dirs:
        category = tracklet.split("_")[1:]
        category = "_".join(category)
        if category not in statistics.keys():
            statistics[category] = 1
        else:
            statistics[category] += 1
    print(statistics)

    avg_confidence_list = []
    avg_content = ""

    for dir in target_dirs:
        img_paths = os.listdir(os.path.join(data_root, dir))
        if len(img_paths) <= 1:
            continue
        confidence_list = []
        for img_path in img_paths:
            img_path = os.path.join(data_root, dir, img_path)
            result = predictor.predict(image_path=img_path, as_dict=False)
            predict = result.predicted_class
            confidence = result.confidence
            print(img_path,predict,confidence)
            confidence_list.append(round(confidence, 4))
        avg_conf = sum(confidence_list) / len(confidence_list)
        avg_confidence_list.append(round(avg_conf,4))
        avg_content += f"{dir}: {avg_conf}\n"


    avg_confidence_list = sorted(avg_confidence_list)
    print(f"Total Number of {target}s:{len(avg_confidence_list)}")
    print(f"Average Confidence:{sum(avg_confidence_list) / len(avg_confidence_list)}")

    avg_content += f"Total Number of {target}s: {len(avg_confidence_list)}\n"
    avg_content += f"Average Confidence: {sum(avg_confidence_list) / len(avg_confidence_list)}"

    txt_save_path = os.path.join(save_root, "_".join(data_root.split("/")) + f"_{target}_confidence.txt")
    plot_save_path = os.path.join(save_root, "_".join(data_root.split("/")) + f"_{target}_confidence.png")
    with open(txt_save_path, 'w') as file:
        file.write(avg_content)

    plot_confidence_histogram(avg_confidence_list, bins=100, save_path=plot_save_path, target=target)