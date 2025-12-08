import os
import math
import torch
import torch.nn as nn
import os.path as osp
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class BasicBlock_IBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(BasicBlock_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.IN = nn.InstanceNorm2d(planes, affine=True) if ibn == 'b' else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 ):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        # last_stride=1
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, ibn=ibn_cfg[3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.embed_dim = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks-1) else ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x



def resnet50_ibn_a():
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('a', 'a', 'a', None))
    return model


class build_model(nn.Module):
    def __init__(self, ckpt_path,neck_feat):
        super(build_model, self).__init__()
        self.ckpt_path = ckpt_path
        self.neck_feat = neck_feat
        self.base = resnet50_ibn_a()
        self.in_planes = self.base.embed_dim
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.load_param(self.ckpt_path)

    def forward(self, x):
        global_feat = self.base(x)
        feat = self.bottleneck(global_feat)

        if self.neck_feat == 'after':
            # print("Test with feature after BN")
            return feat
        else:
            # print("Test with feature before BN")
            return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu',weights_only=True)
        param_dict = {k: v for k, v in param_dict.items() if "classifier" not in k}
        self.load_state_dict(param_dict, strict=True)
        print(f"{self.ckpt_path} successfully loaded!")



def read_image(img_path,transform):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    img = transform(img)
    return img


def compute_identity_consistency(image_features, threshold=0.7):
    """
    计算图像序列的身份一致性

    参数:
    - image_features: 序列中所有图像的特征向量，形状为 [N, feature_dim]
    - threshold: 余弦相似度阈值，用于判断一致性

    返回:
    - avg_similarity: 平均相似度
    - consistency_score: 一致性分数 (0-1之间)
    - similarity_matrix: 相似度矩阵
    """
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(image_features)

    # 排除对角线元素（自身与自身的相似度为1）
    n = len(similarity_matrix)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    valid_similarities = similarity_matrix[mask]

    # 计算平均相似度
    avg_similarity = np.mean(valid_similarities)

    # 计算一致性分数：相似度大于阈值的比例
    consistency_score = np.mean(valid_similarities > threshold)

    return avg_similarity, consistency_score, similarity_matrix


def plot_similarity_histogram(similarity_data, bins=50, save_path='test.png',target='Pedestrian'):
    """绘制相似度分布柱状图"""
    # 计算平均值
    mean_value = np.mean(similarity_data)
    #mean_value_rounded = round(mean_value, 4)  # 保留4位小数

    # 创建画布
    plt.figure(figsize=(10, 6))

    # 绘制柱状图
    n, bins, patches = plt.hist(similarity_data, bins=bins, edgecolor='black', alpha=0.7)

    # 绘制平均值竖线（虚线）
    plt.axvline(
        x=mean_value,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Average: {mean_value}'  # 图例标签
    )

    # 设置标题和轴标签
    plt.title(f'{target} Similarity Distribution', fontsize=16)
    plt.xlabel('Similarity', fontsize=14)
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
    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")


def reidentification_similarity_calculation():
    target = "Vehicle"    # choose from ['Pedestrian','Vehicle']
    ckpt_path = f"ckpt/{target.lower()}_reid.pth"
    data_root = 'nuscenes_gt/gen0'
    save_root = "eval_results/"

    pedestrian_categories = ["pedestrian"]
    vehicle_categories = ["car","construction_vehicle","bus","traiycle","truck"]
    assert target in ['Pedestrian', 'Vehicle']
    if target == 'Pedestrian':
        target_categories = pedestrian_categories
    else:
        target_categories = vehicle_categories
    os.makedirs(save_root, exist_ok=True)

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    model = build_model(ckpt_path,neck_feat='before').to('cuda')
    model.eval()

    tracklet_dirs = os.listdir(data_root)
    target_dirs = [x for x in tracklet_dirs if "_".join(x.split("_")[1:]) in target_categories]
    target_dirs = sorted(target_dirs)
    avg_sim_list = []
    avg_content = ""

    statistics = {}
    for tracklet in tracklet_dirs:
        category = tracklet.split("_")[1:]
        category = "_".join(category)
        if category not in statistics.keys():
            statistics[category]=1
        else:
            statistics[category]+=1
    print(statistics)

    for dir in target_dirs:
        img_paths = os.listdir(os.path.join(data_root,dir))
        if len(img_paths)<=1:
            continue
        feats_list = []
        for img_path in img_paths:
            img_path = os.path.join(data_root,dir,img_path)
            img = read_image(img_path,val_transforms).to('cuda').unsqueeze(0)
            feat = model(img).cpu().detach()
            feats_list.append(feat)
        feats = torch.cat(feats_list,dim=0)
        avg_sim, consistency, sim_matrix = compute_identity_consistency(feats)
        print(f"{dir}: {avg_sim}, {consistency}")
        avg_content += f"{dir}: {avg_sim}\n"
        avg_sim_list.append(round(avg_sim,4))

    avg_sim_list = sorted(avg_sim_list)
    print(f"Total Number of {target}s:{len(avg_sim_list)}")
    print(f"Average Similarities:{sum(avg_sim_list)/len(avg_sim_list)}")

    avg_content += f"Total Number of {target}s: {len(avg_sim_list)}\n"
    avg_content += f"Average Similarities: {sum(avg_sim_list)/len(avg_sim_list)}"

    txt_save_path = os.path.join(save_root,"_".join(data_root.split("/")) + f"_{target}_similarity.txt")
    plot_save_path = os.path.join(save_root,"_".join(data_root.split("/")) + f"_{target}_similarity.png")
    with open(txt_save_path, 'w') as file:
        file.write(avg_content)

    plot_similarity_histogram(avg_sim_list, bins=100, save_path=plot_save_path,target=target)