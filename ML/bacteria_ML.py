import os

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image # 导入PIL库用于图像处理
from torchvision import models

# 设置图像文件夹路径
bacteria_folders = ['D:\\Download\\data\\6\\DeepHost_host_117','D:\\Download\\data\\7\\DeepHost_host_117','D:\\Download\\data\\8\\DeepHost_host_117']
phage_folders = ['D:\\Download\\data\\6\\DeepHost_train','D:\\Download\\data\\7\\DeepHost_train','D:\\Download\\data\\8\\DeepHost_train']

# 设置图像分辨率（用于缩放到统一大小）
image_sizes = [64, 128, 256]
target_size = 64 # 将所有图像统一到64x64大小

# 图像预处理和缩放
transform = transforms.Compose([
    transforms.Resize((target_size, target_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 使用预训练模型所需的标准化
])

# 加载图像并调整大小
def load_and_preprocess_images(folder_paths, image_sizes):
    image_data = {}
    for folder, size in zip(folder_paths, image_sizes):
        for filename in os.listdir(folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, (target_size, target_size))

                # 将 numpy 图像转换为 PIL 图像
                img_pil = Image.fromarray(img_resized).convert('RGB') # 转换为 RGB 格式

                # 应用 transforms
                img_tensor = transform(img_pil).unsqueeze(0)

                if filename not in image_data:
                    image_data[filename] = []
                image_data[filename].append(img_tensor)
    return image_data

# 使用注意力机制融合特征
class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(3, input_dim))

    def forward(self, features):
        weights = torch.softmax(self.attention_weights, dim=0)
        fused_feature = sum(w * f for w, f in zip(weights, features))
        return fused_feature

# 提取融合特征
def extract_features(image_data, model):
    fused_features = {}
    for name, images in image_data.items():
        images = [img.to(device) for img in images]
        with torch.no_grad():
            features = [model(img).squeeze() for img in images]
            fused_feature = fusion_model(features).cpu()
            fused_features[name] = fused_feature.numpy()
    return fused_features

# 计算噬菌体和细菌的相似性，并找到最相似的细菌
def find_closest_bacteria(bacteria_features, phage_features):
    results = []
    for phage_name, phage_feature in phage_features.items():
        closest_bacteria = min(bacteria_features, key=lambda b: euclidean_distances([phage_feature], [bacteria_features[b]])[0][0])
        results.append((phage_name, closest_bacteria))
    return results

# 输出结果到Excel
def save_results_to_excel(results, filename='results.xlsx'):
    df = pd.DataFrame(results, columns=['Phage', 'Closest Bacteria'])
    df.to_excel(filename, index=False)

# 主程序
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用预训练的 ResNet18 模型
resnet18 = models.resnet18(pretrained=True) # 加载预训练的 ResNet18
resnet18 = nn.Sequential(*list(resnet18.children())[:-1]) # 删除 ResNet18 的最后全连接层，只保留卷积部分
resnet18 = resnet18.to(device)

# 计算特征提取器的输出维度
dummy_input = torch.randn(1, 3, target_size, target_size).to(device) # 创建一个虚拟输入（RGB 图像）
dummy_output = resnet18(dummy_input) # 通过 ResNet18 提取特征
input_dim = dummy_output.shape[1] # 获取输出的特征维度

# 使用计算出的 input_dim 来初始化 AttentionFusion
fusion_model = AttentionFusion(input_dim=input_dim).to(device)

# 加载图像数据
bacteria_images = load_and_preprocess_images(bacteria_folders, image_sizes)
phage_images = load_and_preprocess_images(phage_folders, image_sizes)

# 特征融合
bacteria_features = extract_features(bacteria_images, resnet18)
phage_features = extract_features(phage_images, resnet18)

# 计算相似性并保存结果
results = find_closest_bacteria(bacteria_features, phage_features)
save_results_to_excel(results)
