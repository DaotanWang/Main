import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import models

# 设置图像文件夹路径
bacteria_folders = ['D:\\Download\\data\\6\\DeepHost_host_117','D:\\Download\\data\\7\\DeepHost_host_117','D:\\Download\\data\\8\\DeepHost_host_117']
phage_folders = ['D:\\Download\\data\\6\\DeepHost_train','D:\\Download\\data\\7\\DeepHost_train','D:\\Download\\data\\8\\DeepHost_train']

# 设置图像分辨率（用于缩放到统一大小）
image_sizes = [64, 128, 256]
target_size = 224 # 将所有图像统一到EfficientNet的输入大小

# 图像预处理和缩放
transform = transforms.Compose([
    transforms.Resize((target_size, target_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载图像并调整大小
def load_and_preprocess_images(folder_paths):
    image_data = {}
    for folder in folder_paths:
        for filename in os.listdir(folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0) # 处理图像并增加批次维度
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

# 使用同一个PCA降维，确保细菌和噬菌体的维度一致
def apply_pca(features, pca):
    feature_matrix = list(features.values())
    transformed_features = pca.transform(feature_matrix)
    reduced_features = {name: transformed_features[i] for i, name in enumerate(features.keys())}
    return reduced_features

# 计算噬菌体和细菌的相似性，并找到最相似的细菌
# 计算噬菌体和细菌的相似性，并找到最相似的细菌
def find_closest_bacteria(bacteria_features, phage_features):
    results = []
    for phage_name, phage_feature in phage_features.items():
        similarities = {b_name: cosine_similarity([phage_feature], [b_feature])[0][0]
                        for b_name, b_feature in bacteria_features.items()}
        closest_bacteria = max(similarities, key=similarities.get) # 选择余弦相似度最高的细菌
        results.append((phage_name, closest_bacteria))
    return results

# 输出结果到Excel
def save_results_to_excel(results, filename='results.xlsx'):
    df = pd.DataFrame(results, columns=['Phage', 'Closest Bacteria'])
    df.to_excel(filename, index=False)

# 主程序
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用预训练的 EfficientNet 模型
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1') # 使用权重替代pretrained参数
efficientnet = nn.Sequential(*list(efficientnet.children())[:-1]) # 删除全连接层，只保留卷积部分
efficientnet = efficientnet.to(device)

# 计算特征提取器的输出维度
dummy_input = torch.randn(1, 3, target_size, target_size).to(device) # 创建虚拟输入
dummy_output = efficientnet(dummy_input) # 通过 EfficientNet 提取特征
input_dim = dummy_output.shape[1] # 获取输出的特征维度

# 使用 AttentionFusion 进行特征融合
fusion_model = AttentionFusion(input_dim=input_dim).to(device)

# 加载图像数据
bacteria_images = load_and_preprocess_images(bacteria_folders)
phage_images = load_and_preprocess_images(phage_folders)

# 提取和融合特征
bacteria_features = extract_features(bacteria_images, efficientnet)
phage_features = extract_features(phage_images, efficientnet)

# 使用同一个PCA实例进行降维，确保维度一致
pca = PCA(n_components=128)
all_features = list(bacteria_features.values()) + list(phage_features.values())
pca.fit(all_features) # 在全部数据上进行拟合

bacteria_features_reduced = apply_pca(bacteria_features, pca)
phage_features_reduced = apply_pca(phage_features, pca)

# 打印降维后的特征维度，确保一致性
print("Bacteria features after PCA:", list(bacteria_features_reduced.values())[0].shape)
print("Phage features after PCA:", list(phage_features_reduced.values())[0].shape)

# 计算相似性并保存结果
results = find_closest_bacteria(bacteria_features_reduced, phage_features_reduced)
save_results_to_excel(results)
