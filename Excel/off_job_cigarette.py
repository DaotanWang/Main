import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. 数据加载
data = pd.read_excel("C:\\Users\\daota\\Desktop\\data.xlsx", header=None)
data.columns = ["A"] + ["B"] + ["C"] + [f"D{i}" for i in range(2179)]


# 2. 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)  # 防止过拟合
        self.fc1 = None  # 暂时设置为 None，稍后根据输入形状动态初始化
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))

        # 展平
        x = x.view(x.size(0), -1)

        # 动态初始化 fc1
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)  # 根据展平后的尺寸动态调整

        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 在循环外部初始化一个字典来记录最佳模型和R方
best_models = {}

# 3. 训练和评估
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for category in data['A'].unique():
    # 筛选当前类别的数据
    category_data = data[data['A'] == category]
    X = category_data.iloc[:, 3:].values  # 输入特征 D 到 CEX 列
    y = category_data['C'].values  # 目标值 C 列

    # 标准化输入特征和归一化目标
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 确保分割后数据集不为空
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        print(f"Error: No samples in training set for category {category}.")
        continue  # 跳过此类别

    # 转换为张量
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 变为 (样本数, 1, 2179)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # 加载数据
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = CNNModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    # 初始化记录每个类别的最高R方值
    best_r2 = -float('inf')

    # 训练模型
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            y_pred_list = []
            y_true_list = []
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).cpu().numpy()
                y_pred_list.extend(outputs.flatten())
                y_true_list.extend(targets.numpy().flatten())

            # 逆归一化目标值
            y_pred_list = scaler_y.inverse_transform(np.array(y_pred_list).reshape(-1, 1)).flatten()
            y_true_list = scaler_y.inverse_transform(np.array(y_true_list).reshape(-1, 1)).flatten()

            # 计算R方
            r2 = r2_score(y_true_list, y_pred_list)
            print(f"类别 {category} - Epoch [{epoch + 1}/{num_epochs}], R方: {r2}, Loss: {loss.item()}")

            # 动态调整学习率
            scheduler.step(loss)

            # R方达到0.8时保存模型，并更新最佳记录
            if r2 >= 0.8:
                torch.save(model.state_dict(), f"model_category_{category}.pth")
                print(f"类别 {category} 的模型已保存，R方：{r2}")
                best_r2 = r2  # 更新最佳R方
                best_models[category] = (f"model_category_{category}.pth", best_r2)
                break  # 提前停止训练

            # 若没有达到0.8，但R方高于当前最高值，则更新记录（非保存模型）
            if r2 > best_r2:
                best_r2 = r2
                best_models[category] = (f"model_category_{category}.pth", best_r2)

# 程序结束时，打印每种产品的最优模型和对应的R方
print("\n每种产品的最优模型及其R方值：")
for category, (model_path, r2) in best_models.items():
    print(f"产品类别 {category} 的最佳模型路径: {model_path}, R方: {r2}")
