import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

# 读取数据
file_path = 'd:\\user\\tc029861\\桌面\\data.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path, header=None)

# 数据预处理
data.columns = ['Product_Category'] + [f'Input_{i}' for i in range(data.shape[1] - 2)] + ['Result']

# 获取产品类别列表
categories = data['Product_Category'].unique()

# 定义神经网络模型
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 增加神经元数量
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # 额外的隐藏层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 存储每个模型的结果
models = {}
r2_scores = {}

# 对每种产品类别训练模型
for category in categories:
    category_data = data[data['Product_Category'] == category]

    # 分离特征和标签
    X = category_data.iloc[:, 1:-1].values  # D列到CEX列
    y = category_data.iloc[:, -1].values     # C列

    # 检查数据是否有缺失值
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print(f'Product Category: {category} has NaN values. Skipping this category.')
        continue

    # KFold 交叉验证
    kf = KFold(n_splits=5)
    r2_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 转换为 PyTorch 张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

        # 初始化模型、损失函数和优化器
        model = RegressionModel(input_size=X_train.shape[1])
        criterion = nn.MSELoss()  # 可以考虑使用 HuberLoss
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        num_epochs = 5000  # 增加训练周期
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # 测试模型并计算 R²
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)
            r2 = r2_score(y_test, predictions.numpy())
            r2_list.append(r2)

    # 计算平均 R²
    avg_r2 = np.mean(r2_list)
    models[category] = model
    r2_scores[category] = avg_r2
    print(f'Product Category: {category}, Average R²: {avg_r2:.4f}')

# 输出每种产品的最优模型和 R²
for category in models:
    print(f'最优模型：{category}, 平均 R²: {r2_scores[category]:.4f}')
