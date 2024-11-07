import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# 读取数据
file_path = 'C:\\Users\\daota\\Desktop\\data11.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path, header=None)

# 数据预处理
data.columns = ['Product_Category'] + [f'Input_{i}' for i in range(data.shape[1] - 2)] + ['Result']

# 获取产品类别列表
categories = data['Product_Category'].unique()

# 存储每个模型的结果
r2_scores = {}

# 对每种产品类别训练模型
for category in categories:
    category_data = data[data['Product_Category'] == category]

    # 分离特征和标签
    X = category_data.iloc[:, 1:-1].values  # D列到CEX列
    y = category_data.iloc[:, -1].values  # C列

    # 检查数据是否有缺失值
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print(f'Product Category: {category} has NaN values. Skipping this category.')
        continue

    # 随机拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数据标准化（可选）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 初始化梯度提升回归模型
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 测试模型并计算 R²
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)

    r2_scores[category] = r2
    print(f'Product Category: {category}, R²: {r2:.4f}')

# 输出每种产品的 R²
for category in r2_scores:
    print(f'最优模型：{category}, R²: {r2_scores[category]:.4f}')
