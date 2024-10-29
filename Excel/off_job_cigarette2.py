from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

# 假设 data 是您加载的 DataFrame
data = pd.read_excel('C:\\Users\\daota\\Desktop\\data111.xlsx', header=None)
data.columns = ['Product_Category'] + [f'Input_{i}' for i in range(data.shape[1] - 2)] + ['Result']

# 存储每个产品类别的 R²
r2_scores = {}
categories = data['Product_Category'].unique()

for category in categories:
    category_data = data[data['Product_Category'] == category]
    X = category_data.iloc[:, 1:-1].values
    y = category_data.iloc[:, -1].values

    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print(f'Product Category: {category} has NaN values. Skipping this category.')
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.265, random_state=42)

    # 基础模型
    xgb = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.01, random_state=42)
    rf = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=42)

    # 训练基础模型
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # 获得基础模型的预测结果
    xgb_pred = xgb.predict(X_test).reshape(-1, 1)
    rf_pred = rf.predict(X_test).reshape(-1, 1)

    # 将基础模型的预测结果作为新的特征
    stacked_features = np.hstack((xgb_pred, rf_pred))

    # 使用线性回归作为最终的堆叠模型
    meta_model = LinearRegression()
    meta_model.fit(stacked_features, y_test)
    final_predictions = meta_model.predict(stacked_features)

    # 计算 R²
    r2 = r2_score(y_test, final_predictions)
    r2_scores[category] = r2
    print(f'Product Category: {category}, Stacked Model R²: {r2:.4f}')

for category in r2_scores:
    print(f'最优堆叠模型：{category}, R²: {r2_scores[category]:.4f}')
