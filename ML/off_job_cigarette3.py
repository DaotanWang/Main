import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint

# 加载数据
data = pd.read_excel('d:\\user\\tc029861\\桌面\\data1111.xlsx', header=None)
data.columns = ['Product_Category'] + [f'Input_{i}' for i in range(data.shape[1] - 2)] + ['Result']

# 存储每个产品类别的 R²
r2_scores = {}
categories = data['Product_Category'].unique()

# 定义一个函数来训练堆叠模型并返回 R²
def train_stacked_model(category_data):
    X = category_data.iloc[:, 1:-1].values
    y = category_data.iloc[:, -1].values

    # 检查是否有缺失值
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print(f'Product Category: {category} has NaN values. Skipping this category.')
        return None

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.207, random_state=42)

    # 简化的参数范围
    param_dist_xgb = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(2, 6),
        'learning_rate': uniform(0.01, 0.1),
        'subsample': uniform(0.7, 0.3)
    }
    param_dist_rf = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(2, 6),
        'max_features': ['auto', 'sqrt']
    }

    # 使用随机搜索进行模型调优
    xgb_random_search = RandomizedSearchCV(
        XGBRegressor(random_state=42), param_distributions=param_dist_xgb,
        n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42
    )
    rf_random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42), param_distributions=param_dist_rf,
        n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42
    )

    xgb_random_search.fit(X_train, y_train)
    rf_random_search.fit(X_train, y_train)

    best_xgb = xgb_random_search.best_estimator_
    best_rf = rf_random_search.best_estimator_

    # 获得基础模型的预测结果
    xgb_pred = best_xgb.predict(X_test).reshape(-1, 1)
    rf_pred = best_rf.predict(X_test).reshape(-1, 1)

    # 将基础模型的预测结果作为新的特征
    stacked_features = np.hstack((xgb_pred, rf_pred))

    # 使用线性回归作为元模型
    scaler = StandardScaler()
    stacked_features_scaled = scaler.fit_transform(stacked_features)

    meta_model = LinearRegression()
    meta_model.fit(stacked_features_scaled, y_test)
    final_predictions = meta_model.predict(stacked_features_scaled)

    # 计算 R²
    r2 = r2_score(y_test, final_predictions)
    return r2

# 对每个产品类别训练堆叠模型
for category in categories:
    category_data = data[data['Product_Category'] == category]
    r2 = train_stacked_model(category_data)
    if r2 is not None:
        r2_scores[category] = r2
        print(f'Product Category: {category}, Stacked Model R²: {r2:.4f}')

# 打印每个产品类别的最优堆叠模型的 R²
for category in r2_scores:
    print(f'最优堆叠模型：{category}, R²: {r2_scores[category]:.4f}')
