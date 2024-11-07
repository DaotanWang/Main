import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# 假设您已经有了一个名为data.xlsx的Excel文件
# 读取Excel文件
df = pd.read_excel('d:\\user\\tc029861\\桌面\\data1.xlsx')

# 查看数据结构
print(df.head())

# 分离特征和目标
X = df[['参数1', '参数2', '参数3']]
y = df[['目标量1', '目标量2']]

# 使用固定的目标值
target_1, target_2 = 260, 20  # 原始目标值
normalized_target = np.array([1.3, 1])  # 归一化目标值

# 定义训练模型
model = MultiOutputRegressor(RandomForestRegressor())
model.fit(X, y)


# 定义损失函数（归一化目标值）
def objective(params):
    param1, param2, param3 = map(int, params)  # 将参数强制转换为整数
    # 进行预测
    pred = model.predict(np.array([[param1, param2, param3]]))[0]

    # 归一化预测值
    norm_pred1 = pred[0] / target_1  # 将目标1归一化到接近1
    norm_pred2 = pred[1] / target_2  # 将目标2归一化到接近1

    # 计算归一化误差
    error1 = (norm_pred1 - normalized_target[0]) ** 2  # 目标1的归一化误差
    error2 = (norm_pred2 - normalized_target[1]) ** 2  # 目标2的归一化误差

    return error1 + error2  # 总误差


# 参数范围
param_bounds = [(380, 480), (200, 300), (1, 15)]

# 记录最佳结果
best_result = None
best_error = float('inf')

# 多次优化
for i in range(1000):  # 进行100次优化
    # 初始参数（强制为整数）
    initial_guess = np.random.randint(low=[380, 200, 1], high=[481, 301, 16])  # 使用 randint 确保是整数

    # 进行优化
    result = minimize(objective, initial_guess, bounds=param_bounds)

    # 如果找到更好的结果，则更新最佳结果
    if result.success and result.fun < best_error:
        best_error = result.fun
        best_result = result.x

# 使用最佳参数进行最终预测
final_predictions = model.predict(np.array([[int(best_result[0]), int(best_result[1]), int(best_result[2])]]))

# 最终输出
print(f"最佳组合: 参数1={int(best_result[0])}, 参数2={int(best_result[1])}, 参数3={int(best_result[2])}")
print(f"预测的目标1值: {final_predictions[0][0]:.2f}")
print(f"预测的目标2值: {final_predictions[0][1]:.2f}")
print(f"归一误差: {best_error:.6f}")
