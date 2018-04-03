from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
from decimal import Decimal

filename = r'./d_train_20180102.csv'
df = pd.read_csv(filename, encoding='gb2312', na_values='NULL')  # 0~35行数据不读取 skiprows=[36]
df = df.drop(['id', '体检日期', '乙肝表面抗原', '乙肝表面抗体',
              '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体', '血小板计数',
              '血小板平均体积', '血小板体积分布宽度', '血小板比积',
              '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
              '白细胞计数', '红细胞压积', '红细胞平均血红蛋白量'], axis=1)
sex = {
    '男': 1,
    '女': 0,
}
df['性别'] = df['性别'].map(sex)
df = df.fillna(method='pad')      # 数据向上填充
# df = df.dropna()        # 去除缺失值
row = 22
train_x = df.iloc[:, :row].as_matrix()
train_x = np.array(train_x)
train_y = df.iloc[:, row:].as_matrix()
train_y = np.array(train_y.reshape(-1, 1))
'''
test_x = df.iloc[4000:, :row].as_matrix()
test_x = np.array(test_x)
test_y = df.iloc[4000:, :row].as_matrix()
test_y = np.array(test_y.reshape(-1, 1))
'''
df2 = pd.read_csv(r'./d_test_A_20180102.csv', encoding='gb2312')
df2 = df2.drop(['id', '体检日期', '乙肝表面抗原', '乙肝表面抗体',
                '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体', '血小板计数',
                '血小板平均体积', '血小板体积分布宽度', '血小板比积',
                '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
                '白细胞计数', '红细胞压积', '红细胞平均血红蛋白量'], axis=1)

df2['性别'] = df2['性别'].map(sex)
test_x = df2.iloc[:, :22].as_matrix()
test_x = np.array(test_x)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)        # 用均值填充缺失值
imp.fit(test_x)
test_x = imp.transform(test_x)
'''
print('#########################将训练数据分成训练集和测试集使用网格优化选择参数##############')
model_gbr_GridSearch = GradientBoostingRegressor()
params_grid = {
            'n_estimators': range(20, 81, 10),
            'learning_rate': [0.2, 0.1, 0.005, 0.002, 0.001],
            'max_depth': [4, 6, 8, 10],
            'min_samples_leaf': [3, 5, 9, 14],
            'max_features': [0.8, 0.5, 0.3, 0.1]
            }
estimator = GridSearchCV(model_gbr_GridSearch, params_grid)
estimator.fit(train_x, train_y.ravel())
print(estimator.best_params_)
# {'learning_rate': 0.1, 'max_depth': 4, 'max_features': 0.1, 'min_samples_leaf': 5, 'n_estimators': 80}
'''
model_gbr_best = GradientBoostingRegressor(learning_rate=0.1, max_depth=4, max_features=0.1, min_samples_leaf=5, n_estimators=80)
model_gbr_best.fit(train_x, train_y.ravel())

gbr_pridict_disorder = model_gbr_best.predict(test_x)       # 使用默认参数的模型进行预测
data = pd.DataFrame(gbr_pridict_disorder)
data = np.array(data)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data[i][j] = round(data[i][j], 3)
print(data)
data = pd.DataFrame(data)
data.to_csv('C:/GradientBoostingRegressor28.csv')