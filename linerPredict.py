# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

def predict():
    filename = r'./train.csv'
    df = pd.read_csv(filename, encoding='gb2312', na_values='NULL')  # 0~35行数据不读取 skiprows=[36]
    df = df.drop(['体检日期',   '乙肝表面抗原', '乙肝表面抗体',
                                    '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体', '血小板计数',
                                    '血小板平均体积', '血小板体积分布宽度', '血小板比积',
                                    '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
                                    '白细胞计数',  '红细胞压积', '红细胞平均血红蛋白量'], axis=1)
    sex = {
        '男': 1,
        '女': 0,
    }
    df['性别'] = df['性别'].map(sex)
    df = df.fillna(method='pad')
    row = 22
    train_x = df.iloc[:, :row].as_matrix()
    train_x = np.array(train_x)
    train_y = df.iloc[:, row:].as_matrix()
    train_y = np.array(train_y)
    df2 = pd.read_csv(r'./d_test_A_20180102.csv', encoding='gb2312')
    df2 = df2.drop(['id', '体检日期',   '乙肝表面抗原', '乙肝表面抗体',
                                    '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体', '血小板计数',
                                    '血小板平均体积', '血小板体积分布宽度', '血小板比积',
                                    '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%',
                                    '白细胞计数',  '红细胞压积', '红细胞平均血红蛋白量'], axis=1)

    df2['性别'] = df2['性别'].map(sex)
    test_x = df2.iloc[:, :22].as_matrix()
    test_x = np.array(test_x)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(test_x)
    test_x = imp.transform(test_x)
    # print(test_x)
    linreg = LinearRegression()
    linreg.fit(train_x, train_y)
    test_y_predict = linreg.predict(test_x)
    test_y_predict = np.array(test_y_predict)
    data = pd.DataFrame(test_y_predict)
    print(data)
    data.to_csv('C:/predict8.csv')

if __name__ ==  '__main__':
    predict()