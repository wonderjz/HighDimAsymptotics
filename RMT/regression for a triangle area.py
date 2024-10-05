#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 22:58:29 2023

@author: lijinze
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import random



reg = []
for i in range(300000):
    x = random.random()
    y = random.uniform(0,x)
    # y = min(random.random(),x)
    reg.append([x,y])

regdf = pd.DataFrame(reg)

# y = []
# for i in range(3000):
#     y.append(random.random())
    
    
# datas = pd.read_excel(r'D:\Users\chen_\git\Statistics-book\datas\linear_regression.xlsx') # 读取 excel 数据，引号里面是 excel 文件的位置
# y = datas.iloc[:, 1] # 因变量为第 2 列数据
# x = datas.iloc[:, 2] # 自变量为第 3 列数据

x = regdf.iloc[:,0]
y = regdf.iloc[:,1]
x = sm.add_constant(x) # 若模型中有截距，必须有这一步
model = sm.OLS(y, x).fit() # 构建最小二乘模型并拟合
print(model.summary()) # 输出回归结果

# 画图
# 这两行代码在画图时添加中文必须用
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

predicts = model.predict() # 模型的预测值
x = regdf.iloc[:,0] # 自变量为第 3 列数据
plt.scatter(x, y, label='实际值') # 散点图
plt.plot(x, predicts, color = 'red', label='预测值')
plt.legend() # 显示图例，即每条线对应 label 中的内容
plt.show() # 显示图形
