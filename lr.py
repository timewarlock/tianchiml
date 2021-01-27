#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-27 18:52:08
# @Author  : Shuaikang Cao (seacon@163.com)
# @Link    : https://github.com/timewarlock
# @Version : $Id$

#基于逻辑的回归预测
'''
1.介绍
  logistic regression 是一个分类模型 
  突出特点 ：模型简单 可解释性强
  优点： 实现简单 易于理解和实现 计算代价低 速度快 存储资源低
  缺点： 容易欠拟合  分类精度可能不高
2.应用
  lr是很多分类算法的基础组件 例：基于GBDT算法+LR的信用卡交易反欺诈 CTR点击通过率预测等 好处在于输出值在0-1之间有概率意义
  模型清晰有概率学理论基础
  它拟合出的参数就代表每个特征对结果的影响，也是一个理解数据的工具
  本质上是线性分类器 所以 不能应对较为复杂的数据情况 可作为任务尝试的基线
 3.学习
    推导
    理论
    sklearn 函数运用到鸢尾花数据集
4.代码流程 
    Part1 Demo实践
        Step1:库函数导入
        Step2:模型训练
        Step3:模型参数查看
        Step4:数据和模型可视化
        Step5:模型预测

    Part2 基于鸢尾花（iris）数据集的逻辑回归分类实践
        Step1:库函数导入
        Step2:数据读取/载入
        Step3:数据信息简单查看
        Step4:可视化描述
        Step5:利用 逻辑回归模型 在二分类上 进行训练和预测
        Step5:利用 逻辑回归模型 在三分类(多分类)上 进行训练和预测

'''
#demo1
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from  sklearn.linear_model import LogisticRegression
x_features = np.array([[-1,-2],[-2,-1],[-3,-2],[1,3],[2,1],[3,2]])
y_label = np.array([0,0,0,1,1,1])
lr_clf = LogisticRegression()
lr_clf = lr_clf.fit(x_features, y_label)
print("the weight of logistic regression:",lr_clf.coef_)
print("the intercept if logistic regression:",lr_clf.intercept_)

#可视化数据样本点
plt.figure()
plt.scatter(x_features[:,0],x_features[:,1],c=y_label,s=50,cmap="viridis")
plt.title("Dataset")
plt.show()

#可视化决策边界
plt.figure()
plt.scatter(x_features[:,0],x_features[:,1], c=y_label, s=50, cmap="viridis")
plt.title("Dataset")

Nx, Ny = 200, 100
x_min, x_max = plt.xlim()
print(x_min,x_max)
y_min, y_max = plt.ylim() 
x_grid, y_grid = np.meshgrid(np.linspace(x_min,x_max, Nx), np.linspace(y_min, y_max, Ny))

p_proba = lr_clf.predict_proba(np.c_[x_grid.ravel(),y_grid.ravel()])
p_proba = p_proba[:,1].reshape(x_grid.shape)
plt.contour(x_grid, y_grid, p_proba, [0.5], linewidths=2, colors="blue")
plt.show()

#可视化新样本
plt.figure()
x_features_new1 = np.array([[0,-1]])
plt.scatter(x_features_new1[:,0],x_features_new1[:,1], s=50, cmap="viridis")
plt.annotate(s="New point1",xy=(0,-1),xytext=(-2,0),color="blue",arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3",color="red"))

x_features_new2 = np.array([[1,2]])
plt.scatter(x_features_new2[:,0],x_features_new2[:,1], s=50, cmap='viridis')
plt.annotate(s='New point 2',xy=(1,2),xytext=(-1.5,2.5),color='red',bbox=dict(boxstyle="rarrow"),arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='red'))

#训练样本

plt.scatter(x_features[:,0],x_features[:,1], c=y_label, s=100, cmap='viridis')
plt.title('Dataset')

# 可视化决策边界
plt.contour(x_grid, y_grid, p_proba, [0.5], linewidths=2., colors='blue')

plt.show()

#模型预测
