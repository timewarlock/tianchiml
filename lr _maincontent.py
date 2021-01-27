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
#逻辑回归 原理简介：

#Logistic回归虽然名字里带“回归”，但是它实际上是一种分类方法，主要用于两分类问题（即输出只有两种，分别代表两个类别），
#所以利用了Logistic函数（或称为Sigmoid函数），函数形式为：
#𝑙𝑜𝑔𝑖(𝑧)=1/(1+𝑒−𝑧)

#其对应的函数图像可以表示如下:

import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-5,5,0.01)
y = 1/(1+np.exp(-x))

plt.plot(x,y)
plt.xlabel('z')
plt.ylabel('y')
plt.grid()
plt.show()