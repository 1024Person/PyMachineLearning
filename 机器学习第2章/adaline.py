#!/usr/bin/python3
# -*- coding:utf-8 -*-
'''
@File    :   adaline.py
@Time    :   2021/06/27 15:34:17
@Author  :   1024Person
@Version :   1.0
@Contact :   822713663@qq.com
@Desc    :   None
@License :
'''

import numpy as np
import matplotlib.pyplot as plt
from perceptron import get_data_from_file,plot_decision_regions
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname="/usr/share/fonts/YaHeiConsolas.ttf", size=10)


class Adaline:
    def __init__(self, eta, n_iter=10, seed=10):
        """

        :param eta: 学习率
        :param n_iter: 迭代次数
        :param seed: 随机种子
        """
        self.eta = eta
        self.n_iter = n_iter
        self.seed = seed

    # 训练
    def fit(self, X, Y):
        # 种下随机种子
        np.random.RandomState(self.seed)
        self.w_ = np.random.normal(loc=0.0, scale=0.1, size=X.shape[1] + 1)

        self.cost_ = []
        for _ in range(self.n_iter):
            z = self.net_input(X)
            output = self.activation(z)
            # errors是预测值和真实值之间的偏差，
            errors = (Y - output)
            update = self.eta * X.T.dot(errors)
            self.w_[1:] += update
            self.w_[0] += errors.sum() * self.eta
            # cost和errors的平方成正比，所以这里cost越小就代表着误差越小
            cost = (errors ** 2).sum() / 2
            self.cost_.append(cost)
        return self

    # 激活函数
    def activation(self, X):
        return X

    # 净输入
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0 , 1,-1)


def main():
    data = get_data_from_file("./iris.csv")
    plt.figure(num=1)
    X = data.iloc[:100, [1, 2]].values
    Y = data.iloc[:100, 4]
    Y = np.where(Y == "Iris-setosa", 1, -1)
    # 进行一次数据预处理，将数据标准化
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std();

    # 通过将数据标准化之后，确实再很大的学习效率的时候，就能找到全局最小值了

    ada1 = Adaline(0.0001, 10).fit(X_std, Y)
    ada2 = Adaline(0.01).fit(X_std, Y)
    # 注意这次是自适应神经元，所以这次产生的是一个连续的实数，而不再只是两个标签了！！
    # 所以这次画出来的等高线有很多条分界线
    plot_decision_regions(X_std,Y,ada1,fignum = 1)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs', fontproperties=font)  # 迭代
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.0001')

    ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel('log(Sum-squared-error)')
    ax[1].set_title('Adaline - Learning rate 0.01')



    plt.show()


if __name__ == "__main__":
    main()
