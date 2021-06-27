#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/6/26 下午2:42
# @Author  : 1024Person
# @File    : adaline.py

import numpy as np
import matplotlib.pyplot as plt
from perceptron import get_data_from_file
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname="/usr/share/fonts/YaHeiConsolas.ttf", size=14)


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


def main():
    data = get_data_from_file("./iris.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    X = data.iloc[:100, [1, 2]].values
    Y = data.iloc[:100, 4]
    Y = np.where(Y == "Iris-setosa", 1, -1)
    ada1 = Adaline(0.0001, 10).fit(X, Y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs', fontproperties=font)  # 迭代
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.0001')

    ada2 = Adaline(0.1).fit(X, Y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel('log(Sum-squared-error)')
    ax[1].set_title('Adaline - Learning rate 0.1')

    plt.show()


if __name__ == "__main__":
    main()
