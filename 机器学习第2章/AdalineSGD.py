#!/usr/bin/python3
# -*- coding:utf-8 -*-
'''
@File    :   AdalineSGD.py
@Time    :   2021/06/27 18:46:38
@Author  :   1024Person
@Desc    :   None
'''
# here put the import lib
import numpy as np
from perceptron import plot_decision_regions, get_data_from_file
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/usr/share/fonts/YaHeiConsolas.ttf', size=14)


class AdalineSGD(object):

    def __init__(self, eta=0.01, n_iter=10, shuffle=True,
                 random_state=0) -> None:
        """
        eta:学习率
        n_iter:迭代次数
        shuffle:是否刷新数据集
        random_state：？
        return : None
        """
        super().__init__()
        self.eta = eta
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        调用这个方法就说明要重新训练整个数据集，所以直接就初始化权重
        训练数据集，就好像是感知机的训练一样，一个一个的训练
        X：X.shape = (n_samples,n_features)
        y: vector y.shape = (n_samples,1)
        return : object self
        """
        self._initialize_weights(X.shape[1])
        self._cost = []     # 代价（平均偏差）
        for _ in range(self.n_iter):
            # 每次训练整个数据集的时候，刷新一下
            if self.shuffle:
                X, y = self._shuffel_data(X, y)
            cost = []
            for xi, target in zip(X, y):
                # 保存每一个预测和实例的偏差，
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(cost)
            self._cost.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """
        训练局部数据集合，就是当整体的数据集都训练完了，
        在这时又来了零零星星的几个数据，没有必要重新训练整个数据集了
        单独训练这几个数据就好了
        X : X.shape = (n_samples,n_features)
        y : vector or scalar  y.shape = (n_samples,1)
        return : object self
        """
        cost = []
        # 更新数据集
        if not self.shuffle:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            # 不止一个数据，但是用的着这么麻烦吗？
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self

    def predict(self, xi):
        """
        预测数据
        xi：xi.shape = (1,n_features)
        return : np.where(self.net_input(xi)>=0.0, 1, -1)
        """
        return np.where(self.net_input(xi) >= 0.0, 1, -1)

    def net_input(self, X):
        """
        计算净输入
        X: X.shape = (n_samples,n_features)
        w: w.shape = (n_feature,1)
        return : X.dot(self._w[1:]) + self._w[0]
        """
        return X.dot(self._w[1:]) + self._w[0]


    def activition(self, X):
        """
        激活了函数
        return : X
        """
        return X


    def _update_weights(self, xi, target):
        """
        更新权重,应用adaline的规则来更新权重
        xi:xi.shape = (1,n_feartures)
        target: target.shape = (n_samples,1)
        return : cost 代价
        """
        # 激活函数
        output = self.activition(self.net_input(xi))
        # 偏差
        error = target - output
        update = self.eta * xi*error
        self._w[1:] += update
        self._w[0] += self.eta * error

        cost = 1/2 * error**2
        return cost

    def _shuffel_data(self, X, y):
        """
        更新数据集
        return : X[r],y[r]
        """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """
        初始化权重
        return : None
        """
        self._regn = np.random.RandomState(self.random_state)
        # 为什么不给我提示了？
        self._w = self._regn.normal(loc=0.0, scale=0.01, size=m+1)
        self._initialize_weights = True


def main():
    """
    开始调用AdalineGDK的各种函数了
    """
    data = get_data_from_file("./iris.csv")
    X = data.iloc[0:100,[0,2]].values
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    y = data.iloc[:100,[4]].values
    y = np.where(y == "Iris-setosa",1,-1)
    y.shape = (100,)
    adaSGD = AdalineSGD(n_iter=16)
    adaSGD = adaSGD.fit(X_std,y) # 训练完成
    plot_decision_regions(X_std,y,adaSGD,fignum=1)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.figure(num=2)
    plt.plot(range(1,len(adaSGD._cost)+1),adaSGD._cost,
    marker='d',color='blue')

    plt.xlabel("迭代",fontproperties=font)
    plt.ylabel("SEE")
    plt.show()




if __name__ == '__main__':
    main()
