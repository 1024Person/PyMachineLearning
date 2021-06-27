# /usr/bin/python
# @ FileName: Perceptron
# @ Author : 1024Person
# @ Description：机器学习中的感知机，对花的种类进行分类
# @ DateTime: 2021/06/26 14:29:46


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron:

    def __init__(self, eta=0.1, n_iter=500, seed=0):
        '''
        eta:学习lv
        n_iter:迭代次数，查找真正合适的权重
        seed:随机数种子
        '''
        self.eta = eta
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, Y):
        """

        :param X:  X{like-array} ,shape = [sample_n,features_n]
        :param Y:  Y{like-array},shape = [1+X.shape[0]]
        :return:  self
        """
        rgen = np.random.RandomState(self.seed)
        # 权重是对于每一个特征来说的
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=X.shape[1] + 1)
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                # 这个偏差值计算的是在当前这个特征上面的偏差，
                errors += int(update != 0.0)
                # 如果update == 0的话，就说明这次预测非常的完美，没有偏差
                # 如果不等于0的话，就是真，预测失败，errors加上1，这个errors的值，就是预测失败的值
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        '''
        计算净输入z的值
        :param X:
        :return:
        '''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        预测值
        :param X:
        :return:
        """
        # 返回1或者-1
        return np.where(self.net_input(X) > 0.0, 1, -1)


def get_data_from_file(filepath):
    """
    从文件中读取数据
    :param filepath:
    :return: dataFrame
    """
    df = pd.read_csv(filepath, header=None)
    print('最后五条数据：')
    print(df.tail())
    return df


def plot_decision_regions(X, y, classifier, resolution=0.02,fignum = 1):
    marker = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 产生一个颜色映射

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # 下面将xx1,xx2降维成一维数组，然后再进行转置
    # 但是为什么要对这些值进行预测呢？
    # 为了下面画图的需要吗？
    # 对就是这样是对二维空间中每一个坐标进行预测，判断出这个坐标应该属于哪一类，
    # 然后这个坐标平面就会被分成两部分，
    # 然后再对这两部分绘制不同的颜色
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 因为上面的z的形状是进行了一些调整，所以这里还要在调回来
    # xx1，是网格化之后的数据，所以xx1的形状就是整个网格的形状
    z = z.reshape(xx1.shape)
    plt.figure(num=fignum)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        # 这里的y == cl对应的会产生一个逻辑数组，然后通过这个逻辑数组进行选取数组X中的值
        # 数组下标中逻辑值为真的就会选取，逻辑值为假的就不会选取
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=marker[idx],
                    label=cl,
                    edgecolor='black')


def main():
    per = Perceptron(eta=1, n_iter=10)
    data = get_data_from_file("iris.csv")
    y = data.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", 1, -1)
    X = data.iloc[0:100, [1, 2]].values
    plt.figure(1, figsize=(10, 30))
    plt.subplot(221)
    plt.scatter(X[:50, 0], X[:50, 1], color="blue",
                marker="x", label="setosa")
    plt.scatter(X[50:, 0], X[50:, 1], color="red",
                marker='o', label='versicolor')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc="upper left")
    plt.subplot(222)
    per.fit(X, y)
    plt.plot(range(1, len(per.errors_) + 1), per.errors_, marker='o')
    plt.xlabel('Epochs')  # 迭代次数
    plt.ylabel("Number of updates")  # 偏差
    plt.subplot(2, 2, (3, 4))
    plot_decision_regions(X, y, classifier=per)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
