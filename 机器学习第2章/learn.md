# 学习记录

## matplotlib

[绘制等高线图](https://blog.csdn.net/cymy001/article/details/78513712)

[ListedColormap产生一个颜色映射表](https://codingdict.com/sources/py/matplotlib.colors/11189.html)

[ListedColormap产生一个颜色映射表](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html)

## numpy

np.meshgrid函数，用来网格化数据用的，这样之后，就和matlab上面的meshgird一样了

np.ravel函数，[降维函数,返回引用](https://blog.csdn.net/lanchunhui/article/details/50354978)

[np.uniqe()去重函数](https://blog.csdn.net/u012193416/article/details/79672729)

```python
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                       np.arange(x2_min, x2_max, resolution))
z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

```

## 感知机的收敛性
感知机的规则就是

$$
z = x*\omega\\
\omega = \omega + \Delta\omega\\
\phi(z) =
\begin{cases}
1, \quad z>0\\
-1,\quad z<=0
\end{cases}
$$



这里通过感知机进行分类，之所以能够成功的原因是原本的数据本身就是线性可分的 ，如果遇到了一个线性不可分的话，那么最终会不停的迭代,不停的更新权重，最终也不会收敛，得到的感知机也不会是一个合格的机器