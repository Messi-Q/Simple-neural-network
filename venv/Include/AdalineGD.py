# -*_coding:utf8-*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


# 自适应神经元
class AdalineGD(object):
    """
    eta:float
    学习率 ：0~1
    n_iter:int
    对训练数据进行学习改进次数
    w_:一维向量
    存储权重数值
    error_：
    存储每次迭代改进时，网络对数据进行错误判断的次数
    """
    # 初始化
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    # 神经网络输入
    def net_input(self, X):
        """
        z=W0+W1*X1+...+Xn*Xn
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # 激活函数
    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    # 训练函数
    def fit(self, X, y):
        """
        X:二维数组[n_samples，n_features]
        n_samples:表示X中含有训练数据条目数
        n_features：含有4个数据的一维向量，用于表示一条训练条目
        y:一维向量
        用于存储每一条训练条目对应的正确分类
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            # 神经元参数的更新
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self


# 绘图函数
def plot_decision_regins(X, y, classifier, resolutions=0.02):
    marker = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()

    print(x1_min, x1_max)
    print(x2_min, x2_max)

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolutions), np.arange(x2_min, x2_max, resolutions))

    # 预测
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=marker[idx], label=cl)


file = open("F:/data.csv")
df = pd.read_csv(file, header=None)
print(df.head(10))

y = df.loc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:101, [0, 2]].values

ada = AdalineGD(eta=0.0001, n_iter=50)
ada.fit(X, y)
plot_decision_regins(X, y, classifier=ada)
plt.title('Adaline-Gradient descent')
plt.xlabel('Petal length')
plt.ylabel('Flower diameter length')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('sum-squared-error')
plt.show()
