# -*_coding:utf8-*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


# 感知器算法
class Perceptron(object):
    """
    eta:学习率
    n_iter:权重向量的训练次数
    W：神经分叉权重向量
    errors：用于记录神经元判断出错次数
    """
    # 初始化
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    # 神经网络输入
    def net_input(self, X):
        """
        z=W0+W1*X1+...+Xn*Xn
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    # 训练函数
    def fit(self, X, y):
        """
        输入训练数据，培训神经元
        X:输入样本向量
        y:对应样本分类
        X：shape[n_samples，n_features]
        X:[[1,2,3],[4,5,6]]
        y:[1.-1]
        n_samples:2
        n_features:3
        初试话权重向量为0
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            """
           X:[[1,2,3],[4,5,6]]
           y:[1,-1]
           zip(X,y)=[[1,2,3,1],[4,5,6,-1]] 
           """
            for xi, target in zip(X, y):
                """
                update=η*(y-y')
              """
                update = self.eta * (target - self.predict(xi))
                """
                xi是一个向量
                update * xi 等价：[ΔW(1)=X[1]*update,ΔW(2)=X[2]*update,ΔW(3)=X[3]*update]
              """
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)


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


file = open("F:/=data.csv")
df = pd.read_csv(file, header=None)
print(df.head(10))

y = df.loc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:101, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel("Petal length")
plt.ylabel("Flower diameter length")
plt.legend(loc="upper left")
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('sum-squared-error')
plt.show()

plot_decision_regins(X, y, ppn, resolutions=0.02)
plt.xlabel("Petal length")
plt.ylabel("Flower diameter length")
plt.legend(loc="upper left")
plt.show()
