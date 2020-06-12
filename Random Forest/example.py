#  导入数据
from sklearn import datasets

# mnist = datasets.load_digits()
# X = mnist.data  # [1797, 64] 64个特征
# Y = mnist.target

#  数据处理
import pandas as pd

data = pd.read_csv('E:/machine learning/datasets/wine.csv')
x = data.drop('Wine', axis=1)
# x = x.drop(['Nonflavanoid.phenols', 'Proanth'], axis=1)
y = data['Wine']
#  数据归一化处理
for feature in list(x):
    max_ = x[feature].max()
    min_ = x[feature].min()
    x[feature] = x[feature].apply(lambda a: (a - min_) / (max_ - min_))

y = y.apply(lambda a: max(a, 0))

#  划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)

#  构建分类器
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, oob_score=True)
clf.fit(X_train, Y_train)

#  测试
y_pred = clf.predict(X_test)

#  计算准确率
from sklearn.metrics import accuracy_score

print("accuracy:%.6f" % accuracy_score(Y_test, y_pred))

#  估计特征的重要性
feature_importances = pd.Series(clf.feature_importances_).sort_values(ascending=False)

#  可视化特征的重要性
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x=feature_importances.index, y=feature_importances)
plt.xlabel("feature index")
plt.ylabel("featrue importance")
plt.title("visualizing feature importance")
plt.savefig("featrue importance.png")
# plt.legend()
# plt.show()

