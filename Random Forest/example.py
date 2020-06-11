#  导入数据
from sklearn import datasets

mnist = datasets.load_digits()
X = mnist.data  # [1797, 64] 64个特征
Y = mnist.target

#  数据处理
import pandas as pd

# Y = pd.Series(Y).astype('int').astype('category')
# X = pd.DataFrame(X)

# print(X.head())
# print(Y.head())

#  划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

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
plt.legend()
plt.show()
