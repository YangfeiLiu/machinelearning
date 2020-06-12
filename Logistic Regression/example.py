from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn
from sklearn import datasets
from scipy.stats import pearsonr

#  准备数据
# data = datasets.load_iris()
# x = data.data
# y = data.target

data = pd.read_csv('E:/machine learning/datasets/wine.csv')
x = data.drop('Wine', axis=1)
x = x.drop(['Nonflavanoid.phenols', 'Proanth'], axis=1)
y = data['Wine']
#  数据归一化处理
for feature in list(x):
    max_ = x[feature].max()
    min_ = x[feature].min()
    x[feature] = x[feature].apply(lambda a: (a - min_) / (max_ - min_))

y = y.apply(lambda a: max(a, 0))
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)

#  训练模型
model = LogisticRegression(solver='sag', tol=0.0001, C=0.1, penalty='l2', multi_class='auto')
model.fit(x_train, y_train)

#  测试模型
y_pred = model.predict(x_test)
y_preda = model.predict_proba(x_test)
print("accuracy:%.6f" % accuracy_score(y_test, y_pred))
plt.figure(figsize=(9, 9))
seaborn.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.title("accuracy %.6f" % accuracy_score(y_test, y_pred))
plt.savefig('./confusion_matrix.png')
# plt.show()
