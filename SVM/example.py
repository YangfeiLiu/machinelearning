import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn

#  导入数据
df = pd.read_csv('E:/machine learning/datasets/Boston.csv')
# print(df.describe())
#  数据共8个特征，最后一列是标签
x = df.drop('chas', axis=1)
# x = df[['Glucose', 'Age']]
#  归一化数据
for feature in list(x):
    max_ = x[feature].max()
    min_ = x[feature].min()
    x[feature] = x[feature].apply(lambda a: (a - min_) / (max_ - min_))
y = df['chas']
#  划分数据
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)
#  定义模型
model = svm.SVC(C=1.0, kernel='poly', gamma='auto')
model.fit(x_train, y_train)
#  测试模型
y_pred = model.predict(x_test)
print("accuracy:%.6f" % accuracy_score(y_test, y_pred))
plt.figure(figsize=(9, 9))
seaborn.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.title("accuracy %.6f" % accuracy_score(y_test, y_pred))
plt.savefig('./confusion_matrix.png')
plt.show()
