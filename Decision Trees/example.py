import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import numpy as np

#  导入数据
iris = datasets.load_iris()
features = iris.data  # four features: sepal length, sepal width, petal length, and petal width
target = iris.target  # three classes: Iris Setosa, Iris Versicolour, and Iris Virginica

data = list(zip(features, target))
np.random.shuffle(data)
features,  target = list(zip(*data))

#  划分训练和测试
train_features = features[:120]
train_target = target[:120]

test_features = features[120:]
test_target = target[120:]

# print(features)
# print(target)

#  创建决策树分类器
decisiontree = DecisionTreeClassifier(criterion="entropy", splitter="random", random_state=0)

#  训练模型
model = decisiontree.fit(train_features, train_target)

#  测试模型
correct = 0
for test in list(zip(test_features, test_target)):
    observation = np.expand_dims(test[0], axis=0)  # Predict observation's class
    gt = test[1]
    pre_class = model.predict(observation)[0]
    pre_proba = model.predict_proba(observation)
    if gt == pre_class:
        correct += 1
print("accuracy:%.4f" % (correct / 30))
# print(pre_class)
# print(pre_proba)

#  导出模型
from sklearn import tree
dot_data = tree.export_graphviz(decisiontree, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names)

#  可视化决策树
graph = pydotplus.graph_from_dot_data(dot_data)  # Show graph
graph.write_png('tree.png')
