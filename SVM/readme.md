##### 使用sklearn实现支持向量机

###### 关于支持向量机介绍看[这里]([https://yangfeiliu.github.io/2020/05/25/SVM%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%E7%9A%84%E7%90%86%E8%A7%A3%E4%B8%8E%E6%8E%A8%E5%AF%BC/](https://yangfeiliu.github.io/2020/05/25/SVM支持向量机的理解与推导/))

###### 实现部分：

1. 数据，发现对于有的数据，归一化后结果有变化，有的数据没有变化

2. 使用sklearn.svm里的svc类，主要包括以下参数：

   - **Kernel:** 核函数，默认是“rbf”，可选择的还有"linear", "poly", "sigmoid", "precomputed"。
     - "linear"是最常用的核之一，当数据线性可分时选择
     - "rbf" 径向基函数，当数据不是线性可分时选择

   - **Regularization C:** 正则化参数，与C呈反比，C必须为正。
   - **Degree:**  多项式核的阶数
   - **Verbose:**  详细的日志输出，默认False
   - **Random_State:** 随机数的种子，可以确保结果复现