# LectureNotes_for_Recommendation_System
Lecture Notes for Recommendation System Algorithms

<p align="center">
    <a href="https://github.com/elegantcoin/LectureNotes_for_Recommendation_System"><img src="https://img.shields.io/badge/status-updating-brightgreen.svg"></a>
    <a href="https://github.com/python/cpython"><img src="https://img.shields.io/badge/Python-3.7-FF1493.svg"></a>
    <a href="https://github.com/elegantcoin/LectureNotes_for_Recommendation_System"><img src="https://img.shields.io/badge/platform-Windows%7CLinux%7CmacOS-660066.svg"></a>
    <a href="https://opensource.org/licenses/mit-license.php"><img src="https://badges.frapsoft.com/os/mit/mit.svg"></a>
    <a href="https://github.com/elegantcoin/LectureNotes_for_Recommendation_System/stargazers"><img src="https://img.shields.io/github/stars/elegantcoin/LectureNotes_for_Recommendation_System.svg?logo=github"></a>
    <a href="https://github.com/elegantcoin/LectureNotes_for_Recommendation_System/network/members"><img src="https://img.shields.io/github/forks/elegantcoin/LectureNotes_for_Recommendation_System.svg?color=blue&logo=github"></a>
    <a href="https://www.python.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" align="right" height="48" width="48" ></a>
</p>
<br />


## :fire: 1.常用到的库、模型、数据集等
序号| 库、模型/数据集 | 链接
------|------|------
1.|Scikit-Learn(Sklearn) |https://github.com/scikit-learn/scikit-learn (https://www.cntofu.com/book/170/index.html)
2.|Spark MLlib |https://spark.apache.org/docs/latest/ml-guide.html
3.|TensorFlow |https://github.com/tensorflow/tensorflow （https://www.tensorflow.org/api_docs/）
4.|libffm |https://github.com/ycjuan/libffm
5.|fastFM |https://github.com/ibayer/fastFM
6.|gcforest深度森林 |https://github.com/kingfengji/gcForest
7.|xLearn|https://github.com/aksnzhy/xLearn （https://xlearn-doc.readthedocs.io/en/latest/）
8.|XGBoost|https://github.com/dmlc/xgboost （https://xgboost.readthedocs.io/en/latest/）
9.|Million Song Dataset（MSD数据集） |http://millionsongdataset.com/
10.|Yahoo Movies公开?乐数据集|https://webscope.sandbox.yahoo.com
11.|Outbrain 点击率数据|https://www.outbrain.com
12.|Amazon product data亚马逊|http://jmcauley.ucsd.edu/data/amazon/index.html

## :fire: 2.推荐系统的主要流程
![https://zhuanlan.zhihu.com/p/259985388](https://pic1.zhimg.com/80/v2-8670b6282301ee6ce727e54f0d8c78c0_720w.jpg)

- 推荐系统重要的3个模块
    - 召回模块：物品太多（数以亿计），需要挑选（~100），需要快速查询（不超过**100毫秒**）
    - 排序模块：对后选集精排，特征交叉，评分，更加精确的预测，TOPk
    - 后排模块：最终给到用户前的调整，运行干预、优先级调整、下发规则等
- 推荐系统的主要元素
    - 物品集合：
    - 用户：基本信息、行为、兴趣爱好
    - 场景：网络环境、时间等
    - 推荐引擎：学习用户喜欢什么物品的模型
    - 推荐结果集：一般是排序的集合

## :fire: 3.推荐系统的主要算法
- 召回模型算法
    - 内容相似：word2vec、LDA、FastText、LSF-SCNN、LSTM
    - 行为相似：ItemCF、UserCF、关联规则
    - 深度学习：DNN
- 排序模型算法
    - 线性：LR、FM、特征交叉+LR/FM
    - 非线性：DNN、Wide&Deep（Google play）、FNN、PNN、DeepFM（华为诺亚??）、NFM、AFM、DCN、DIN（阿里）
    - 树模型：决策树、随机森林、GBDT、XGBoost、GBDT+LR
    - 集成学习：数类型集成、GCForest

## :fire: 4.常见优化方法
- 梯度法（一阶导）：
    - SGD：适应数量大的数据集，容易陷入局部最优 θ = θ -εg
    - 动量：方向加速度，抑制震荡，  v = αv-εg，θ = θ + v
    - Nesterov动量：提高灵敏度，先临时更新θ，再更新速度，更新θ。
    - AdaGrad：计算累积平方梯度，适合处理稀疏梯度。
    

- 牛顿法（二阶导，海森矩阵）：
    - Adam：利用有偏一阶矩、二阶矩，参数平稳，使用大数据集和高维、非凸优化
    - L-BFGS：计算Hessian矩阵的逆，收敛速度快，但是消耗内存。对初始值有要求，容易陷入鞍点。


TODO List：
## :fire: 5.推荐系统历史进展梳理
![https://www.zhihu.com/question/20830906/answer/681688041](https://pic4.zhimg.com/80/v2-763b523bd17349cd6cfecae2765db3d5_720w.jpg?source=1940ef5c)
## :fire: 6.协同过滤算法
## :fire: 7.Word2vec算法
## :fire: 8.线性的LR模型
## :fire: 9.线性的FM模型
## :fire: 10.树的决策树
## :fire: 11.树的集成学习
## :fire: 12.深度学习的DNN算法
## :fire: 13.深度学习的Wide&Deep模型
## :fire: 14.深度学习的DeepFM模型



## :fire: Appendex
- A1.pandas 操作
    - explode 操作 一行变多行 stack()
    ```python
    # 测试一行变多行
    import pandas as pd
    import numpy as np
    from ast import literal_eval
    df = pd.DataFrame({'key1': ['K0', 'K0', np.nan, 'K1'],
                    'A': ["[{'id': 3, 'name': 'Fa'}, {'id': 4, 'name': 'Th'}]", "[{'id': 1, 'name': 'An'}, {'id': 2, 'name': 'Co'}]", "[{'id': 1, 'name': 'An'}, {'id': 2, 'name': 'Co'}]", "[{'id': 1, 'name': 'An'}, {'id': 2, 'name': 'Co'}]"]})
    print(df,"\n")
    df['A'] = df['A'].apply(literal_eval)
    df['A'] = df['A'].apply(lambda x: [i['name'].lower() for i in x] if isinstance(x, list) else [])
    s = df.apply(lambda x: pd.Series(x['A']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'B'
    B_df = df.drop('A', axis=1).join(s)
    B_df
    ```
    结果
    ```python
      key1                                                  A
    0   K0  [{'id': 3, 'name': 'Fa'}, {'id': 4, 'name': 'T...
    1   K0  [{'id': 1, 'name': 'An'}, {'id': 2, 'name': 'C...
    2  NaN  [{'id': 1, 'name': 'An'}, {'id': 2, 'name': 'C...
    3   K1  [{'id': 1, 'name': 'An'}, {'id': 2, 'name': 'C...

    key1	B
    0	K0	fa
    0	K0	th
    1	K0	an
    1	K0	co
    2	NaN	an
    2	NaN	co
    3	K1	an
    3	K1	co
    ```
    
