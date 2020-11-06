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
