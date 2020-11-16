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


## :fire: 1.���õ��Ŀ⡢ģ�͡����ݼ���
���| �⡢ģ��/���ݼ� | ����
------|------|------
1.|Scikit-Learn(Sklearn) |https://github.com/scikit-learn/scikit-learn (https://www.cntofu.com/book/170/index.html)
2.|Spark MLlib |https://spark.apache.org/docs/latest/ml-guide.html
3.|TensorFlow |https://github.com/tensorflow/tensorflow ��https://www.tensorflow.org/api_docs/��
4.|libffm |https://github.com/ycjuan/libffm
5.|fastFM |https://github.com/ibayer/fastFM
6.|gcforest���ɭ�� |https://github.com/kingfengji/gcForest
7.|xLearn|https://github.com/aksnzhy/xLearn ��https://xlearn-doc.readthedocs.io/en/latest/��
8.|XGBoost|https://github.com/dmlc/xgboost ��https://xgboost.readthedocs.io/en/latest/��
9.|Million Song Dataset��MSD���ݼ��� |http://millionsongdataset.com/
10.|Yahoo Movies�����������ݼ�|https://webscope.sandbox.yahoo.com
11.|Outbrain ���������|https://www.outbrain.com
12.|Amazon product data����ѷ|http://jmcauley.ucsd.edu/data/amazon/index.html
13.|github daicoolb������Ƽ����ݼ�|https://github.com/daicoolb/RecommenderSystem-DataSet

## :fire: 2.�Ƽ�ϵͳ����Ҫ����
___
![https://zhuanlan.zhihu.com/p/259985388](https://pic1.zhimg.com/80/v2-8670b6282301ee6ce727e54f0d8c78c0_720w.jpg)
___

![](files/recom_sys.jpg)
- �Ƽ�ϵͳ��Ҫ��3��ģ��
    - �ٻ�ģ�飺��Ʒ̫�ࣨ�����ڼƣ�����Ҫ��ѡ��~100����������Ҫ���ٲ�ѯ��������**100����**��
    - ����ģ�飺�Ժ�ѡ�����ţ��������棬���֣����Ӿ�ȷ��Ԥ�⣬TOPk
    - ����ģ�飺���ո����û�ǰ�ĵ��������и�Ԥ�����ȼ��������·����򡢶����ԡ�ʵʱ�ԡ����жȡ����ʶȡ���������

    - ʵʱ���ݴ�����ƽ̨��������ƽ̨������ѵ�������߸��¡�����������A/B����
- �Ƽ�ϵͳ����ҪԪ��
    - ��Ʒ���ϣ�
    - �û���������Ϣ����Ϊ����Ȥ����
    - ���������绷����ʱ���
    - �Ƽ����棺ѧϰ�û�ϲ��ʲô��Ʒ��ģ��
    - �Ƽ��������һ��������ļ���

## :fire: 3.�Ƽ�ϵͳ����Ҫ�㷨
![](files/trand_evolution.jpg)
- �ٻ�ģ���㷨
    - �������ƣ�word2vec��LDA��FastText��LSF-SCNN��LSTM
    - ��Ϊ���ƣ�ItemCF��UserCF����������
    - ���ѧϰ��DNN
- ����ģ���㷨
    - ���ԣ�LR��FM����������+LR/FM
    - �����ԣ�DNN��Wide&Deep��Google play����FNN��PNN��DeepFM����Ϊŵ�Ƿ��ۣ���NFM��AFM��DCN��DIN�����
    - ��ģ�ͣ������������ɭ�֡�GBDT��XGBoost��GBDT+LR
    - ����ѧϰ�������ͼ��ɡ�GCForest

## :fire: 4.�����Ż�����
- �ݶȷ���һ�׵�����
    - SGD����Ӧ����������ݼ�����������ֲ����� �� = �� -��g
    - ������������ٶȣ������𵴣�  v = ��v-��g���� = �� + v
    - Nesterov��������������ȣ�����ʱ���¦ȣ��ٸ����ٶȣ����¦ȡ�
    - AdaGrad�������ۻ�ƽ���ݶȣ��ʺϴ���ϡ���ݶȡ�
    - Adam��������ƫһ�׾ء����׾أ�����ƽ�ȣ�ʹ�ô����ݼ��͸�ά����͹�Ż�
    
- ţ�ٷ������׵�����ɭ���󣩣�
    - L-BFGS������Hessian������棬�����ٶȿ죬���������ڴ档�Գ�ʼֵ��Ҫ���������배�㡣
- ��ţ�ٷ���




TODO List��
## :fire: 5.�Ƽ�ϵͳ��ʷ��չ����
- **��ʷ**
    - 1992��`ʩ�ֹ�˾``David Goldberg`�������ʼ�Эͬ����ϵͳ
    - 1997��`Resnick`��� ���Ƽ�ϵͳ��һ��
    - 1998��`����ѷ` ����Эͬ����
    - 2001��`IBM` Websphere���Ӹ��Ի��Ƽ�
    - 2003��`Google` ������AdWordsģʽ��2007����Ӹ��Ի�Ԫ��
    - 2006��`Netflix` ��Ӱ�Ƽ��㷨������`����ֽ�`����
    - 2007��`�Ż�` SmartAds ��淽��
    - 2007��`ACM` ��һ���Ƽ�ϵͳ���
    - 2010, `Steffen Rendle`��CTRԤ�⾺�������`FM`ģ��
    - 2015��`Facebook` �������Ƽ�ϵͳ`GBDT+LR`ԭ��
    - 2016��`Youtube` ��������Ƶ�Ƽ�ϵͳ
    - 2016��`Google` ����App�̵��Ƽ�ϵͳ`Wide & Deep`��˫�����ѧϰ��
    - 2017��`��Ϊ` ŵ�Ƿ����Ŷ���IJCAI���Ƴ�`DeepFM`
    - 2017��`����Ͱ�` �Ƴ�`DIN`������`DIEN`��`MIMN`��`ESSM`ģ��
    - 2017��`����` �Ƴ����ǿ��ѧϰģ��,v3(2019)
```
(��Ҫ3���ڵ㣬`2010`��ǰЭͬ���ˡ��߼��ع飻->>> `2010-2015`�����ӷֽ�����ݶ���������->>>`2015`������ѧϰ��һ����ϵļܹ�ģ��)
```
Bandit��BPR��CMN��DIEN��DKN��DMF��DSIN��Evaluation-metrics��FTRL��IRGAN��MKR��[MLR](https://arxiv.org/pdf/1704.05194.pdf)��NAIS��NCF��RippleNet��SRGNN��XDeepFM��[GBDT+LR](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)��LR��FM��FFM��FNN��[PNN](https://arxiv.org/pdf/1611.00144)��IPNN��OPNN��PNN*����[NFM](https://arxiv.org/abs/1708.05027)��[AFM](https://www.comp.nus.edu.sg/~xiangnan/papers/ijcai17-afm.pdf)��[Wide �� Deep](https://dl.acm.org/citation.cfm?id=2988454)��[DeepFM](https://arxiv.org/abs/1703.04247)��[DCN](https://arxiv.org/pdf/1708.05123)��[DIN](https://arxiv.org/abs/1706.06978)��[LinUCB](https://arxiv.org/pdf/1003.0146.pdf)��[ESSM](https://arxiv.org/abs/1804.07931)
___
![https://www.zhihu.com/question/20830906/answer/681688041](files/deep_learning_recom.jpg)
___



## :fire: 6.Эͬ�����㷨
- ��ȱ�㣺
    - Эͬ����ֱ�ۡ��ɽ�����ǿ
    - ���߱���ǿ�ķ�������������ϡ��������������
    - ����ֽ������Эͬ���ˣ��и�ǿ�ķ�����������ΪӰ���������ɹ���ʵ����ȫ����ϵĹ��̣�ӵ��ȫ����Ϣ��

- ���ƶȼ��㣺
    - ͬ�����ƶ�
    - ŷ����þ���
    - Ƥ��ѷ���ϵ��
        - Ƥ��ѷ���ϵ��ͨ��ʹ���û�ƽ���ֶԸ��������ֽ�����������С���û�����ƫ�õ�Ӱ�졣
    - Cosine���ƶ�
    - Tanimotoϵ��
## :fire: 7.Word2vec�㷨
- Embedding

- TODO

## :fire: 8.���Ե�LRģ��
- ��ȱ�㣺
    - �����û���Ϊ�ȶ����������������ںϣ�����ȫ��
    - ��ѧ�ϵ�֧�ţ�������������Ӳ�Ŭ���ֲ�
    - ������ǿ��������Ȩ�ͺ�����ͬ������Ӱ�죬Ȩ�ش���������Ҫ�̶ȣ�sigmoidӳ�䵽0~1��Χ
    - ���̻���Ҫ�����ڲ��С�ѵ������С
    - ���ǣ��޷������������桢����ɸѡ�������Ϣ��ʧ��`����ɭ���`


## :fire: 9.���Ե�FM/FFMģ��
- ���ӷֽ����Factorization Machine,FM��
    - ��ȱ�㣺
        - ѵ���������С������������ߣ������ƶϼ򵥣�
        - 

    - `one-hot����`������ÿ��ά��ռ��һ��λ����������ĳ��ά�ȣ���ӦλΪ1������λΪ0.������������ѹ���ռ䣬��123...����ᵼ��ͬ���������ƶȲ�һ�£�
    - �������ڻ���ȡ����`POLY2`�ı���������������Ȩ��ϵ����
    - ������ֽ�ĵ����û�����Ʒ��������չ�������������ϡ�
- FFM��Field-aware Factorization Machine��
    - ��ȱ�㣺
        - ��������������������м�ֵ��Ϣ��ģ�ͱ��������ǿ
        - ���Ǽ��㸴�Ӷ���������Ҫ��ģ��Ч���͹���Ͷ��֮��Ȩ�⡣        
    - ÿ��������ʱ��ӦΨһ��������������һ����������
- ��������Ƽ�ϵͳ�У�CTRԤ��(click-through rate)
- �ڽ���CTRԤ��ʱ�����˵������⣬����Ҫ������������ϡ�
- ����ģ��û�п���������Ĺ��������ö���ʽģ�ͱ���������������(��������)
- 


## :fire: 10.���ľ�����
## :fire: 11.���ļ���ѧϰ
- ��ȱ��
    - `FMM`ֻ�������׽��棬`GBDT+LR`ʵ�ָ��߽׵Ľ��棬�����˹��������̣��ܹ��˵���ѵ����
    - LR��������ģ�ͣ��ŵ��ǲ��л����������ɴ�������������
    - ȱ����ѧϰ�������ޣ���Ҫ��������������������ģ�͵�ѧϰ������
    - `GBDT`��ȱ�������ײ�������ϡ���̫�߽׵Ľ��浼�£�������ת����ʧ�˴�����������ֵ��Ϣ��
- ��������߽׽���
    - ÿ�����Ľڵ����һ�ξ���һ�ν��棬������������Ⱦ�����
- Ϊ��Ҫ����
    - FMģ��ͨ���������ķ�ʽ��������������֮�����Ϲ�ϵ��(���߲�ε�������Ϲ�ϵ��Ҫ���������)��
    - ΪʲôҪʹ�ü��ɵľ�����ģ�ͣ������ǵ��õľ�����ģ�ͣ�һ�����ı�����������������Ա�����������Ե��������
- Ϊʲô��������GBDT����RF��
    -  RFҲ�Ƕ����������Ч������ʵ��֤������GBDT����GBDTǰ�����������������Ҫ���ֶԶ������������ֶȵ������������������Ҫ���ֵ��Ǿ���ǰN�������в���Ȼ�ϴ������������
    - ����ѡ���������������ֶȵ���������ѡ������������������ֶȵ�������˼·���Ӻ���

- ʵ����
    - ���ڵ�GBDT��LR���ںϷ�������ʺ����ڵĴ����ҵ������ô�����ڵ�ҵ�������Ǵ�����ɢ�������µĸ�ά����ɢ���ݡ�����ģ�Ͷ���������ɢ�������ǲ��ܺܺô���ģ����׵��¹���ϡ�
    - GBDTû���ƹ��ԣ�����˵����������

## :fire: 12.���ѧϰ��DNN�㷨
## :fire: 13.���ѧϰ��Wide&Deepģ��
## :fire: 14.���ѧϰ��DeepFMģ��

## :fire: 15.��ҵ�Ƽ�ϵͳʹ��



## :fire: Appendex
- A1.pandas ����
    - explode ���� һ�б���� stack()
    ```python
    # ����һ�б����
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
    ���
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
    
- A2.�����Ĳο���ϢԴ
    - [����ʦ](https://www.zhihu.com/people/nphard-79)
    - [����](https://www.zhihu.com/people/wang-zhe-58)
    - [billlee](https://www.zhihu.com/people/billlee-83)
    - [ʯ����](https://www.zhihu.com/people/shi-xiao-wen-19-51)
    - [Ģ������ѧϰ��](http://xtf615.com/2018/05/03/recommender-system-survey/)
    - [mJackie](https://github.com/mJackie/RecSys)
    - [princewen](https://github.com/princewen/tensorflow_practice)
    - [��һ��](https://www.zhihu.com/people/yisong)
    - [�ֽ������Ƽ�ϵͳ](https://www.volcengine.cn/docs/4462/37486)