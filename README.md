# IMDB电影评论情感分析
## 概述
这是一个电影评论情感分析（正面或负面）项目，使用IMDB 50K数据集构建了三个传统的机器学习模型，通过优化，最终预测准确率达到90%。

## kaggle地址
[Sentiment Analysis of IMDB Movie Reviews (90%）](https://www.kaggle.com/code/chenxucool/sentiment-analysis-of-imdb-movie-reviews-90)

## 技术博客地址
[我完成第一个AI项目的全过程记录——对IMDB电影评论的情感分析](https://blog.csdn.net/weixin_43221845/article/details/141826318?csdn_share_tail=%7B%22type%22:%22blog%22,%22rType%22:%22article%22,%22rId%22:%22141826318%22,%22source%22:%22weixin_43221845%22%7D)

## 主要工作
- 完成了数据清洗、文本预处理（文本的清洗、分词、停用词过滤、Stemming 等）、模型训练、评估等完整的NLP处理流程
- 尝试用 spaCy 替代 nltk 来加速词干提取和移除停用词
- 对三个模型的评估结果进行了对比分析

## 亮点
- 使用并行计算将词干提取和移除停用词的时间缩短了一倍
- 通过优化 TF-IDF 的参数将逻辑回归模型和朴素贝叶斯模型的准确率从75%提升到90%，将LSVM模型从50%提升到了90%

## 技术栈
- 框架和库：Scikit-learn, NLTK, TF-IDF，Bag of Words，joblib（并行计算）
- 算法模型：逻辑回归、朴素贝叶斯、支持向量机（SVM）
- 模型评估：准确率、精确率、召回率、F1值、混淆矩阵
- 数据集：imdb-dataset-of-50k-movie-reviews

## 模型表现

| 模型        | 准确率  |
|-------------|---------|
| LR        | 90%     |
| LSVM | 89.5%     |
| MNB | 89%     |

## 未来工作
- 增加多语言支持，扩展到其他语言的评论情感分析
- 尝试更复杂的深度学习模型（如BERT、GPT等）
- 模型集成：结合多个分类器提升性能
- 超参数调优：通过网格搜索找到最佳模型参数
- 实现Web接口，提供在线情感分析服务
