> The English project introduction is below.

# 项目简介：IMDB电影评论情感分析
## 概述
本项目是一个电影评论情感分析（正面或负面）项目，使用IMDB 50K数据集构建了三个传统的机器学习模型，最终预测准确率达到90%。
## 我的主要工作
- 完成了数据清洗、文本预处理（文本的清洗、分词、停用词过滤、Stemming等）、模型训练、评估等完整的NLP处理流程
- 尝试用spaCy替代nltk来加速词干提取和移除停用词
- 对三个模型的评估结果进行了对比分析

## 亮点
- 使用并行计算将词干提取和移除停用词的时间缩短了一倍
- 通过优化TFIDF的参数将LR和MNB的准确率从75%提升到90%，将MNB从50%提升到了90%
## 使用说明
我的代码、数据集和运行结果放在kaggle上面，可以复制后直接运行notebook，地址： [Sentiment Analysis of IMDB Movie Reviews (90%）](https://www.kaggle.com/code/chenxucool/sentiment-analysis-of-imdb-movie-reviews-90)
## 技术栈
- 编程语言：Python 
- 框架和库：Scikit-learn, NLTK, TextBlob，TF-IDF，Bag of Words，joblib（并行计算）
- 算法模型：逻辑回归、朴素贝叶斯、支持向量机（SVM）
- 模型评估：准确率、精确率、召回率、F1值
- 数据集：imdb-dataset-of-50k-movie-reviews

## 模型效果

我在 IMDB 数据集上测试了三种模型的表现，结果如下：

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
