> The English project introduction is below.

# 项目简介：IMDB电影评论情感分析
## 概述
本项目是一个电影评论情感分析（正面或负面）项目，使用IMDB 50K数据集构建了三个传统的机器学习模型，最终预测准确率达到90%。
## 我的主要工作和收获
- 完成了数据清洗、文本预处理（文本的清洗、分词、停用词过滤、Stemming等）、模型训练、评估等完整的NLP处理流程
- 尝试用spaCy替代nltk来加速词干提取和移除停用词
- 对三个模型的评估结果进行了对比分析
- 熟悉了kaggle、各个框架、模型的使用和原理
## 亮点
- 使用并行计算将词干提取和移除停用词的时间缩短了一倍
- 通过优化TFIDF的参数将LR和MNB的准确率从75%提升到90%，将MNB从50%提升到了90%
## 使用说明
我的代码、数据集和运行结果放在kaggle上面，可以复制后直接运行notebook，地址： [Sentiment Analysis of IMDB Movie Reviews (90%）](https://www.kaggle.com/code/chenxucool/sentiment-analysis-of-imdb-movie-reviews-90)
## 技术栈
- 编程语言：Python 
- 框架和库：Scikit-learn, NLTK, TextBlob，BeautifulSoup，WordCloud，joblib
- 模型：LR（逻辑回归）、LSVM（线性支持向量机）和MNB（多项式朴素贝叶斯）
- 数据集：imdb-dataset-of-50k-movie-reviews

## 模型效果

我们在 IMDB 数据集上测试了两种模型的表现，结果如下：

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


---

# Project Introduction: IMDB Movie Review Sentiment Analysis

## Overview
This project focuses on sentiment analysis (positive or negative) of movie reviews, utilizing the IMDB 50K dataset to build three traditional machine learning models, achieving a final prediction accuracy of 90%.

## My Main Contributions and Achievements
- Completed the entire NLP pipeline, including data cleaning, text preprocessing (cleaning, tokenization, stopword removal, stemming, etc.), model training, and evaluation.
- Experimented with replacing NLTK with spaCy to accelerate stemming and stopword removal.
- Conducted comparative analysis of the evaluation results of the three models.
- Gained familiarity with Kaggle, various frameworks, models, and their underlying principles.

## Highlights
- Reduced the time for stemming and stopword removal by half using parallel computing.
- Optimized the parameters of TFIDF to improve the accuracy of LR and MNB from 75% to 90%, and MNB from 50% to 90%.

## Usage Instructions
My code, dataset, and results are available on Kaggle, where you can copy and directly run the notebook. Link: [Sentiment Analysis of IMDB Movie Reviews (90%)](https://www.kaggle.com/code/chenxucool/sentiment-analysis-of-imdb-movie-reviews-90).

## Tech Stack
- **Programming Language**: Python
- **Frameworks & Libraries**: Scikit-learn, NLTK, TextBlob, BeautifulSoup, WordCloud, joblib
- **Models**: LR (Logistic Regression), LSVM (Linear Support Vector Machine), and MNB (Multinomial Naive Bayes)
- **Dataset**: imdb-dataset-of-50k-movie-reviews

## Model Performance
We tested the performance of three models on the IMDB dataset, with the results as follows:

| Model   | Accuracy  |
|---------|-----------|
| LR      | 90%       |
| LSVM    | 89.5%     |
| MNB     | 89%       |

## Future Work
- Add multilingual support to expand sentiment analysis to reviews in other languages.
- Explore more advanced deep learning models such as BERT and GPT.
- Model ensemble: Combine multiple classifiers to improve performance.
- Hyperparameter tuning: Use grid search to find the best model parameters.
- Implement a web interface to provide online sentiment analysis services.


