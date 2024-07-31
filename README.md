# Market Sentiment Analysis Using NLP

## Introduction
Welcome to our Market Sentiment Analysis project. This project aims to leverage Natural Language Processing (NLP) techniques to predict market sentiment from textual data, specifically tweets. The model we developed classifies texts into three categories: Bearish, Bullish, or Neutral. This classification is crucial for traders and investors who use sentiment analysis to inform their investment decisions. The project was completed by Luís Veloso and Riccardo Gurzu as part of the Text Mining 2023 course.

## Project Overview
The project is structured around several key phases: data exploration, data preprocessing, model training, evaluation, and prediction. Each phase plays a critical role in ensuring that the final model is both accurate and reliable.

### Data Exploration
We began with a thorough exploration of our dataset, which contains 9543 tweets, each labeled as Bearish (0), Bullish (1), or Neutral (2). Our initial analysis focused on understanding the distribution of these labels and the characteristics of the tweets. We verified data quality, ensuring there were no missing values, and used visualizations such as box plots and bar charts to examine the distribution of tweet lengths and the most frequently used words. This step was essential to gain insights into the dataset and identify any potential issues early on.

### Data Preprocessing
Preprocessing the data was a critical step in our project. We developed a comprehensive cleaning function to prepare the tweets for analysis. This function removed URLs, mentions, and hashtags, converted text to lowercase, and stripped out numerical and punctuation characters, leaving only alphabetical ones. We then tokenized the text and applied Part of Speech (POS) filtering to retain only nouns and adjectives, which are most likely to convey sentiment. Stop words were removed using NLTK’s built-in list, and we applied lemmatization to reduce words to their base forms. Optionally, stemming was applied using the Porter Stemming algorithm to further reduce words to their root forms. This rigorous preprocessing ensured that our text data was in the best possible format for analysis.

### Classification Models
To identify the best model for predicting market sentiment, we implemented a robust evaluation framework using GridSearchCV from Scikit-learn. This allowed us to perform cross-validation and hyperparameter tuning simultaneously. We experimented with various vectorizers (Bag of Words, TF-IDF, Word Embeddings) and classifiers (Linear Regression, Support Vector Machines, Naïve Bayes, Random Forest, Multi-Layer Perceptron, K-Nearest Neighbors, and LSTM). The feature variable was the text of the tweets, and the target variable was the corresponding sentiment label. We split the data into training and validation sets, ensuring the model was trained on a subset and evaluated on unseen data. After extensive testing, we found that the Linear Support Vector Classifier combined with TF-IDF Vectorizer provided the highest accuracy and F1 score.

### Evaluation and Results
The hyperparameter tuning process allowed us to select the best-performing model, considering various metrics such as accuracy, precision, recall, F1 score, and confusion matrix. Despite computational limitations that prevented a broader parameter search, our chosen model (Linear Support Vector Classifier with TfidfVectorizer) achieved excellent results. We then applied this model to new market behavior data, preprocessing the data similarly and generating predictions that were added to the original dataframe for further analysis.

### Conclusion
This project demonstrated the effectiveness of NLP techniques in sentiment analysis for market prediction. By carefully preprocessing the data and rigorously evaluating various models, we were able to develop a reliable and accurate classifier for market sentiment. This work has significant implications for the field of finance, providing a tool that can help traders and investors make more informed decisions based on textual data.

