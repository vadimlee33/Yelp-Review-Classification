# Yelp Review Classification (NLP)
## Project Overview
In this project we use NLP to analyze Yelp reviews.
Yelp is an app that provide a crowd-sourced review forum to business and services. The app is used publish crowd-sourced reviews about businesses.
Number of 'stars' indicate the business rating given by a customer, ranging from 1 to 5.
'Cool', 'Useful', 'Funny' indicate the number of cool votes given by other Yelp Users.
## Tools
### Data Visualization:
- Matplotlib
- Seaborn
- WorldCloud
- ProfileReport
### NLP Preprocesing:
- Natural Language Toolkit (NLTK)
- Stopwords
- TfidfVectorizer
- Counter
### Machine Learning Algorithms:
- Random Forest
- Bagging Classifier
- AdaBoost Classifier
- Voting Classifier
### ML Algorithms to handle imbalanced data:
- BalancedBaggingClassifier
- EasyEnsembleClassifier
- RUSBoostClassifier
- BalancedRandomForestClassifier
### Evaluation Metrics:
We are working with classification problem, thats why we use these metrics:
- Balanced Accuracy
- Roc-auc
- Classification Report
- Confusion Matrix

## Conclusion

Yelp review dataset is very unbalanced (81.7% positives reviews vs 18.3% negative reviews).

We used only preprocessed text with TF-IDF vectorizer in training models. In this study we've trained machine learning models with regularization (class_weight = 'balanced'), then implemented balancing methods and specific algorithms from scikit-learn.imbalanced library.

### Insights:
In the case of an unbalanced dataset, they do a better job of predicting models in which you can adjust parameter class_weight = 'balanced' (like Logistic Regression, SVC)

We've tried to apply strategies to balance the dataset, but the results according to machine learning model is not always be better than if we did not change the balance.

Under-sampling can help improve run time and storage problems by reducing the number of training data samples when the training data set is huge.

Under-sampling can discard potentially useful information which could be important for building rule classifiers. The sample chosen by random under-sampling may be a biased sample. And it will not be an accurate representation of the population. Thereby, resulting in inaccurate results with the actual test data set.

Over-sampling unlike under-sampling, this method leads to no information loss. Outperforms under sampling. But it increases the likelihood of overfitting since it replicates the minority class events.

We used specific models from scikit-learn imbalanced library. But they did not scored better than classic machine learning models.

In accordance with the results of the tested models, the best estimate was shown by Multinomial Naive Bayes with undersampling.
