# Predicting Viral Artificial-Intelligence Tweets

## Project Overview
This project predicts the virality of AI-related tweets using an imbalanced binary classification approach. By analyzing a massive corpus of 893,076 tweets (cleaned to 820,951 English tweets), this study identifies the content, structural, and temporal factors that drive high engagement. The project defines "viral" as the top 11.1% of tweets by total engagement (likes, retweets, and replies).

## Tools & Technologies
* **Programming & Environment:** Python, 
* **Analytical Libraries:** Scikit-Learn, Imbalanced-Learn, XGBoost, VADER Sentiment, Pandas
* **Techniques Used:** TF-IDF Vectorization, Random-Forest Feature Selection, SMOTE-Tomek Resampling, Stacking Ensembles
* **Visualization:** Power BI Desktop, 

## Methodology
1. **Feature Engineering:** Extracted 4,000 TF-IDF n-grams (1–2 grams) alongside 13 structured features (e.g., word count, media flags, posting hour) and VADER sentiment scores.
2. **Imbalance-Handling Pipeline:** Applied a leakage-free pipeline using Z-score standardization, Random-Forest Gini feature selection (keeping the top 300 features), and SMOTE-Tomek resampling to balance the training data.
3. **Model Evaluation:** Evaluated 10 baseline classifiers (including Random Forest, SVM, and XGBoost) against a proposed **Resampled Stacking Ensemble (VSE)**. The VSE stacked Logistic Regression, Random Forest, and XGBoost under a Logistic Regression meta-learner.
4. **Validation:** Assessed models across 9 metrics (focusing on F1, PR-AUC, and MCC for imbalance) using a held-out test fold and 10-fold cross-validation, supported by paired t-tests and Wilcoxon tests.
