# Loan Application Approvement Project

The objective of this project is to classify whether a applicant will be approved for loan or not. In this project, we will use some classification algorithms and tune the hyperparameters to improve the performance. We will also utilize a pipeline to integrate some of preprocessing and modeling steps.

## Deployment

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://loan-application-approvement.onrender.com/)

## Data Set

Obtained from Kaggle: <a href="https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset/"> link </a>

This dataset contains personal characteristics and loan status of applicants respectively, with 11 predictor features and 1 target features. Of the 614 instances in the dataset, 68.73% (422) were positive class that approved to loan, and the rest (192) were negative class samples ending with unapproved.

The dataset consists of 4 numerical and 8 categorical features. The 'Loan_Status' feature was used as the class label. Each instance represent for one applicant.

## Insights

- Most of applicants without credit history was not approved.
- For each applicant, the higher income of applicant and/or coapplicant, the higher positive rate.
- The applicants with urban and semiurban tended to be approved more than the rural ones.
- Loan amount inversely proportional to loan status.

**Recommendation**

- Credit history and income are the most affected factors. Therefore, it might be able to set a threshold for income and combine with the sredit status for faster screening.
- With applicants from different regions, it might be able to advise them for more suitable loan packages.

## Cleaning

In our dataset, 5 features have null values, in which 4 can be filled with median for numerical features and most frequent class for categorical features. Except for credit score, we used classification algorithm to predict null cases with charateristics respectively.

## Preprocessing

1. Convert target variable to numerical format.
2. Fill null values using simple imputer.
3. Apply log transformation for numerical features.
4. Feature encoding for categorical and numerical features using one-hot encoding and standard scaler.
5. Split the dataset into 80% training, 20% tesing.

## Evaluation

The evaluation metrics that will be the main concern in this project are weighted F1 score. The reason behind this decision is due to the nature of the data that is quite small and has an imbalance class, thus the accuracy won’t represent the model's actual performance.

## Model Analysis

Amongst the classifier, Logistic Regression performed the best. The performance also seems to be okay on the new data since this model has an F1-score of around 78% on test set. Despite of its simplicity, this model has learned the rules underlying the data and also becomes less overfitted towards the training set after hyperparameter tuning has been done. Even so, there is some room for improvement towards this project, we can try another sampling strategy or more hyperparameters to be specific to have more improvement in the training and validation performances.

