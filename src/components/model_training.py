import os, sys
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_obj
from dataclasses import dataclass

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, classification_report


@dataclass
class ModelTrainingConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()
        
    def initiate_model_training(self, df_train, df_test):
        try:
            logging.info('Split data and setup parameters for building model')
            X_train, X_test = df_train[:,:-1], df_test[:,:-1]
            y_train, y_test = df_train[:,-1], df_test[:,-1]
            
            # Base pipeline
            pipe_base = Pipeline(steps=[
                ['classifier', LogisticRegression()]
            ], memory="cache_folder")
            
            # Grid of hyperparameters
            params_best = [
                {'classifier': [LogisticRegression(random_state=29)], 'classifier__penalty': ['l1', 'l2'], 'classifier__C': [0.1, 0.5, 1, 5]},
                {'classifier': [SGDClassifier(random_state=29)], 'classifier__penalty': ['l1', 'l2'], 'classifier__alpha': [0.1, 0.5, 1, 5], 'classifier__learning_rate': ['constant', 'adaptive'], 'classifier__eta0': [0.01, 0.1, 0.5]},
                {'classifier': [SVC(random_state=29)], 'classifier__kernel': ['rbf', 'poly'], 'classifier__degree': [2, 3, 4], 'classifier__C': [0.1, 0.5, 1, 5]},
                {'classifier': [KNeighborsClassifier()], 'classifier__n_neighbors': [2, 5, 10, 15], 'classifier__p': [1, 2]},
                {'classifier': [DecisionTreeClassifier(random_state=29)], 'classifier__max_depth': [2, 10, 50, 100], 'classifier__min_samples_split': [5, 10, 20], 'classifier__min_samples_leaf' : [5, 10, 20]},
                {'classifier': [RandomForestClassifier(random_state=29)], 'classifier__max_depth': [2, 5, 10], 'classifier__min_samples_split': [5, 10, 20], 'classifier__min_samples_leaf' : [5, 10, 20], 'classifier__n_estimators': [2, 5, 10]},
                {'classifier': [AdaBoostClassifier(random_state=29)], 'classifier__learning_rate': [0.1, 1], 'classifier__n_estimators': [5, 10, 20, 50]},
                {'classifier': [GradientBoostingClassifier(random_state=29)], 'classifier__learning_rate': [0.1, 1], 'classifier__min_samples_split': [5, 10, 20], 'classifier__min_samples_leaf' : [5, 10, 20], 'classifier__max_features' : ['auto', 'sqrt'], 'classifier__n_estimators': [2, 5, 10]}
            ]
            
            # Scoring metrics
            scoring = {
                'auc': 'roc_auc',
                'f1_weighted': 'f1_weighted',
            }

            # KFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=29)
            
            logging.info('Training and evaluating to find best model')
            best_grid = GridSearchCV(estimator=pipe_base, param_grid=params_best, 
                cv=cv, scoring=scoring, refit='f1_weighted', return_train_score=True)
            best_grid.fit(X_train, y_train)    
            logging.info('Training and evaluating completed')
            
            logging.info('Finding best score')
            best_classifier = best_grid.best_estimator_['classifier']
            y_pred = best_classifier.predict(X_test)
            best_score = f1_score(y_test, y_pred, average='weighted')
                
            threshold = 0.7
            if best_score < threshold:
                raise CustomException(f'No performance better than {threshold}')
            print(f'{str(best_classifier)[:-2]}: {best_score}')
            
            logging.info('Saving best model')
            save_obj(
                file_path=self.model_training_config.trained_model_path,
                obj=best_classifier
            )
        
        
        except Exception as e:
            raise CustomException(e, sys)