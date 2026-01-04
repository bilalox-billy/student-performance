import os
import sys
import numpy as np
import pandas as pd
import yaml

from src.exception import CustomException
import dill 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def read_yaml(file_path: str) -> dict:
    """
    Read YAML configuration file
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        dict: Configuration as dictionary
    """
    try:
        with open(file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            return config
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
        


def evaluate_models(X_train, y_train, X_test, y_test, models, param, grid_search_config=None):
    """
    Evaluate multiple models using GridSearchCV
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of model names and model objects
        param: Dictionary of model names and their hyperparameters
        grid_search_config: Configuration for GridSearchCV (cv, scoring, n_jobs, verbose)
        
    Returns:
        dict: Model performance report
    """
    try:
        # Default grid search configuration
        if grid_search_config is None:
            grid_search_config = {'cv': 3, 'scoring': 'r2', 'n_jobs': -1, 'verbose': 1}
        
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            para = param[model_name]

            # Perform grid search if parameters are provided
            if para:
                gs = GridSearchCV(
                    model, 
                    para, 
                    cv=grid_search_config.get('cv', 3),
                    scoring=grid_search_config.get('scoring', 'r2'),
                    n_jobs=grid_search_config.get('n_jobs', -1),
                    verbose=grid_search_config.get('verbose', 1)
                )
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
            
            model.fit(X_train, y_train) # train the model

            y_train_pred = model.predict(X_train) # predict on train data
            y_test_pred = model.predict(X_test) # predict on test data

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                'train_score': train_model_score,
                'test_score': test_model_score,
                'best_params': gs.best_params_ if para else {}
            }
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)







