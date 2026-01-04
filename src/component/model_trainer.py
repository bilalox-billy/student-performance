import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
     AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models, read_yaml


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        # Load configuration from YAML
        self.config = read_yaml('params.yaml')

    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test =(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Load model configurations from YAML
            logging.info("Loading model configurations from params.yaml")
            models_config = self.config['models']
            logging.info(f"Loading model configurations from params.yaml: {models_config}")
            
            # Initialize models based on configuration
            models = {}
            params = {}
            
            for model_name, model_config in models_config.items():
                if model_config.get('enabled', True):
                    # Initialize model instances
                    if model_name == "Random Forest":
                        models[model_name] = RandomForestRegressor()
                    elif model_name == "Decision Tree":
                        models[model_name] = DecisionTreeRegressor()
                    elif model_name == "Gradient Boosting":
                        models[model_name] = GradientBoostingRegressor()
                    elif model_name == "Linear Regression":
                        models[model_name] = LinearRegression()
                    elif model_name == "XGBRegressor":
                        models[model_name] = XGBRegressor()
                    elif model_name == "CatBoosting Regressor":
                        models[model_name] = CatBoostRegressor(verbose=False)
                    elif model_name == "AdaBoost Regressor":
                        models[model_name] = AdaBoostRegressor()
                    
                    # Get hyperparameters from config
                    params[model_name] = model_config.get('params', {})
            
            logging.info(f"Enabled models: {list(models.keys())}")
            
            # Get grid search configuration
            grid_search_config = self.config.get('grid_search', {})
            
            # Evaluate models
            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, 
                models, params, grid_search_config
            )
            logging.info(f"Models trained: {model_report}")

            ## To get best model score from dict
            test_scores = {model_name: scores['test_score'] for model_name, scores in model_report.items()}
            best_model_score = max(test_scores.values())
            
            ## To get best model name from dict
            best_model_name = [model_name for model_name, score in test_scores.items() if score == best_model_score][0]
            best_model = models[best_model_name]
            
            # Get minimum acceptable R2 score from config
            min_r2_score = self.config.get('model_trainer', {}).get('minimum_r2_score', 0.6)

            if best_model_score < min_r2_score:
                raise CustomException(f"No best model found with R2 score >= {min_r2_score}")

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")
            logging.info(f"Best parameters: {model_report[best_model_name].get('best_params', {})}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
    
        except Exception as e:
            logging.info("Exception occured at Model Trainer")
            raise CustomException(e, sys)







