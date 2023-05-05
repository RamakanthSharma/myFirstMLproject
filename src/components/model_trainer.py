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

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Adaboost Regressor": AdaBoostRegressor()
            }

            params = {
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt', 'log2', None],
                    'n_estimators':[8,16,32,64,128,256],
                },
                "Decision Tree":{
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2'],
                    'splitter':['best', 'random']
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'criterion':['friedman_mse', 'squared_error'],
                    'max_features':['sqrt', 'log2'],
                    'learning_rate':[0.1, 0.05, 0.01, 0.001],
                    'subsample':[0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
                    'n_estimators':[100, 150, 200, 250, 300, 500]
                },
                "Linear Regression":{
                    'fit_intercept':['True', 'False'],
                    'positive':['True', 'False']
                },
                "K-Neighbors Regressor":{
                    'n_neighbors':[5, 10, 15, 20, 25, 50, 100],
                    'weights':['uniform', 'distance'],
                    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size':[30, 50, 100],
                },
                "XGB Regressor":{
                    'n_estinators':[100, 500, 1000, 2000],
                    'max_depth':[2,5,10,15],
                    'subsample':[0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
                },
                "CatBoosting Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01, 0.05, 0.1],
                    'iterations':[30, 50, 100]
                },
                "Adaboost Regressor":{
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 150, 200, 250, 300, 500]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test,
                                              y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(
                best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")

            logging.info(f"{best_model_name} is the best model on both train and test datasets")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)