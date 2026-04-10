import os
import sys
from src.exception import CustomException
from dataclasses import dataclass
from catboost import CatBoostRegressor
from src.logger import logging
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')   
class ModelTrainer:     
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }

            model_report: dict = {}

            for i in range(len(models)):
                model = list(models.values())[i]
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                model_report[list(models.keys())[i]] = test_model_score

            best_model_score = max(model_report.values())
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]
            best_model = models[best_model_name]

            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predictions = best_model.predict(X_test)
            r2_square = r2_score(y_test, predictions)
            return r2_square

        except Exception as e:
            logging.info("Exception occurred at Model Training")
            raise CustomException(e, sys)