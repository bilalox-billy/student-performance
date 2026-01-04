from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer

from src.logger import logging

logging.info("The data ingestion step is started")
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    train_model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Model training completed. R2 score: {train_model_score}")