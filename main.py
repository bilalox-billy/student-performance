from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation

from src.logger import logging

logging.info("The data ingestion step is started")
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)