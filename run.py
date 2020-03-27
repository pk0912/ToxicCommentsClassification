"""
Entry point for training and predicting
"""
import os
import pandas as pd
from utils.helpers import logger
from preprocessing import preprocess
from config import SIMPLE_PROCESSING_TYPE, COMPLEX_PROCESSING_TYPE
from settings import (
    KAGGLE_RAW_DATA_DIR,
    SIMPLE_PROCESSED_DATA_DIR,
    COMPLEX_PROCESSED_DATA_DIR,
)


def main():
    logger.info("Execution Started!!!")
    try:
        train_data = pd.read_csv(
            os.path.join(KAGGLE_RAW_DATA_DIR, "train.csv"), encoding="utf-8"
        )
        if SIMPLE_PROCESSING_TYPE:
            logger.info("Performing simple text processing.")
            train_data_simple = preprocess(train_data)
            if type(train_data_simple) == pd.core.frame.DataFrame:
                train_data_simple.to_csv(
                    os.path.join(SIMPLE_PROCESSED_DATA_DIR, "train_data_simple.csv"),
                    index=False,
                    encoding="utf-8",
                )
            else:
                logger.error("Unable to write simple processed data!!!")
        if COMPLEX_PROCESSING_TYPE:
            logger.info("Performing complex text processing.")
            train_data_complex = preprocess(train_data, preprocess_type="complex")
            if type(train_data_complex) == pd.core.frame.DataFrame:
                train_data_complex.to_csv(
                    os.path.join(COMPLEX_PROCESSED_DATA_DIR, "train_data_complex.csv"),
                    index=False,
                    encoding="utf-8",
                )
            else:
                logger.error("Unable to write complex processed data!!!")
    except Exception as e:
        logger.error("Exception in main method : {}".format(str(e)))
        return
    logger.info("Execution successfully completed.")


if __name__ == "__main__":
    main()
