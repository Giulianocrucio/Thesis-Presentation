import logging
import csv
import os
import sys

def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # set it to true to debug
        logger.propagate = False
    return logger

class CSVLogger:
    def __init__(self, filepath, is_classification):
        self.filepath = filepath
        self.is_classification = is_classification
        
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            if self.is_classification:
                writer.writerow(["train_acc", "train_loss", "val_acc", "val_loss", "test_acc", "test_loss"])
            else:
                writer.writerow(["train_mae", "train_mse", "val_mae", "val_mse", "test_mae", "test_mse"])

    def log_epoch(self, train_m1, train_m2, val_m1, val_m2):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([train_m1, train_m2, val_m1, val_m2, "", ""])

    def log_test(self, test_m1, test_m2):
        with open(self.filepath, 'r') as f:
            rows = list(csv.reader(f))
        
        if len(rows) > 1:
            rows[-1][4] = test_m1
            rows[-1][5] = test_m2
            
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
