import Definitions

import os.path as osp
import pandas as pd

from joblib import load
from src.model.DataPreprocessor import DataPreprocessor


class ModelController:

    def __init__(self):
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models", "model.joblib")

        self.model = load(self.model_path)

        self.preprocessor = DataPreprocessor()

    def predict(self, data):
        print("predict ->")

        x = self.preprocessor.transform(data)
        print("Terminó proc")
        """ Prediction probabilities """
        y_pred = self.model.predict(x, verbose=0)
        df = pd.DataFrame(self.get_categories(), columns=["Category"])
        df['Probability'] = y_pred.reshape(-1, 1)

        return df

    def get_categories(self):
        return ["Alto", "Medio", "Bajo"]
