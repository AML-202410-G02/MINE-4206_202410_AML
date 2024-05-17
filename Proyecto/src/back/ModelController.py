import Definitions

import os.path as osp
import pandas as pd

from joblib import load
from keras.models import load_model
from model.DataPreprocessor import DataPreprocessor


class ModelController:

    def __init__(self):
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models", "model.joblib")

        self.model = load_model(self.model_path)

        self.preprocessor = DataPreprocessor()

    def predict(self, data):
        print("predict ->")

        x = self.preprocessor.preprocess_data(df)
        """ Prediction probabilities """
        y_pred = self.model.predict(x, verbose=0)
        df = pd.DataFrame(self.get_categories(), columns=["Category"])
        df['Probability'] = y_pred.reshape(-1, 1)

        return df

    def get_categories(self):
        return ["Alto", "Medio", "Bajo"]
