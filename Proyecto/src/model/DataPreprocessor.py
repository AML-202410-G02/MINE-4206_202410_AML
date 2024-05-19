import Definitions
import os.path as osp
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessor:
    def __init__(self):
        self.train_path = osp.join(Definitions.ROOT_DIR, "resources/data", "train.csv")
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.numeric_scaler = StandardScaler()
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.categorical_encoder = OneHotEncoder(handle_unknown='ignore')

    def preprocess_data(self, data):
        df = data.copy()
        # Paso 1: Seleccionar características
        df['Difference Bat %'] = df['Takeoff Bat %'] - df['Landing Bat %']

        selected_features = [
            'Pilot-in-Command', 'Above Sea Level (Meters)', 'Drone Type', 'Takeoff Bat %',
            'Takeoff mAh', 'Takeoff Volts', 'Max Altitude (Meters)', 'Total Mileage (Kilometers)',
            'Air Seconds', 'Difference Bat %'
        ]
        df = df[selected_features]

        # Paso 2: Crear la nueva columna 'Battery Consumption'
        df['Battery Consumption'] = df['Difference Bat %'].apply(lambda x: 3 if x >= 66.67 else (1 if x <= 33.33 else 2))

        # Paso 3: Seleccionar características finales para el modelo
        final_features = [
            'Pilot-in-Command', 'Above Sea Level (Meters)', 'Drone Type', 'Takeoff Bat %',
            'Takeoff mAh', 'Takeoff Volts', 'Max Altitude (Meters)', 'Total Mileage (Kilometers)',
            'Air Seconds', 'Battery Consumption'
        ]
        df = df[final_features]

        return df

    def fit_numeric_transformers(self, df):
        numeric_features = [
            'Above Sea Level (Meters)', 'Takeoff Bat %', 'Takeoff mAh', 'Takeoff Volts',
            'Max Altitude (Meters)', 'Total Mileage (Kilometers)', 'Air Seconds'
        ]
        self.numeric_imputer.fit(df[numeric_features])
        self.numeric_scaler.fit(df[numeric_features])

    def transform_numeric(self, data):
        df = data.copy()
        numeric_features = [
            'Above Sea Level (Meters)', 'Takeoff Bat %', 'Takeoff mAh', 'Takeoff Volts',
            'Max Altitude (Meters)', 'Total Mileage (Kilometers)', 'Air Seconds'
        ]
        df[numeric_features] = self.numeric_imputer.transform(df[numeric_features])
        df[numeric_features] = self.numeric_scaler.transform(df[numeric_features])
        return df

    def fit_categorical_transformers(self, df):
        categorical_features = ['Pilot-in-Command', 'Drone Type']
        self.categorical_imputer.fit(df[categorical_features])
        df_imputed = self.categorical_imputer.transform(df[categorical_features])
        self.categorical_encoder.fit(df_imputed)

    def transform_categorical(self, df):
        categorical_features = ['Pilot-in-Command', 'Drone Type']
        df[categorical_features] = self.categorical_imputer.transform(df[categorical_features])
        encoded_cats = self.categorical_encoder.transform(df[categorical_features]).toarray()
        df = df.drop(columns=categorical_features)
        df = pd.concat([df, pd.DataFrame(encoded_cats, columns=self.categorical_encoder.get_feature_names_out(categorical_features))], axis=1)
        return df

    def transform(self, df):
        df = self.preprocess_data(df)
        df_train = pd.read_csv(self.train_path)
        self.fit_numeric_transformers(df_train)
        self.fit_categorical_transformers(df_train)
        df = self.transform_numeric(df)
        df = self.transform_categorical(df)
        return df


