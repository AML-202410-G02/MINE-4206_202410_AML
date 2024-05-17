import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None

    def preprocess_data(self, df):
        # Paso 1: Seleccionar características
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

    def create_preprocessor(self):
        # Definir los pasos de preprocesamiento
        numeric_features = [
            'Above Sea Level (Meters)', 'Takeoff Bat %', 'Takeoff mAh', 'Takeoff Volts',
            'Max Altitude (Meters)', 'Total Mileage (Kilometers)', 'Air Seconds'
        ]
        categorical_features = ['Pilot-in-Command', 'Drone Type']

        # Preprocesamiento para características numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocesamiento para características categóricas
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combinación de preprocesamientos
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

    def fit_transform(self, X):
        if self.preprocessor is None:
            self.create_preprocessor()
        return self.preprocessor.fit_transform(X)

    def transform(self, X):
        if self.preprocessor is None:
            self.create_preprocessor()
        return self.preprocessor.transform(X)
