import Definitions
import os.path as osp

class DataPreprocessor:
    def __init__(self):
        return 
    def preprocess_data(self, data):
        df = data.copy()
        # Paso 1: Seleccionar características
        df['Difference Bat %'] = df['Takeoff Bat %'] - df['Landing Bat %']

        selected_features = [
            'Pilot-in-Command', 'Above Sea Level (Meters)', 'Drone Type', 'Takeoff Bat %',
            'Takeoff Volts', 'Max Altitude (Meters)', 'Total Mileage (Kilometers)',
            'Air Seconds', 'Difference Bat %'
        ]
        df = df[selected_features]
        # Paso 2: Crear la nueva columna 'Battery Consumption'
        df['Battery Consumption'] = df['Difference Bat %'].apply(lambda x: 3 if x >= 66.67 else (1 if x <= 33.33 else 2))

        # Paso 3: Seleccionar características finales para el modelo
        final_features = [
            'Pilot-in-Command', 'Above Sea Level (Meters)', 'Drone Type', 'Takeoff Bat %', 'Takeoff Volts', 
            'Max Altitude (Meters)', 'Total Mileage (Kilometers)','Air Seconds', 'Battery Consumption'
        ]
        df = df[final_features]

        return df

    def transform(self, df):
        data = self.preprocess_data(df)
        return data


