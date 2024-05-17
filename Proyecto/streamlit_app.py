import streamlit as st
import pandas as pd
from sklearn.externals import joblib

# Cargar el modelo
#model = joblib.load('modelo.joblib')

# Función para hacer predicciones
#def predict(data):
#    prediction = model.predict(data)
#    return prediction

# UI
st.title('Predicción con Modelo Guardado')

# Subir archivo CSV
uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Datos del archivo CSV:")
    st.write(data.head())

    # Realizar predicciones si hay datos cargados
    if st.button('Hacer predicciones'):
        #predictions = predict(data)
        st.write("### Predicciones:")
        #st.write(predictions)


