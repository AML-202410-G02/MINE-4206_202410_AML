import streamlit as st
import pandas as pd
from sklearn.externals import joblib

from src.back.ModelController import ModelController

st.set_page_config(
    layout="centered", page_title="Make predictions", page_icon="🛩️"
)

### Support functions
def generate_progress_bar(value):
    return f'<div style="width: 100%; border: 1px solid #eee; border-radius: 10px;"><div style="width: {value * 100}%; height: 24px; background: linear-gradient(90deg, rgba(62,149,205,1) 0%, rgba(90,200,250,1) 100%); border-radius: 10px;"></div></div>'

ctrl = ModelController()
default_text = "Capital markets regulator Sebi has notified rules allowing private equity funds to sponsor a mutual fund house as they can bring in strategic guidance and talent to drive the growth of the industry. Currently, any entity that owns 40 per cent or more stake in a mutual fund is considered a sponsor and is required to fulfill the eligibility criteria. A private equity fund or a pooled investment vehicle or a pooled investment fund may also be permitted to sponsor mutual funds,” Sebi said in a notification adding that this is subject to certain conditions. Under the eligibility criteria for the sponsor of MF, Sebi said that sponsors need to adequately capitalise the asset management company (AMC) such that the positive liquid net worth of AMC should be at least Rs 150 crore. In addition, the capital contributed to the AMC would be locked in for 5 years, besides, the minimum sponsor stake of 40 per cent would also be locked in for five years, according to the notification issued on Monday. Further, the regulator said that Self Sponsored AMCs” can continue the mutual fund business. This is subject to AMCs fulfilling certain conditions. The move would give the original sponsor flexibility to voluntarily disassociate itself from the MF without needing to induct a new and eligible sponsor. In addition, the regulator has increased the role and accountability of the trustees in a bid to safeguard unitholders’ interests. Also, the regulator enhanced the accountability of the board of AMC. ADVERTISEMENT Further, the regulator said that a unitholder protection committee (UHPC) by the board of an AMC would be constituted. This is part of Sebi’s attempt to have an independent review mechanism for the decisions of AMC from the perspective of the unitholders’ interest across all products and services. To give this effect, the Securities and Exchange Board of India (Sebi) has amended mutual fund rules. "


# UI
st.title('Predicción con Modelo Guardado')

# Subir archivo CSV
uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Datos del archivo CSV:")
    st.write(df.head())

    # Realizar predicciones si hay datos cargados
    if st.button('Hacer predicciones'):
        try:
            result_df = ctrl.predict(uploaded_file)
            result_df['Ramp Bar'] = result_df['Probability'].apply(generate_progress_bar)
            result_df['Probability'] = result_df['Probability'] * 100

            st.success("✅ Done!")

            st.markdown(result_df.to_html(escape=False), unsafe_allow_html=True)
        except:
            st.error("Something happened", icon="🚨")


