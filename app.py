
import streamlit as st
import joblib
import pandas as pd

# Título de la aplicación
st.set_page_config(page_title='Predicción de Deserción Estudiantil', layout='centered')
st.title('📊 Predictor de Deserción Estudiantil')
st.markdown('Ingresa los datos del estudiante para predecir si desertará o no.')

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    try:
        model = joblib.load('modelo_desercion.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: El archivo 'modelo_desercion.pkl' no se encontró. Asegúrate de que el modelo esté guardado en el mismo directorio.")
        return None

model = load_model()

if model is not None:
    st.sidebar.header('Parámetros del Estudiante')

    # Función para recolectar las entradas del usuario
    def user_input_features():
        edad = st.sidebar.slider('Edad', 18, 30, 22)
        promedio = st.sidebar.slider('Promedio (0.0 - 5.0)', 2.0, 5.0, 3.5, 0.01)
        asistencia = st.sidebar.slider('Asistencia (0.0 - 1.0)', 0.5, 1.0, 0.8, 0.01)
        horas_estudio = st.sidebar.slider('Horas de Estudio por Semana', 5, 40, 20)
        uso_plataforma = st.sidebar.slider('Uso de Plataforma (%)', 0, 100, 50)
        materias_perdidas = st.sidebar.slider('Materias Perdidas', 0, 5, 1)
        nivel_socioeconomico = st.sidebar.selectbox('Nivel Socioeconómico (1=bajo, 5=alto)', [1, 2, 3, 4, 5], index=2)
        
        trabaja_map = {'No': 0, 'Sí': 1}
        trabaja_input = st.sidebar.selectbox('¿Trabaja?', list(trabaja_map.keys()))
        trabaja = trabaja_map[trabaja_input]
        
        acceso_internet_map = {'No': 0, 'Sí': 1}
        acceso_internet_input = st.sidebar.selectbox('¿Tiene Acceso a Internet?', list(acceso_internet_map.keys()))
        acceso_internet = acceso_internet_map[acceso_internet_input]

        data = {
            'edad': edad,
            'promedio': promedio,
            'asistencia': asistencia,
            'horas_estudio': horas_estudio,
            'uso_plataforma': uso_plataforma,
            'materias_perdidas': materias_perdidas,
            'nivel_socioeconomico': nivel_socioeconomico,
            'trabaja': trabaja,
            'acceso_internet': acceso_internet
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.subheader('Datos de Entrada del Estudiante')
    st.write(input_df)

    # Realizar la predicción
    if st.button('Predecir'):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Resultado de la Predicción')
        if prediction[0] == 1:
            st.error("¡ATENCIÓN! El estudiante **desertará** 🚨")
            st.write(f"Probabilidad de deserción: **{prediction_proba[0][1]:.2f}**")
        else:
            st.success("¡Excelente! El estudiante **NO desertará** ✅")
            st.write(f"Probabilidad de NO deserción: **{prediction_proba[0][0]:.2f}**")

    st.markdown("""
    --- 
    *Nota: Este modelo es solo un ejemplo y los resultados deben interpretarse con precaución. No debe ser usado como única base para decisiones críticas.*
    """)
