import io
import requests
import platform
import numpy as np
import streamlit as st
from PIL import Image
from keras.models import load_model

# Configuración de página
st.set_page_config(page_title="Reconocimiento de Imágenes", page_icon="📷", layout="wide")

st.title("Reconocimiento de Imágenes")
st.write("Versión de Python:", platform.python_version())

# Cargar Modelo
@st.cache_resource
def cargar_modelo():
    return load_model('keras_model.h5')

model = cargar_modelo()

# Sidebar
with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usarlo en esta app para identificar")
    try:
        st.image('OIG5.jpg', width=250)
    except:
        pass

# Seleccionar la fuente de la imagen (Manteniendo tus 3 opciones de entrada)
tab1, tab2, tab3 = st.tabs(["📸 Toma una Foto", "📁 Subir Imagen", "🔗 Desde Link/URL"])

img_file_buffer = None

with tab1:
    camera_photo = st.camera_input("Toma una Foto")
    if camera_photo:
        img_file_buffer = camera_photo.getvalue()

with tab2:
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_file_buffer = uploaded_file.getvalue()

with tab3:
    url_input = st.text_input("Ingresa el link de la imagen:")
    if url_input:
        try:
            res = requests.get(url_input, timeout=10)
            if res.status_code == 200:
                img_file_buffer = res.content
            else:
                st.error("No se pudo descargar la imagen desde el link.")
        except Exception as e:
            st.error(f"Error al cargar la URL: {e}")

# Procesamiento con TU lógica original intacta
if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Lectura de la imagen
    img = Image.open(io.BytesIO(img_file_buffer)).convert("RGB")
    st.image(img, width=350, caption="Imagen seleccionada")

    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalización exacta a tu código
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Ejecución de la inferencia
    prediction = model.predict(data)
    print(prediction)

    st.markdown("---")
    st.subheader("Resultados:")

    # TU LÓGICA DE EVALUACIÓN ORIGINAL
    if prediction[0][0] > 0.5:
        st.header('Izquierda, con Probabilidad: ' + str(prediction[0][0]))
    if prediction[0][1] > 0.5:
        st.header('Arriba, con Probabilidad: ' + str(prediction[0][1]))
    if len(prediction[0]) > 2 and prediction[0][2] > 0.5:
        st.header('Derecha, con Probabilidad: ' + str(prediction[0][2]))
