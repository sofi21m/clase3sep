import io
import requests
import platform
import numpy as np
import streamlit as st
from PIL import Image
from keras.models import load_model

# ===== CONFIGURACIÓN DE PÁGINA =====
st.set_page_config(
    page_title="Teachable Machine Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== ESTILOS CSS PERSONALIZADOS =====
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ===== CARGA DEL MODELO =====
@st.cache_resource
def get_model():
    try:
        return load_model('keras_model.h5', compile=False)
    except Exception as e:
        st.error(f"❌ Error al cargar 'keras_model.h5': {str(e)}")
        return None

# Cargar etiquetas del archivo labels.txt si existe
@st.cache_data
def get_labels():
    try:
        with open("labels.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    except Exception:
        # Clases por defecto si no existe labels.txt
        return ["Izquierda", "Arriba", "Derecha"]

model = get_model()
class_names = get_labels()

# ===== ENCABEZADO =====
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🧠 Classifier Studio")
    st.caption("Reconocimiento de imágenes entrenado en Teachable Machine")

with header_col2:
    st.write("") 
    st.status(f"Python {platform.python_version()}", state="complete")

st.divider()

# ===== BARRA LATERAL =====
with st.sidebar:
    st.subheader("ℹ️ Información")
    # Carga la imagen de portada si está disponible
    try:
        st.image('OIG5.jpg', use_container_width=True)
    except Exception:
        pass
    st.info("Usando un modelo Keras (.h5) para clasificar la orientación u objeto en tiempo real.")

if not model:
    st.stop()

# ===== SELECCIÓN DE ENTRADA (3 OPCIONES) =====
st.markdown("### 1. Selecciona la Fuente de Imagen")
tab1, tab2, tab3 = st.tabs(["📸  Usar Cámara", "📁  Subir Archivo", "🔗  Desde URL"])

image_input = None

with tab1:
    camera_file = st.camera_input("Capturar foto", label_visibility="collapsed")
    if camera_file:
        image_input = camera_file.getvalue()

with tab2:
    uploaded_file = st.file_uploader("Arrastra una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_input = uploaded_file.getvalue()

with tab3:
    url_input = st.text_input("Ingresa el enlace directo de la imagen:", placeholder="https://ejemplo.com/foto.jpg")
    if url_input:
        try:
            res = requests.get(url_input, timeout=10)
            if res.status_code == 200:
                image_input = res.content
            else:
                st.error("❌ No se pudo descargar la imagen de la URL.")
        except Exception as e:
            st.error(f"❌ Error al conectar: {str(e)}")

# ===== PROCESAMIENTO E INFERENCIA =====
if image_input:
    st.markdown("---")
    st.markdown("### 2. Resultado de la Clasificación")

    try:
        # 1. Abrir y asegurar que sea RGB
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
        
        # 2. Preprocesamiento estándar para Keras / Teachable Machine
        newsize = (224, 224)
        img_resized = img.resize(newsize)
        img_array = np.array(img_resized, dtype=np.float32)

        # Normalización [-1, 1]
        normalized_image_array = (img_array / 127.0) - 1.0
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # 3. Predicción
        with st.spinner("Clasificando con la red neuronal..."):
            prediction = model.predict(data)[0]

    except Exception as e:
        st.error(f"Error procesando la imagen: {str(e)}")
        st.stop()

    # Visualización en 2 columnas
    col_view, col_res = st.columns([1, 1], gap="large")

    with col_view:
        st.image(img, caption="Imagen analizada", use_container_width=True)

    with col_res:
        st.subheader("🎯 Predicción")

        # Detectar la clase con mayor probabilidad
        top_idx = int(np.argmax(prediction))
        top_conf = float(prediction[top_idx])
        top_label = class_names[top_idx] if top_idx < len(class_names) else f"Clase {top_idx}"

        # Métrica Principal
        st.metric(
            label="Clase Detectada", 
            value=top_label, 
            delta=f"Probabilidad: {top_conf*100:.2f}%"
        )

        st.markdown("---")
        st.write("**Probabilidad por categoría:**")

        # Mostrar barras para cada clase encontrada
        for i, conf in enumerate(prediction):
            label = class_names[i] if i < len(class_names) else f"Clase {i}"
            st.write(f"**{label}**: `{conf*100:.1f}%`")
            st.progress(float(conf))

st.markdown("---")
st.caption("Teachable Machine Studio • Powered by Keras & Streamlit") con Probabilidad: '+str( prediction[0][2]))


