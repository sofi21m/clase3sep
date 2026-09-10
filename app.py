import io
import requests
import platform
import numpy as np
import streamlit as st
from PIL import Image
from keras.models import load_model

# ===== CONFIGURACIÓN Y ESTILOS =====
st.set_page_config(
    page_title="Reconocimiento de Imágenes", 
    page_icon="📷", 
    layout="wide"
)

# Estilo personalizado para las tarjetas y las pestañas
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
        }
        .main-card {
            background-color: #1e222d;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
        }
    </style>
""", unsafe_allow_html=True)

# ===== ENCABEZADO =====
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("📷 Reconocimiento de Imágenes")
    st.caption("Clasificación en tiempo real basada en modelos de Teachable Machine")

with col_head2:
    st.write("")
    st.info(f"🐍 Python {platform.python_version()}")

st.divider()

# ===== MODELO Y SIDEBAR =====
@st.cache_resource
def cargar_modelo():
    return load_model('keras_model.h5')

model = cargar_modelo()

with st.sidebar:
    st.subheader("ℹ️ Panel de Información")
    try:
        st.image('OIG5.jpg', use_container_width=True)
    except:
        pass
    st.write("Usando un modelo entrenado en Teachable Machine para identificar mediante tu cámara, archivos locales o URLs.")

# ===== DIAGRAMACIÓN EN COLUMNAS ======
col_left, col_right = st.columns([1, 1], gap="large")

# COLUMNA IZQUIERDA: Carga y previsualización de imagen
with col_left:
    st.markdown("### 📥 Fuente de Imagen")
    
    tab1, tab2, tab3 = st.tabs(["📸 Cámara", "📁 Archivo", "🔗 Link URL"])

    img_file_buffer = None

    with tab1:
        camera_photo = st.camera_input("Toma una Foto", label_visibility="collapsed")
        if camera_photo:
            img_file_buffer = camera_photo.getvalue()

    with tab2:
        uploaded_file = st.file_uploader("Sube tu archivo de imagen", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img_file_buffer = uploaded_file.getvalue()

    with tab3:
        url_input = st.text_input("Enlace directo a la imagen:")
        if url_input:
            try:
                res = requests.get(url_input, timeout=10)
                if res.status_code == 200:
                    img_file_buffer = res.content
                else:
                    st.error("No se pudo descargar la imagen desde la URL.")
            except Exception as e:
                st.error(f"Error al conectar con la URL: {e}")

    # Muestra de la imagen seleccionada abajo de las pestañas
    if img_file_buffer is not None:
        st.markdown("---")
        st.markdown("#### Vista Previa")
        img_preview = Image.open(io.BytesIO(img_file_buffer)).convert("RGB")
        st.image(img_preview, use_container_width=True)

# COLUMNA DERECHA: Resultados del Modelo
with col_right:
    st.markdown("### 📊 Resultados de Clasificación")
    
    if img_file_buffer is not None:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        # Lectura y preparación
        img = Image.open(io.BytesIO(img_file_buffer)).convert("RGB")
        newsize = (224, 224)
        img = img.resize(newsize)
        img_array = np.array(img)

        # Normalización original
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        # Inferencia
        with st.spinner("Procesando predicción..."):
            prediction = model.predict(data)
            print(prediction)

        st.success("¡Análisis completado!")

        # TU LÓGICA DE CONDICIONALES EXACTA
        st.markdown("---")
        if prediction[0][0] > 0.5:
            st.subheader("👈 **Izquierda**")
            st.metric(label="Probabilidad", value=f"{prediction[0][0]*100:.2f}%")
            
        if prediction[0][1] > 0.5:
            st.subheader("👆 **Arriba**")
            st.metric(label="Probabilidad", value=f"{prediction[0][1]*100:.2f}%")
            
        if len(prediction[0]) > 2 and prediction[0][2] > 0.5:
            st.subheader("👉 **Derecha**")
            st.metric(label="Probabilidad", value=f"{prediction[0][2]*100:.2f}%")

    else:
        st.info("👈 Selecciona o captura una imagen en el panel de la izquierda para ver los resultados.")
