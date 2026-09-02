import os

import requests
import streamlit as st

# Configuración de página amplia y título profesional
st.set_page_config(
    page_title="Subocol - Evaluación de Siniestros",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Variable de entorno para Docker/Local
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------
# ESTILOS CSS PERSONALIZADOS (UI/UX PRO)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Fondo general suave */
    .main {
        background-color: #f8fafc;
    }

    /* Header principal ejecutiva */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 0.4rem;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #93c5fd;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Cards de métricas personalizadas */
    .metric-card {
        background-color: white;
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }

    .card-delivered {
        border-left-color: #10b981 !important;
        background-color: #f0fdf4;
    }

    .card-objected {
        border-left-color: #ef4444 !important;
        background-color: #fef2f2;
    }

    /* Estilo para botones principales */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.65rem 1.5rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* Contenedores de texto e inputs */
    .stTextArea textarea, .stTextInput input {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# ENCABEZADO PRINCIPAL
# ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🛡️ Sistema de Evaluación de Siniestros</h1>
        <p>Subocol — Clasificador Inteligente de Riesgo de Reclamos (Reglas Espaciales + ML + Groq LLM)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# BARRA LATERAL (SIDEBAR)
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Panel de Control")
    st.markdown("---")

    st.subheader("Estado del Microservicio")
    try:
        health_res = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if health_res.status_code == 200:
            st.success("🟢 Backend API Conectado")
        else:
            st.warning("🟡 Backend respondió con estado inusual")
    except Exception:
        st.error("🔴 Backend no disponible")

    st.markdown("---")
    st.markdown("### 📊 Motor de Decisión")
    st.info(
        "**Inferencia Híbrida:**\n\n"
        "1. **Filtro Espacial:** Inconsistencias de posición.\n"
        "2. **ML Classifier:** Umbral optimizado de objeción.\n"
        "3. **GenAI Audit:** Análisis causa-raíz con Groq."
    )
    st.markdown("---")
    st.caption("Subocol Claims v2.0 • MLOps Production Ready")

# ---------------------------------------------------------
# FORMULARIO DE INGRESO DE DATOS
# ---------------------------------------------------------
st.subheader("📝 Formulario de Entrada del Reclamo")

with st.form("claim_form"):
    col_form_left, col_form_right = st.columns([2, 1])

    with col_form_left:
        version_hechos = st.text_area(
            "Versión de los hechos",
            placeholder="Ej: Vehículo impacta por la parte delantera contra un poste a baja velocidad.",
            height=130,
        )
        piezas_afectadas = st.text_input(
            "Piezas afectadas (separadas por coma)",
            placeholder="Ej: parachoque trasero, stop derecho",
        )

    with col_form_right:
        piezas_totales = st.number_input(
            "Piezas totales involucradas", min_value=0, max_value=50, value=5, step=1
        )
        piezas_cambio = st.number_input(
            "Piezas requeridas para cambio", min_value=0, max_value=50, value=2, step=1
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🚀 Evaluar Siniestro")

# ---------------------------------------------------------
# PROCESAMIENTO E INFERENCIA
# ---------------------------------------------------------
if submitted:
    if not version_hechos.strip() or not piezas_afectadas.strip():
        st.warning(
            "Por favor completa la versión de los hechos y las piezas afectadas."
        )
    else:
        payload = {
            "version_hechos": version_hechos,
            "piezas_afectadas": piezas_afectadas,
            "piezas_totales": int(piezas_totales),
            "piezas_cambio": int(piezas_cambio),
        }

        with st.spinner(
            "🔍 Evaluando siniestro en el motor híbrido (ML + GenAI Groq)..."
        ):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict", json=payload, timeout=15
                )

                if response.status_code == 200:
                    # Guardamos el resultado en el session_state para mantener la vista activa al interactuar con el PDF
                    st.session_state["evaluation_result"] = response.json()
                    st.session_state["last_payload"] = payload
                else:
                    st.error(
                        f"Error en la respuesta del servidor (HTTP {response.status_code}): {response.text}"
                    )

            except requests.exceptions.Timeout:
                st.error(
                    "Tiempo de espera agotado al conectar con el backend (Timeout de 15s)."
                )
            except Exception as e:
                st.error(f"No se pudo conectar con el microservicio backend: {e}")

# ---------------------------------------------------------
# PRESENTACIÓN DE RESULTADOS Y DESCARGA DE PDF
# ---------------------------------------------------------
if "evaluation_result" in st.session_state and st.session_state["evaluation_result"]:
    res = st.session_state["evaluation_result"]
    last_payload = st.session_state.get("last_payload", {})

    prediccion = res.get("prediccion", "N/A")
    prob = res.get("probabilidad_objetado", 0.0)
    flag = res.get("flag_inconsistencia_posicion", 0)
    umbral = res.get("umbral_aplicado", 0.20)
    justificacion = res.get("justificacion_llm", "Sin justificación disponible.")

    st.markdown("---")
    st.subheader("📊 Resultado del Análisis Pericial")

    # Columnas de Métricas Destacadas
    c1, c2, c3, c4 = st.columns(4)

    es_objetado = prediccion == "OBJETADO"
    card_class = "card-objected" if es_objetado else "card-delivered"
    badge_icon = "🔴" if es_objetado else "🟢"

    with c1:
        st.markdown(
            f"""
            <div class="metric-card {card_class}">
                <small style="color: #64748b; font-weight: 600;">DICTAMEN FINAL</small>
                <h3 style="margin: 0; color: {"#dc2626" if es_objetado else "#16a34a"}; font-weight: 700;">
                    {badge_icon} {prediccion}
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <small style="color: #64748b; font-weight: 600;">PROBABILIDAD OBJETADO</small>
                <h3 style="margin: 0; color: #1e293b; font-weight: 700;">
                    {prob * 100:.1f}%
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <small style="color: #64748b; font-weight: 600;">UMBRAL APLICADO</small>
                <h3 style="margin: 0; color: #1e293b; font-weight: 700;">
                    {umbral}
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        flag_txt = "SÍ (1)" if flag == 1 else "NO (0)"
        flag_color = "#dc2626" if flag == 1 else "#16a34a"
        st.markdown(
            f"""
            <div class="metric-card">
                <small style="color: #64748b; font-weight: 600;">INCONSISTENCIA ESPACIAL</small>
                <h3 style="margin: 0; color: {flag_color}; font-weight: 700;">
                    {flag_txt}
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sección de Auditoría GenAI (Groq)
    st.markdown("### 🧠 Auditoría de IA Generativa (Groq)")
    if es_objetado:
        st.warning(f"**Justificación Técnica:** {justificacion}")
    else:
        st.info(f"**Justificación Técnica:** {justificacion}")

    # ---------------------------------------------------------
    # SECCIÓN DE DESCARGA DEL REPORTE PERICIAL (PDF)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📄 Exportación de Dictamen Pericial")

    col_pdf_info, col_pdf_btn = st.columns([2, 1])

    with col_pdf_info:
        st.caption(
            "Genera y descarga el documento oficial en PDF con la auditoría completa "
            "del siniestro, las métricas del modelo y la justificación técnica de Groq."
        )

    with col_pdf_btn:
        pdf_payload = {
            "version_hechos": last_payload.get("version_hechos", ""),
            "piezas_afectadas": last_payload.get("piezas_afectadas", ""),
            "piezas_totales": int(last_payload.get("piezas_totales", 0)),
            "piezas_cambio": int(last_payload.get("piezas_cambio", 0)),
            "prediccion": prediccion,
            "probabilidad_objetado": float(prob),
            "flag_inconsistencia_posicion": int(flag),
            "justificacion_llm": justificacion,
        }

        try:
            pdf_res = requests.post(
                f"{BACKEND_URL}/generate-pdf", json=pdf_payload, timeout=10
            )
            if pdf_res.status_code == 200:
                st.download_button(
                    label="📥 Descargar Dictamen (PDF)",
                    data=pdf_res.content,
                    file_name="Dictamen_Siniestro_Subocol.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.error(f"Error al generar el PDF (Status {pdf_res.status_code})")
        except Exception as e:
            st.error(f"Error al conectar para generar el PDF: {e}")
