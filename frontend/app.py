import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Subocol - Evaluación de Siniestros",
    layout="centered"
)

# En Docker la URL será http://backend:8000, en local http://localhost:8000
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("Sistema de Evaluación de Siniestros")
st.subheader("Subocol - Clasificador de Riesgo de Reclamos")
st.markdown("---")

with st.form("claim_form"):
    version_hechos = st.text_area(
        "Versión de los hechos",
        placeholder="Ej: Vehículo impacta por la parte delantera contra un poste a baja velocidad."
    )
    piezas_afectadas = st.text_input(
        "Piezas afectadas (separadas por coma)",
        placeholder="Ej: parachoque trasero, stop derecho"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        piezas_totales = st.number_input("Piezas totales involucradas", min_value=0, value=5)
    with col2:
        piezas_cambio = st.number_input("Piezas requeridas para cambio", min_value=0, value=2)
        
    submitted = st.form_submit_button("Evaluar Siniestro")

if submitted:
    if not version_hechos or not piezas_afectadas:
        st.warning("Por favor completa la versión de los hechos y las piezas afectadas.")
    else:
        payload = {
            "version_hechos": version_hechos,
            "piezas_afectadas": piezas_afectadas,
            "piezas_totales": int(piezas_totales),
            "piezas_cambio": int(piezas_cambio)
        }
        
        try:
            # Timeout ampliado a 15s para dar margen a la inferencia del LLM en Groq
            response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=15)
            
            if response.status_code == 200:
                res = response.json()
                st.markdown("### Resultado del Análisis")
                
                prediccion = res["prediccion"]
                prob = res["probabilidad_objetado"]
                flag = res["flag_inconsistencia_posicion"]
                justificacion = res.get("justificacion_llm", "Sin justificación disponible.")
                
                # Banner principal del dictamen
                if prediccion == "OBJETADO":
                    st.error(f"**Dictamen:** {prediccion}")
                else:
                    st.success(f"**Dictamen:** {prediccion}")
                
                # Métricas numéricas
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Probabilidad Objetado", f"{prob * 100:.1f}%")
                col_b.metric("Umbral Aplicado", f"{res['umbral_aplicado']}")
                col_c.metric("Inconsistencia Espacial", "SÍ (1)" if flag == 1 else "NO (0)")

                st.markdown("---")
                st.markdown("### Audito de IA Generativa (Groq)")
                
                # Renderizado dinámico de la explicación según el dictamen
                if prediccion == "OBJETADO":
                    st.warning(f"**Justificación Técnica:** {justificacion}")
                else:
                    st.info(f"**Justificación Técnica:** {justificacion}")

            else:
                st.error(f"Error en la respuesta del servidor (HTTP {response.status_code}): {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("Tiempo de espera agotado al conectar con el backend (Timeout).")
        except Exception as e:
            st.error(f"No se pudo conectar con el microservicio backend: {e}")