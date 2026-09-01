import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from groq import Groq
from dotenv import load_dotenv

# Carga inicial intentando ubicar el archivo .env en la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def obtener_cliente_groq() -> Groq | None:
    """
    Inicializa el cliente de Groq utilizando la variable de entorno GROQ_API_KEY.

    Asegura la lectura directa desde os.getenv y realiza una verificación previa
    de la clave de autenticación antes de instanciar el cliente oficial de la SDK.

    Returns:
        Groq | None: Instancia del cliente de Groq si la API key está configurada, de lo contrario None.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def limpiar_texto(texto: str) -> str:
    """
    Limpia y normaliza un texto eliminando caracteres especiales y espacios redundantes.

    Args:
        texto (str): Cadena de texto a procesar.

    Returns:
        str: Texto normalizado en minúsculas, sin tildes/caracteres especiales y con espacios simples.
    """
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ0-9\s]', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()


def detectar_inconsistencia_posicion(hechos_clean: str, piezas_clean: str) -> int:
    """
    Evalúa contradicciones espaciales entre la narración del evento y los repuestos solicitados.

    Args:
        hechos_clean (str): Narración de los hechos en texto limpio.
        piezas_clean (str): Listado de piezas afectadas en texto limpio.

    Returns:
        int: 1 si detecta una contradicción espacial (frente vs. atrás), 0 en caso contrario.
    """
    frente_en_hechos = any(w in hechos_clean for w in ['frente', 'delante', 'delantera', 'delantero', 'trompa', 'capo'])
    atras_en_piezas = any(w in piezas_clean for w in ['trasero', 'trasera', 'atras', 'baul', 'stop', 'compuerta'])
    
    atras_en_hechos = any(w in hechos_clean for w in ['atras', 'atrás', 'trasera', 'trasero', 'reversa', 'cola'])
    frente_en_piezas = any(w in piezas_clean for w in ['delantero', 'delantera', 'frente', 'farola', 'capo', 'parachoque del'])
    
    if (frente_en_hechos and atras_en_piezas) or (atras_en_hechos and frente_en_piezas):
        return 1
    return 0


def analizar_siniestro_con_llm(
    version_hechos: str, 
    piezas_afectadas: str, 
    probabilidad_ml: float, 
    flag_rule: int
) -> Tuple[int, str]:
    """
    Utiliza un LLM (Llama 3 en Groq) para evaluar la coherencia semántico-espacial del siniestro
    y generar una justificación técnica en lenguaje natural para el auditor.

    Realiza una recarga de la variable de entorno GROQ_API_KEY, construye la interacción
    con respuesta estructurada JSON, captura excepciones detalladas de la API y garantiza
    mensajes explícitos en caso de errores de autenticación o configuración.

    Args:
        version_hechos (str): Descripción textual del accidente narrada por el usuario.
        piezas_afectadas (str): Listado de repuestos o piezas reportadas como dañadas.
        probabilidad_ml (float): Probabilidad calculada por el modelo de machine learning.
        flag_rule (int): Flag determinista de inconsistencia detectado por reglas de palabras clave.

    Returns:
        Tuple[int, str]: Una tupla con (flag_inconsistencia_llm, justificacion_tecnica).
    """
    # Forzar recarga del archivo .env si no se ha detectado la variable previamente
    if not os.getenv("GROQ_API_KEY"):
        load_dotenv(dotenv_path=ENV_PATH)

    client = obtener_cliente_groq()
    
    # Manejo de error claro si la clave no está configurada en el entorno
    if not client:
        return flag_rule, f"[ERROR CONFIG]: No se encontró la variable GROQ_API_KEY en {ENV_PATH}"

    prompt = f"""
    Eres un perito experto en auditoría de siniestros de automóviles para una aseguradora.
    Analiza la coherencia lógica y la física del impacto del siguiente reclamo:

    DATOS DEL RECLAMO:
    - Versión de los hechos: "{version_hechos}"
    - Piezas afectadas reclamadas: "{piezas_afectadas}"
    - Probabilidad de objeción calculada por ML: {probabilidad_ml:.1%}
    - Inconsistencia espacial detectada por reglas: {"SÍ" if flag_rule == 1 else "NO"}

    TAREAS:
    1. Evalúa si existe una incompatibilidad física o semántica real entre la zona del impacto narrada y la ubicación de los repuestos exigidos.
    2. Redacta una justificación técnica concisa (máximo 2 oraciones) orientada al auditor pericial.

    RESPONDE EXACTAMENTE EN EL SIGUIENTE FORMATO JSON:
    {{
        "inconsistencia_detectada": 1 o 0,
        "justificacion": "Explicación breve aquí."
    }}
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Responde únicamente en formato JSON válido."},
                {"role": "user", "content": prompt}
            ],
            model="qwen/qwen3.6-27b",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        res_json = json.loads(response.choices[0].message.content)
        flag_llm = int(res_json.get("inconsistencia_detectada", flag_rule))
        justificacion = res_json.get("justificacion", "Análisis completado exitosamente.")
        
        return flag_llm, justificacion

    except Exception as e:
        # Reportar el detalle real de la excepción de Groq o de la red en lugar de un mensaje genérico
        return flag_rule, f"[ERROR GROQ API]: {str(e)}"


def preparar_features_input(data: Dict[str, Any], tfidf_hechos: Any, tfidf_piezas: Any) -> np.ndarray:
    """
    Transforma el diccionario o petición de entrada en la matriz de características para el modelo.

    Extrae características numéricas, calcula la tasa de reemplazo de piezas, identifica
    inconsistencias espaciales y genera los vectores TF-IDF para la narración y los repuestos.

    Args:
        data (Dict[str, Any]): Diccionario con las llaves 'version_hechos', 'piezas_afectadas',
            'piezas_totales' y 'piezas_cambio'.
        tfidf_hechos (TfidfVectorizer): Vectorizador TF-IDF ajustado para la versión de los hechos.
        tfidf_piezas (TfidfVectorizer): Vectorizador TF-IDF ajustado para el listado de piezas.

    Returns:
        np.ndarray: Vector 2D (1, N_features) con el formato exacto requerido por el modelo.
    """
    version_hechos = data.get("version_hechos", "")
    raw_piezas = data.get("piezas_afectadas", "")

    # Normalizar piezas_afectadas a string y lista según el tipo recibido
    if isinstance(raw_piezas, list):
        piezas_afectadas_str = " | ".join(raw_piezas)
        cant_piezas_distintas = len(raw_piezas)
    else:
        piezas_afectadas_str = str(raw_piezas)
        cant_piezas_distintas = len([p for p in piezas_afectadas_str.split(',') if p.strip()])

    piezas_totales = data.get("piezas_totales", cant_piezas_distintas)
    piezas_cambio = data.get("piezas_cambio", cant_piezas_distintas)

    hechos_clean = limpiar_texto(version_hechos)
    piezas_clean = limpiar_texto(piezas_afectadas_str)

    ratio_piezas_cambio = piezas_cambio / (piezas_totales + 1e-5)
    len_hechos = len(hechos_clean.split())
    inconsistencia = detectar_inconsistencia_posicion(hechos_clean, piezas_clean)

    vec_num = np.array([[piezas_totales, piezas_cambio, ratio_piezas_cambio, len_hechos, cant_piezas_distintas, inconsistencia]])
    
    vec_tfidf_h = tfidf_hechos.transform([hechos_clean]).toarray()
    vec_tfidf_p = tfidf_piezas.transform([piezas_clean]).toarray()

    return np.hstack([vec_num, vec_tfidf_h, vec_tfidf_p])