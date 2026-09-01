import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path

from src.processing import analizar_siniestro_con_llm, preparar_features_input, limpiar_texto, detectar_inconsistencia_posicion

# Inicializamos el servicio de la app o FastAPI
app = FastAPI(title="Reclamo de seguro - Subocol API",
              description=("Microservicio de inferencia de machine learning para clasificación "
                           "de riesgo de objeción en siniestros de vehículos. Utiliza un modelo "
                           "HistGradientBoostingClassifier calibrado para maximizar el Recall."
                           ), 
              version="2.0.0",
              docs_url="/docs", 
              redoc_url="/redoc"
              )

# Construye la ruta absoluta hacia el archivo .joblib
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_subocol_calibrado.joblib"

# Cargamos los artefactos adquiridos en la realización del notebook y entrenamiento del modelo
try:
    artefactos = joblib.load(MODEL_PATH)
    model = artefactos['model']
    tfidf_hechos = artefactos['tfidf_hechos']
    tfidf_piezas = artefactos['tfidf_piezas']
    THRESHOLD = artefactos['threshold']
except Exception as e:
    raise RuntimeError(f"Error al cargar los artefactos del modelo: {e}")

# Schemas de entrada y salida
class ClaimRequest(BaseModel):
    """
    Esquema de datos de entrada para la evaluación de un siniestro.
    """

    version_hechos: str = Field(
        ..., 
        description="Descripción textual de cómo ocurrió el accidente.",
        example="Vehículo impacta por la parte delantera contra poste."
    )
    piezas_afectadas: str = Field(
        ..., 
        description="Listado de repuestos o piezas reportadas como dañadas.",
        example="parachoque trasero, stop derecho"
    )
    piezas_totales: int = Field(
        ..., 
        ge=0, 
        description="Cantidad total de componentes involucrados en el evento.",
        example=5
    )
    piezas_cambio: int = Field(
        ..., 
        ge=0, 
        description="Cantidad de piezas para las que se solicita sustitución.",
        example=2
    )

class PredictionResponse(BaseModel):
    """
    Esquema de respuesta de inferencia híbrida (ML + LLM).
    """

    prediccion: str = Field(..., description="Resultado final: 'OBJETADO' o 'ENTREGADO'.", json_schema_extra={"example": "OBJETADO"})
    probabilidad_objetado: float = Field(..., description="Probabilidad continua calculada.", json_schema_extra={"example": 0.5343})
    umbral_aplicado: float = Field(..., description="Umbral de decisión calibrado (tau).", json_schema_extra={"example": 0.20})
    flag_inconsistencia_posicion: int = Field(..., description="Flag de contradicción espacial (0 o 1).", json_schema_extra={"example": 1})
    justificacion_llm: str = Field(..., description="Justificación técnica generada por el LLM.", json_schema_extra={"example": "Se detecta contradicción entre la zona del impacto narrada y la ubicación de las piezas."})    

class HealthResponse(BaseModel):
    """
    Esquema de respuesta del chequeo de salud.
    """
    
    status: str = Field(..., json_schema_extra={"example": "ok"})
    threshold: float = Field(..., json_schema_extra={"example": 0.20})

# Endpoints de la API

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """
    Verifica el estado operativo de la API y la disponibilidad de los servicios.

    Punto de enlace (health check) utilizado por orquestadores y sistemas de monitoreo
    (Liveness / Readiness probes en Docker/Kubernetes) para confirmar que el microservicio 
    está activo, receptivo y listo para procesar solicitudes de inferencia.

    Returns:
        HealthResponse: Objeto Pydantic con el diagnóstico operativo del servicio:
            - status (str): Estado operativo actual ('ok').
            - service (str): Nombre identificador del microservicio ('subocol-claims-auditor').
            - version (str): Versión semántica del API desplegada ('1.0.0').
    """
    return HealthResponse(
        status="ok",
        service="subocol-claims-auditor",
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_claim(claim: ClaimRequest):
    """
    Realiza la inferencia de riesgo de objeción para un reclamo de aseguradora.

    Combina la predicción probabilística de un modelo HistGradientBoostingClassifier,
    la detección de inconsistencias espaciales mediante reglas deterministas y la auditoría
    semántica guiada por un LLM (Llama 3 en Groq) para argumentar la decisión técnica final.

    Args:
        claim (ClaimRequest): Objeto Pydantic con los datos de entrada del siniestro:
            - version_hechos (str): Descripción textual del accidente.
            - piezas_afectadas (str): Listado de repuestos solicitados.
            - piezas_totales (int): Cantidad total de componentes involucrados.
            - piezas_cambio (int): Cantidad de piezas requeridas para cambio.
    
    Returns:
        PredictionResponse: Objeto Pydantic con los resultados de la inferencia:
            - prediccion (str): 'OBJETADO' si supera el umbral o 'ENTREGADO'.
            - probabilidad_objetado (float): Probabilidad continua entre 0.0 y 1.0.
            - umbral_aplicado (float): Umbral de decisión calibrado (ej. 0.20).
            - flag_inconsistencia_posicion (int): 1 si detecta contradicción espacial, 0 si no.
            - justificacion_llm (str): Explicación técnica en lenguaje natural generada por el agente GenAI.
    
    Raises:
        HTTPException: Si la entrada no cumple con el esquema validado por Pydantic.
    """

    data_dict = claim.model_dump()
    
    # Preparar la matriz de características usando processing.py y predicción del modelo
    X_input = preparar_features_input(data_dict, tfidf_hechos, tfidf_piezas)
    prob_objetado = float(model.predict_proba(X_input)[0, 1])
    
    # Extraer el flag de inconsistencia espacial directamente
    hechos_clean = limpiar_texto(claim.version_hechos)
    piezas_clean = limpiar_texto(claim.piezas_afectadas)
    flag_regla = detectar_inconsistencia_posicion(hechos_clean, piezas_clean)

    # Análisis Semántico y Generación de Explicabilidad con Groq (LLM)
    flag_llm, justificacion = analizar_siniestro_con_llm(
        version_hechos=claim.version_hechos,
        piezas_afectadas=claim.piezas_afectadas,
        probabilidad_ml=prob_objetado,
        flag_rule=flag_regla
    )
    
    # Predicción del modelo
    prob_objetado = float(model.predict_proba(X_input)[0, 1])
    
    # Consenso de decisión: Se objeta si ML supera el umbral O si alguna regla/LLM detecta inconsistencia física
    flag_inconsistencia_final = 1 if (flag_regla == 1 or flag_llm == 1) else 0
    es_objetado = (prob_objetado >= THRESHOLD) or (flag_inconsistencia_final == 1)
    
    dictamen = "OBJETADO" if es_objetado else "ENTREGADO"

    return PredictionResponse(
        prediccion=dictamen,
        probabilidad_objetado=round(prob_objetado, 4),
        umbral_aplicado=THRESHOLD,
        flag_inconsistencia_posicion=flag_regla,
        justificacion_llm=justificacion
    )

