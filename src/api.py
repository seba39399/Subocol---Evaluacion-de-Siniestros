import os
from pathlib import Path

import boto3
import joblib
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from src.processing import (
    analizar_siniestro_con_llm,
    detectar_inconsistencia_posicion,
    generar_pdf_dictamen,
    limpiar_texto,
    preparar_features_input,
)

# Inicializamos el servicio de la app o FastAPI
app = FastAPI(
    title="Reclamo de seguro - Subocol API",
    description=(
        "Microservicio de inferencia de machine learning para clasificación "
        "de riesgo de objeción en siniestros de vehículos. Utiliza un modelo "
        "HistGradientBoostingClassifier calibrado para maximizar el Recall."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Construye la ruta absoluta hacia el archivo .joblib
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_subocol_calibrado.joblib"

# Cargamos los artefactos adquiridos en la realización del notebook y entrenamiento del modelo
try:
    artefactos = joblib.load(MODEL_PATH)
    model = artefactos["model"]
    tfidf_hechos = artefactos["tfidf_hechos"]
    tfidf_piezas = artefactos["tfidf_piezas"]
    THRESHOLD = artefactos["threshold"]
except Exception as e:
    raise RuntimeError(f"Error al cargar los artefactos del modelo: {e}") from e


# Schemas de entrada y salida
class ClaimRequest(BaseModel):
    """
    Esquema de datos de entrada para la evaluación de un siniestro.
    """

    version_hechos: str = Field(
        ...,
        description="Descripción textual de cómo ocurrió el accidente.",
        json_schema_extra={"example": "Choque frontal a baja velocidad"},
    )
    piezas_afectadas: str = Field(
        ...,
        description="Listado de repuestos o piezas reportadas como dañadas.",
        json_schema_extra={"example": "parachoque trasero, stop derecho"},
    )
    piezas_totales: int = Field(
        ...,
        ge=0,
        description="Cantidad total de componentes involucrados en el evento.",
        json_schema_extra={"example": 5},
    )
    piezas_cambio: int = Field(
        ...,
        ge=0,
        description="Cantidad de piezas para las que se solicita sustitución.",
        json_schema_extra={"example": 2},
    )


class PDFReportRequest(ClaimRequest):
    """
    Esquema para solicitar la generación del PDF pericial (extiende ClaimRequest).
    """

    prediccion: str = Field(
        ..., description="Dictamen final ('OBJETADO' o 'ENTREGADO')."
    )
    probabilidad_objetado: float = Field(..., description="Probabilidad calculada.")
    flag_inconsistencia_posicion: int = Field(
        ..., description="Flag de inconsistencia espacial."
    )
    justificacion_llm: str = Field(
        ..., description="Justificación técnica del agente LLM."
    )


class PredictionResponse(BaseModel):
    """
    Esquema de respuesta de inferencia híbrida (ML + LLM).
    """

    prediccion: str = Field(
        ...,
        description="Resultado final: 'OBJETADO' o 'ENTREGADO'.",
        json_schema_extra={"example": "OBJETADO"},
    )
    probabilidad_objetado: float = Field(
        ...,
        description="Probabilidad continua calculada.",
        json_schema_extra={"example": 0.5343},
    )
    umbral_aplicado: float = Field(
        ...,
        description="Umbral de decisión calibrado (tau).",
        json_schema_extra={"example": 0.20},
    )
    flag_inconsistencia_posicion: int = Field(
        ...,
        description="Flag de contradicción espacial (0 o 1).",
        json_schema_extra={"example": 1},
    )
    justificacion_llm: str = Field(
        ...,
        description="Justificación técnica generada por el LLM.",
        json_schema_extra={
            "example": "Se detecta contradicción entre la zona del impacto narrada y la ubicación de las piezas."
        },
    )


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
    return HealthResponse(status="ok", threshold=THRESHOLD)


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

    # Preparar matriz de características y calcular probabilidad ML
    X_input = preparar_features_input(data_dict, tfidf_hechos, tfidf_piezas)
    prob_objetado = float(model.predict_proba(X_input)[0, 1])

    # Extraer flag de inconsistencia espacial por reglas deterministas
    hechos_clean = limpiar_texto(claim.version_hechos)
    piezas_clean = limpiar_texto(claim.piezas_afectadas)
    flag_regla = detectar_inconsistencia_posicion(hechos_clean, piezas_clean)

    # Análisis semántico y generación de explicabilidad con Groq (LLM)
    flag_llm, justificacion = analizar_siniestro_con_llm(
        version_hechos=claim.version_hechos,
        piezas_afectadas=claim.piezas_afectadas,
        probabilidad_ml=prob_objetado,
        flag_rule=flag_regla,
    )

    # Flag de inconsistencia física integrado (Regla de posición O Auditoría LLM)
    flag_inconsistencia_final = 1 if (flag_regla == 1 or flag_llm == 1) else 0

    # Lógica de Consenso Ajustada:
    # - Si existe inconsistencia física (regla o LLM), se objeta de inmediato.
    # - Si NO hay ninguna inconsistencia física, se requiere que la probabilidad de ML
    #   sea significativamente más alta (ej. >= 0.50) para objetar, evitando falsos positivos.
    if flag_inconsistencia_final == 1:
        es_objetado = True
    else:
        # Si la narración es coherente con las piezas, solo se objeta ante un riesgo alto de ML (50%+)
        UMBRAL_SIN_INCONSISTENCIA = 0.50
        es_objetado = prob_objetado >= UMBRAL_SIN_INCONSISTENCIA

    dictamen = "OBJETADO" if es_objetado else "ENTREGADO"

    return PredictionResponse(
        prediccion=dictamen,
        probabilidad_objetado=round(prob_objetado, 4),
        umbral_aplicado=THRESHOLD,
        flag_inconsistencia_posicion=flag_inconsistencia_final,  # Usamos el flag consolidado
        justificacion_llm=justificacion,
    )


@app.post("/generate-pdf", tags=["Reports"])
def generate_pdf_report(report_data: PDFReportRequest):
    """Genera dinámicamente un documento PDF con la auditoría pericial completa del siniestro.

    Args:
        report_data (PDFReportRequest): Datos completos de la evaluación e inferencia.

    Returns:
        Response: Stream binario `application/pdf` listo para ser descargado o renderizado.
    """
    try:
        # Invoca la función encargada de armar el buffer binario del PDF
        pdf_bytes = generar_pdf_dictamen(report_data.model_dump())

        # Intentamos subir el PDF generado a un bucket de S3
        s3_bucket = os.getenv("S3_BUCKET_NAME", "subocol-pdf-storage-615296308634")
        file_name = f"dictamen_siniestro_{os.urandom(4).hex()}.pdf"

        if s3_bucket:
            try:
                s3_client = boto3.client("s3")
                # Subimos los bytes directamente al bucket
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=file_name,
                    Body=pdf_bytes,
                    ContentType="application/pdf",
                )
            except (BotoCoreError, ClientError) as s3_err:
                # Opcional: Logueas el error de S3 pero dejas que el usuario descargue el PDF localmente
                print(f"Advertencia: No se pudo subir el PDF a S3: {s3_err}")

        headers = {
            "Content-Disposition": "attachment; filename=dictamen_siniestro_subocol.pdf"
        }

        return Response(
            content=pdf_bytes, media_type="application/pdf", headers=headers
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar el archivo PDF pericial: {str(e)}",
        ) from e
